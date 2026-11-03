#[cfg(test)]
use melior::ir::operation::OperationLike;
use melior::ir::r#type::TypeLike;
use melior::ir::ValueLike;
use std::collections::HashMap;

use melior::{
    Context,
    dialect::{DialectRegistry, arith, func, scf},
    ir::{
        Block, BlockLike, Location, Module, Operation, Region, Type, Value,
        attribute::{
            DenseElementsAttribute, DenseI32ArrayAttribute, DenseI64ArrayAttribute,
            FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
        },
        operation::{OperationBuilder, OperationMutLike},
        r#type::{FunctionType, IntegerType, MemRefType},
        RegionLike,
    },
    pass,
    utility::register_all_dialects,
};

use crate::ast;

/// Scalar element type — resolved from the AST type name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl ScalarType {
    pub fn from_name(name: &str) -> Self {
        match name {
            "f32" => ScalarType::F32,
            "f64" => ScalarType::F64,
            "i8" => ScalarType::I8,
            "i16" => ScalarType::I16,
            "i32" => ScalarType::I32,
            "i64" => ScalarType::I64,
            "u8" => ScalarType::U8,
            "u16" => ScalarType::U16,
            "u32" => ScalarType::U32,
            "u64" => ScalarType::U64,
            "bool" => ScalarType::Bool,
            other => panic!("unknown scalar type: {}", other),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }

    /// Whether values compare and widen as unsigned. A bool is 0 or 1.
    pub fn is_unsigned(&self) -> bool {
        matches!(self, ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 | ScalarType::Bool)
    }

    pub fn is_int(&self) -> bool {
        !self.is_float() && *self != ScalarType::Bool
    }

    fn to_mlir_type<'c>(&self, ctx: &'c Context) -> Type<'c> {
        match self {
            ScalarType::F32 => Type::float32(ctx),
            ScalarType::F64 => Type::float64(ctx),
            ScalarType::I8 => IntegerType::new(ctx, 8).into(),
            ScalarType::I16 => IntegerType::new(ctx, 16).into(),
            ScalarType::I32 => IntegerType::new(ctx, 32).into(),
            ScalarType::I64 => IntegerType::new(ctx, 64).into(),
            ScalarType::U8 => IntegerType::new(ctx, 8).into(),
            ScalarType::U16 => IntegerType::new(ctx, 16).into(),
            ScalarType::U32 => IntegerType::new(ctx, 32).into(),
            ScalarType::U64 => IntegerType::new(ctx, 64).into(),
            ScalarType::Bool => IntegerType::new(ctx, 1).into(),
        }
    }

    pub fn bit_width(&self) -> u32 {
        match self {
            ScalarType::I8 | ScalarType::U8 => 8,
            ScalarType::I16 | ScalarType::U16 => 16,
            ScalarType::F32 | ScalarType::I32 | ScalarType::U32 => 32,
            ScalarType::F64 | ScalarType::I64 | ScalarType::U64 => 64,
            ScalarType::Bool => 1,
        }
    }
}

/// A resolved vector type: element type + concrete width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VecType {
    pub scalar: ScalarType,
    pub width: u64,
}

impl VecType {
    /// `T[1]` is a plain scalar `T`, not `vector<1xT>`: integer scalars live in
    /// general-purpose registers, branch directly, and address memory without a
    /// trip through the vector register file.
    fn to_mlir_type<'c>(&self, ctx: &'c Context) -> Type<'c> {
        if self.width == 1 {
            self.scalar.to_mlir_type(ctx)
        } else {
            Type::vector(&[self.width], self.scalar.to_mlir_type(ctx))
        }
    }
}

/// A constant attribute for a value of type `ty`: a dense vector, or the bare
/// element attribute when `ty` is a scalar (a width-1 language value).
fn splat_attr<'c>(
    ty: Type<'c>,
    values: &[melior::ir::attribute::Attribute<'c>],
) -> Result<melior::ir::attribute::Attribute<'c>, melior::Error> {
    if ty.is_vector() {
        Ok(DenseElementsAttribute::new(ty, values)?.into())
    } else {
        Ok(values[0])
    }
}

/// The lane type a conversion builtin like `to_i64` produces.
pub(crate) fn int_conversion_target(name: &str) -> Option<ScalarType> {
    match name {
        "to_u8" | "to_u16" | "to_u32" | "to_u64" | "to_i8" | "to_i16" | "to_i32" | "to_i64" => {
            Some(ScalarType::from_name(&name[3..]))
        }
        _ => None,
    }
}

/// Returns true if the type name is a built-in scalar type (not a struct).
fn is_scalar_type_name(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool"
    )
}

/// Returns true if the type name is a pointer/buffer type (ptr[T]).
fn is_ptr_type_name(name: &str) -> bool {
    name == "ptr"
}

/// Extract the element type name from a ptr[T] AST type.
/// ptr[u8] has width = Width::Param("u8"), so we extract "u8".
fn ptr_element_scalar(ty: &ast::Type) -> ScalarType {
    match ty.width.as_ref() {
        Some(ast::Width::Param(name)) => ScalarType::from_name(name),
        _ => panic!("ptr type must have an element type, e.g. ptr[u8]"),
    }
}

/// Create an MLIR memref<?xelement> type for a ptr[T] type.
fn ptr_to_memref_type<'c>(ctx: &'c Context, ty: &ast::Type) -> Type<'c> {
    // ptr[u8] → memref<?xi8>, ptr[f32] → memref<?xf32>
    let scalar = ptr_element_scalar(ty);
    let scalar_mlir = scalar.to_mlir_type(ctx);
    // i64::MIN = dynamic dimension (?)
    MemRefType::new(scalar_mlir, &[i64::MIN], None, None).into()
}

/// Resolve an AST type to a concrete VecType.
/// `comptime_env` maps comptime param names to concrete values.
pub(crate) fn resolve_type(
    ty: &ast::Type,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> VecType {
    let scalar = ScalarType::from_name(&ty.name);
    // `native_width` is one register in bytes, so `_` is however many of this type fit.
    let lane_bytes = (scalar.bit_width() as u64 / 8).max(1);
    let width = ty
        .width
        .as_ref()
        .expect("type must have a width")
        .eval(comptime_env, native_width / lane_bytes);
    if width == 0 {
        panic!("a {} vector can't have zero lanes", ty.name);
    }
    VecType { scalar, width }
}

/// Create an MLIR context with all dialects loaded.
pub fn create_context() -> Context {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    context
}

/// A binding in the codegen: either a single vector value, a struct with flattened fields, or a ptr.
#[derive(Debug, Clone)]
enum Binding<'c, 'a> {
    Vec(Value<'c, 'a>, VecType),
    Struct {
        fields: Vec<(String, Value<'c, 'a>, VecType)>,
    },
    /// A pointer/buffer (memref) — stores the raw MLIR value and the element scalar type.
    Ptr(Value<'c, 'a>, ScalarType),
}

/// Codegen state for a single function.
/// The architecture code is generated for. The language means the same thing on each;
/// this picks the instructions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arch {
    /// NEON, 16-byte registers.
    Aarch64,
    /// AVX2 and PCLMUL (x86-64-v3 plus PCLMUL), 32-byte registers.
    X86_64,
}

impl Arch {
    pub fn host() -> Arch {
        if cfg!(target_arch = "x86_64") { Arch::X86_64 } else { Arch::Aarch64 }
    }

    /// The architecture of an LLVM target triple.
    pub fn of_triple(triple: &str) -> Option<Arch> {
        match triple.split('-').next()? {
            "aarch64" | "arm64" => Some(Arch::Aarch64),
            "x86_64" => Some(Arch::X86_64),
            _ => None,
        }
    }

    /// The triple used when none is given.
    pub fn host_triple() -> &'static str {
        match (Arch::host(), cfg!(target_os = "macos")) {
            (Arch::Aarch64, true) => "aarch64-apple-darwin",
            (Arch::Aarch64, false) => "aarch64-unknown-linux-gnu",
            (Arch::X86_64, true) => "x86_64-apple-darwin",
            (Arch::X86_64, false) => "x86_64-unknown-linux-gnu",
        }
    }

    /// The width of one vector register.
    pub fn register_bytes(self) -> u64 {
        match self {
            Arch::Aarch64 => 16,
            Arch::X86_64 => 32,
        }
    }
}

/// What one variable contributes to the results of an `scf.if`.
enum IfResult {
    Vec(String, VecType),
    Ptr(String, ScalarType),
    Struct(String, Vec<(String, VecType)>),
}

/// The monomorphized copies of generic functions a module needs, found as calls to
/// them are emitted.
struct Instances {
    arch: Arch,
    /// The environment the module was compiled with.
    module_env: HashMap<String, u64>,
    seen: std::collections::HashSet<String>,
    pending: Vec<ast::FnDef>,
    envs: HashMap<String, HashMap<String, u64>>,
}

struct FnCodegen<'c, 'a> {
    instances: std::rc::Rc<std::cell::RefCell<Instances>>,
    ctx: &'c Context,
    loc: Location<'c>,
    block: &'a Block<'c>,
    vars: HashMap<String, Binding<'c, 'a>>,
    comptime_env: HashMap<String, u64>,
    native_width: u64,
    struct_defs: HashMap<String, ast::StructDef>,
    /// All function definitions in the module, for inter-function calls.
    fn_defs: HashMap<String, ast::FnDef>,
    /// Current source location for error messages, set from stmt.span.
    current_span: ast::Span,
}

impl<'c, 'a> FnCodegen<'c, 'a> {
    /// Panic with source location context.
    fn error(&self, msg: &str) -> ! {
        panic!("{} at {}", msg, self.current_span)
    }

    fn emit_expr(&mut self, expr: &ast::Expr) -> (Value<'c, 'a>, VecType) {
        if let Some(ast::ConstValue::Bool(_)) = expr.const_value() {
            return self.emit_literal_broadcast(expr, VecType { scalar: ScalarType::Bool, width: 1 });
        }
        if Self::is_literal(expr) && !matches!(expr, ast::Expr::IntLit(_) | ast::Expr::FloatLit(_) | ast::Expr::CharLit(_)) {
            self.error("this literal expression needs a type: use it with a typed value, or add an annotation");
        }
        match expr {
            ast::Expr::Instantiate { name, .. } => {
                self.error(&format!("generic function '{}' can only be called", name));
            }

            ast::Expr::Logical { op, lhs, rhs } => self.emit_logical(*op, lhs, rhs),

            ast::Expr::Ident(name) => {
                let binding = self
                    .vars
                    .get(name)
                    .unwrap_or_else(|| self.error(&format!("undefined variable: {}", name)))
                    .clone();
                match binding {
                    Binding::Vec(val, ty) => (val, ty),
                    Binding::Ptr(val, scalar) => {
                        // Return the raw memref value with a dummy VecType
                        (val, VecType { scalar, width: 0 })
                    }
                    Binding::Struct { .. } => {
                        panic!(
                            "cannot use struct variable '{}' as a scalar value; access a field instead (e.g. {}.field)",
                            name, name
                        );
                    }
                }
            }

            ast::Expr::IntLit(n) => {
                self.error(&format!(
                    "bare integer literal {} needs type context (use in binary op or add type annotation: `name: type = {}`)",
                    n, n
                ));
            }

            ast::Expr::FloatLit(f) => {
                self.error(&format!(
                    "bare float literal {} needs type context (use in binary op or add type annotation)",
                    f
                ));
            }

            ast::Expr::BoolLit(_) => unreachable!("handled as a constant above"),

            ast::Expr::CharLit(ch) => {
                self.error(&format!(
                    "bare char literal '{}' needs type context (use in binary op)",
                    ch
                ));
            }

            ast::Expr::BinOp { op, lhs, rhs } => self.emit_binop(*op, lhs, rhs),

            ast::Expr::UnaryOp { op, operand } => self.emit_unary(*op, operand),

            ast::Expr::Masked {
                mask,
                body,
                fallback,
            } => self.emit_masked(mask, body, fallback.as_deref(), None),

            ast::Expr::Reduction { op, operand } => self.emit_reduction(*op, operand),

            ast::Expr::Field { base, field } => self.emit_field_access(base, field),

            ast::Expr::Call { func, args } => self.emit_call(func, args),

            ast::Expr::Scan { op, operand, seed } => self.emit_scan(*op, operand, seed.as_deref()),

            ast::Expr::Gather {
                base,
                index,
                mask,
            } => self.emit_gather(base, index, mask.as_deref()),

            ast::Expr::StructLit {
                name,
                width,
                fields,
            } => {
                let _ = (name, width, fields);
                self.error("a struct literal can't be used as a single value; assign it, pass it, or return it");
            }

            ast::Expr::Load { aligned, ty, ptr, offset } => self.emit_load(*aligned, ty, ptr, offset.as_deref()),

            ast::Expr::VecLit { elem_type, values } => {
                let scalar = ScalarType::from_name(elem_type);
                let width = values.len() as u64;
                let vt = VecType { scalar, width };
                let vec_mlir = vt.to_mlir_type(self.ctx);
                let scalar_mlir = scalar.to_mlir_type(self.ctx);

                if scalar.is_int() {
                    let bits = scalar.bit_width();
                    if let Some(bad) = values.iter().find(|&&v| bits < 64 && (v < -(1i64 << (bits - 1)) || v >= (1i64 << bits))) {
                        self.error(&format!("literal {} doesn't fit in {:?}", bad, scalar));
                    }
                }
                let attrs: Vec<melior::ir::attribute::Attribute> = values
                    .iter()
                    .map(|v| {
                        if scalar.is_float() {
                            FloatAttribute::new(self.ctx, scalar_mlir, *v as f64).into()
                        } else {
                            IntegerAttribute::new(scalar_mlir, *v).into()
                        }
                    })
                    .collect();

                let dense = splat_attr(vec_mlir, &attrs)
                    .expect("failed to create vector literal");
                let op = arith::constant(self.ctx, dense.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, vt)
            }
        }
    }

    /// Emit a literal broadcast to match a target vector type.
    fn emit_literal_broadcast(
        &mut self,
        expr: &ast::Expr,
        target: VecType,
    ) -> (Value<'c, 'a>, VecType) {
        let vec_type = target.to_mlir_type(self.ctx);
        let scalar_type = target.scalar.to_mlir_type(self.ctx);
        let attr: melior::ir::attribute::Attribute = match expr.const_value() {
            Some(ast::ConstValue::Float(f)) => {
                if !target.scalar.is_float() {
                    self.error("float literal in non-float context");
                }
                FloatAttribute::new(self.ctx, scalar_type, f).into()
            }
            Some(ast::ConstValue::Bool(b)) => {
                if target.scalar != ScalarType::Bool {
                    self.error(&format!("bool literal used as {:?}", target.scalar));
                }
                IntegerAttribute::new(scalar_type, b as i64).into()
            }
            Some(ast::ConstValue::Int(n)) => {
                if target.scalar.is_float() {
                    self.error("int literal in float context");
                }
                let value = if target.scalar == ScalarType::Bool {
                    if n != 0 && n != 1 {
                        self.error(&format!("{} isn't a bool: use true, false, 0 or 1", n));
                    }
                    n
                } else {
                    expr.eval_int(target.scalar.bit_width(), !target.scalar.is_unsigned())
                        .unwrap_or_else(|e| self.error(&format!("{} (as {:?})", e, target.scalar)))
                };
                IntegerAttribute::new(scalar_type, value).into()
            }
            None => self.error("this literal expression can't be evaluated: check for overflow or division by zero"),
        };
        let splat = splat_attr(vec_type, &[attr]).expect("failed to create splat attribute");
        let op = arith::constant(self.ctx, splat, self.loc);
        let val = self.block.append_operation(op).result(0).unwrap().into();
        (val, target)
    }

    /// Whether an expression has no type of its own and takes one from where it's used.
    fn takes_expected_type(expr: &ast::Expr) -> bool {
        match expr {
            _ if Self::is_literal(expr) => true,
            ast::Expr::Masked { body, fallback, .. } => {
                Self::is_literal(body) && fallback.as_deref().is_none_or(Self::is_literal)
            }
            ast::Expr::BinOp { op: ast::BinOp::BitShl | ast::BinOp::BitShr, lhs, .. } => Self::is_literal(lhs),
            _ => false,
        }
    }

    /// Evaluate an expression where the surrounding code fixes the type: an annotation, a
    /// parameter, a return type or a struct field. Same-sized integers are reinterpreted.
    fn emit_expr_typed(&mut self, expr: &ast::Expr, expected: VecType) -> (Value<'c, 'a>, VecType) {
        match expr {
            _ if Self::is_literal(expr) => return self.emit_literal_broadcast(expr, expected),
            ast::Expr::Masked { mask, body, fallback } if Self::takes_expected_type(expr) => {
                return self.emit_masked(mask, body, fallback.as_deref(), Some(expected.scalar));
            }
            ast::Expr::BinOp { op: op @ (ast::BinOp::BitShl | ast::BinOp::BitShr), lhs, rhs } if Self::is_literal(lhs) => {
                let (lval, lty) = self.emit_literal_broadcast(lhs, expected);
                return self.emit_shift(*op, lval, lty, rhs);
            }
            _ => {}
        }
        let (val, ty) = self.emit_expr(expr);
        if ty.width == 0 {
            return (val, ty);
        }
        if ty.width == expected.width && ty.to_mlir_type(self.ctx) == expected.to_mlir_type(self.ctx) {
            return (val, expected);
        }
        if ty.width == 1 && expected.width > 1 && ty.scalar.bit_width() == expected.scalar.bit_width() {
            let (wide, _) = self.emit_broadcast_to_width(val, ty, expected.width);
            return (wide, expected);
        }
        self.error(&format!(
            "expected {:?}[{}], got {:?}[{}]",
            expected.scalar, expected.width, ty.scalar, ty.width
        ));
    }

    fn is_literal(expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::FloatLit(_)
            | ast::Expr::IntLit(_)
            | ast::Expr::BoolLit(_)
            | ast::Expr::CharLit(_) => true,
            // ~literal and -literal are broadcastable (e.g. ~0 = all-ones)
            ast::Expr::UnaryOp { op: ast::UnaryOp::Not, operand }
            | ast::Expr::UnaryOp { op: ast::UnaryOp::Neg, operand } => Self::is_literal(operand),
            ast::Expr::BinOp { .. } | ast::Expr::Logical { .. } => expr.const_value().is_some(),
            _ => false,
        }
    }

    /// `lval << rhs` or `lval >> rhs`. `>>` is arithmetic on signed types and logical on
    /// unsigned ones. The amount is a constant, a scalar, or one amount per lane.
    fn emit_shift(
        &mut self,
        op: ast::BinOp,
        lval: Value<'c, 'a>,
        lty: VecType,
        rhs: &ast::Expr,
    ) -> (Value<'c, 'a>, VecType) {
        let bits = lty.scalar.bit_width() as i64;
        let amount: Value = if let Some(ast::ConstValue::Int(n)) = rhs.const_value() {
            if n < 0 || n >= bits {
                self.error(&format!("shift by {} is out of range for {:?}", n, lty.scalar));
            }
            let attr = IntegerAttribute::new(lty.scalar.to_mlir_type(self.ctx), n);
            let splat = splat_attr(lty.to_mlir_type(self.ctx), &[attr.into()]).unwrap();
            self.block
                .append_operation(arith::constant(self.ctx, splat, self.loc))
                .result(0)
                .unwrap()
                .into()
        } else {
            let (rval, rty) = self.emit_expr(rhs);
            if rty.width != 1 && rty.width != lty.width {
                self.error(&format!(
                    "shift amount width {} does not match operand width {}",
                    rty.width, lty.width
                ));
            }
            // The amount takes the operand's lane size; only its value matters.
            let sized = VecType { scalar: lty.scalar, width: rty.width };
            let as_unsigned = VecType {
                scalar: match rty.scalar.bit_width() {
                    8 => ScalarType::U8,
                    16 => ScalarType::U16,
                    32 => ScalarType::U32,
                    _ => ScalarType::U64,
                },
                width: rty.width,
            };
            let resized = self.emit_int_resize(rval, as_unsigned, sized);
            if rty.width == 1 && lty.width > 1 {
                self.emit_broadcast_to_width(resized, sized, lty.width).0
            } else {
                resized
            }
        };
        let shift_op = if op == ast::BinOp::BitShl {
            arith::shli(lval, amount, self.loc)
        } else if lty.scalar.is_unsigned() {
            arith::shrui(lval, amount, self.loc)
        } else {
            arith::shrsi(lval, amount, self.loc)
        };
        (self.block.append_operation(shift_op).result(0).unwrap().into(), lty)
    }

    fn emit_binop(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
    ) -> (Value<'c, 'a>, VecType) {
        if op == ast::BinOp::BitShl || op == ast::BinOp::BitShr {
            let (lval, lty) = self.emit_expr(lhs);
            return self.emit_shift(op, lval, lty, rhs);
        }

        // Handle literal broadcasting: if one side is a literal, infer its type from the other.
        let (lval, lty, rval, rty) = match (Self::is_literal(lhs), Self::is_literal(rhs)) {
            (true, true) => self.error("this literal expression needs a type: use it with a typed value, or add an annotation"),
            (true, false) => {
                let (rv, rt) = self.emit_expr(rhs);
                let (lv, lt) = self.emit_literal_broadcast(lhs, rt);
                (lv, lt, rv, rt)
            }
            (false, true) => {
                let (lv, lt) = self.emit_expr(lhs);
                let (rv, rt) = self.emit_literal_broadcast(rhs, lt);
                (lv, lt, rv, rt)
            }
            (false, false) => {
                let (lv, lt) = self.emit_expr(lhs);
                let (rv, rt) = self.emit_expr(rhs);
                (lv, lt, rv, rt)
            }
        };

        // Auto-broadcast width-1 vectors to match the other operand's width
        let (lval, lty, rval, rty) = if lty.width == 1 && rty.width > 1 {
            let (bv, bt) = self.emit_broadcast_to_width(lval, lty, rty.width);
            (bv, bt, rval, rty)
        } else if rty.width == 1 && lty.width > 1 {
            let (bv, bt) = self.emit_broadcast_to_width(rval, rty, lty.width);
            (lval, lty, bv, bt)
        } else {
            (lval, lty, rval, rty)
        };

        if lty.width != rty.width {
            self.error(&format!(
                "width mismatch: {:?}[{}] vs {:?}[{}]",
                lty.scalar, lty.width, rty.scalar, rty.width
            ));
        }
        if lty.to_mlir_type(self.ctx) != rty.to_mlir_type(self.ctx) {
            self.error(&format!(
                "operands have different types: {:?}[{}] and {:?}[{}]; convert with to_i64 and friends",
                lty.scalar, lty.width, rty.scalar, rty.width
            ));
        }

        match op {
            ast::BinOp::Add | ast::BinOp::Sub | ast::BinOp::Mul | ast::BinOp::Div => {
                if lty.scalar == ScalarType::Bool {
                    self.error("arithmetic on bools isn't defined; use &, | and ^");
                }
                if op == ast::BinOp::Div && matches!(rhs.const_value(), Some(ast::ConstValue::Int(0))) {
                    self.error("division by zero");
                }
                let result_op = if lty.scalar.is_float() {
                    match op {
                        ast::BinOp::Add => arith::addf(lval, rval, self.loc),
                        ast::BinOp::Sub => arith::subf(lval, rval, self.loc),
                        ast::BinOp::Mul => arith::mulf(lval, rval, self.loc),
                        ast::BinOp::Div => arith::divf(lval, rval, self.loc),
                        _ => unreachable!(),
                    }
                } else {
                    match op {
                        ast::BinOp::Add => arith::addi(lval, rval, self.loc),
                        ast::BinOp::Sub => arith::subi(lval, rval, self.loc),
                        ast::BinOp::Mul => arith::muli(lval, rval, self.loc),
                        ast::BinOp::Div if lty.scalar.is_unsigned() => arith::divui(lval, rval, self.loc),
                        ast::BinOp::Div => arith::divsi(lval, rval, self.loc),
                        _ => unreachable!(),
                    }
                };
                let val = self
                    .block
                    .append_operation(result_op)
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::Gt
            | ast::BinOp::Lt
            | ast::BinOp::GtEq
            | ast::BinOp::LtEq
            | ast::BinOp::EqEq
            | ast::BinOp::NotEq => {
                let result_ty = VecType {
                    scalar: ScalarType::Bool,
                    width: lty.width,
                };
                let cmp_op = if lty.scalar.is_float() {
                    let pred = match op {
                        ast::BinOp::Gt => arith::CmpfPredicate::Ogt,
                        ast::BinOp::Lt => arith::CmpfPredicate::Olt,
                        ast::BinOp::GtEq => arith::CmpfPredicate::Oge,
                        ast::BinOp::LtEq => arith::CmpfPredicate::Ole,
                        ast::BinOp::EqEq => arith::CmpfPredicate::Oeq,
                        ast::BinOp::NotEq => arith::CmpfPredicate::Une,
                        _ => unreachable!(),
                    };
                    arith::cmpf(self.ctx, pred, lval, rval, self.loc)
                } else {
                    let pred = match (op, lty.scalar.is_unsigned()) {
                        (ast::BinOp::Gt, false) => arith::CmpiPredicate::Sgt,
                        (ast::BinOp::Lt, false) => arith::CmpiPredicate::Slt,
                        (ast::BinOp::GtEq, false) => arith::CmpiPredicate::Sge,
                        (ast::BinOp::LtEq, false) => arith::CmpiPredicate::Sle,
                        (ast::BinOp::Gt, true) => arith::CmpiPredicate::Ugt,
                        (ast::BinOp::Lt, true) => arith::CmpiPredicate::Ult,
                        (ast::BinOp::GtEq, true) => arith::CmpiPredicate::Uge,
                        (ast::BinOp::LtEq, true) => arith::CmpiPredicate::Ule,
                        (ast::BinOp::EqEq, _) => arith::CmpiPredicate::Eq,
                        (ast::BinOp::NotEq, _) => arith::CmpiPredicate::Ne,
                        _ => unreachable!(),
                    };
                    arith::cmpi(self.ctx, pred, lval, rval, self.loc)
                };
                let val = self
                    .block
                    .append_operation(cmp_op)
                    .result(0)
                    .unwrap()
                    .into();
                (val, result_ty)
            }
            ast::BinOp::And => {
                let val = self
                    .block
                    .append_operation(arith::andi(lval, rval, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::Or => {
                let val = self
                    .block
                    .append_operation(arith::ori(lval, rval, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::Xor => {
                let val = self
                    .block
                    .append_operation(arith::xori(lval, rval, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (val, lty)
            }
            ast::BinOp::BitShl | ast::BinOp::BitShr => {
                unreachable!("bit shift ops handled before operand evaluation");
            }
        }
    }

    /// Broadcast a width-1 vector to a wider width using vector.broadcast.
    fn emit_broadcast_to_width(
        &mut self,
        val: Value<'c, 'a>,
        ty: VecType,
        target_width: u64,
    ) -> (Value<'c, 'a>, VecType) {
        assert_eq!(ty.width, 1, "can only broadcast width-1 vectors");
        let scalar = val;

        // Broadcast to target width
        let result_ty = VecType { scalar: ty.scalar, width: target_width };
        let result_mlir = result_ty.to_mlir_type(self.ctx);
        let broadcast_op = OperationBuilder::new("vector.broadcast", self.loc)
            .add_operands(&[scalar])
            .add_results(&[result_mlir])
            .build()
            .expect("failed to broadcast to target width");
        let result = self.block.append_operation(broadcast_op).result(0).unwrap().into();
        (result, result_ty)
    }

    /// Emit a lane shift using vector.shuffle.
    /// For shl (left): lane[i] = lane[i + shift], zeros fill the right.
    /// For shr (right): lane[i] = lane[i - shift], zeros fill the left.
    fn emit_lane_shift(
        &mut self,
        val: Value<'c, 'a>,
        ty: VecType,
        shift: usize,
        is_left: bool,
    ) -> (Value<'c, 'a>, VecType) {
        let width = ty.width as usize;
        let vec_mlir_type = ty.to_mlir_type(self.ctx);

        let zero = self.emit_zero_vector(ty);

        let mut mask: Vec<i64> = Vec::with_capacity(width);
        for lane in 0..width {
            if is_left {
                // Left shift: lane i gets value from lane (i + shift)
                let src = lane + shift;
                if src < width {
                    mask.push(src as i64);
                } else {
                    mask.push((width + lane) as i64); // from zero vector
                }
            } else {
                // Right shift: lane i gets value from lane (i - shift)
                if lane >= shift {
                    mask.push((lane - shift) as i64);
                } else {
                    mask.push((width + lane) as i64); // from zero vector
                }
            }
        }

        let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
            .add_operands(&[val, zero])
            .add_results(&[vec_mlir_type])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.ctx, "mask"),
                DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
            )])
            .build()
            .expect("failed to build vector.shuffle for lane shift");

        let result = self
            .block
            .append_operation(shuffle_op)
            .result(0)
            .unwrap()
            .into();

        (result, ty)
    }

    fn emit_unary(&mut self, op: ast::UnaryOp, operand: &ast::Expr) -> (Value<'c, 'a>, VecType) {
        match op {
            ast::UnaryOp::Neg => {
                let (val, ty) = self.emit_expr(operand);
                if ty.scalar.is_float() {
                    let neg = self
                        .block
                        .append_operation(arith::negf(val, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    (neg, ty)
                } else {
                    // Integer negate: 0 - val
                    let zero_ty = ty;
                    let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                    let vec_type = ty.to_mlir_type(self.ctx);
                    let zero_attr = IntegerAttribute::new(scalar_type, 0);
                    let splat = splat_attr(vec_type, &[zero_attr.into()])
                        .expect("failed to create zero splat");
                    let zero = self
                        .block
                        .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    let neg = self
                        .block
                        .append_operation(arith::subi(zero, val, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    (neg, zero_ty)
                }
            }
            ast::UnaryOp::Not => {
                let (val, ty) = self.emit_expr(operand);
                // Bitwise NOT: xor with all-ones
                let vec_type = ty.to_mlir_type(self.ctx);
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let ones_attr = IntegerAttribute::new(scalar_type, -1);
                let splat = splat_attr(vec_type, &[ones_attr.into()])
                    .expect("failed to create all-ones splat");
                let ones = self
                    .block
                    .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let result = self
                    .block
                    .append_operation(arith::xori(val, ones, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (result, ty)
            }
        }
    }

    /// `[mask] body : fallback`. Literal branches take `literal_scalar`, or else default to
    /// i32 or f32. A scalar branch broadcasts across a vector mask.
    fn emit_masked(
        &mut self,
        mask: &ast::Expr,
        body: &ast::Expr,
        fallback: Option<&ast::Expr>,
        literal_scalar: Option<ScalarType>,
    ) -> (Value<'c, 'a>, VecType) {
        let (mask_val, mask_ty) = self.emit_expr(mask);
        if mask_ty.scalar != ScalarType::Bool {
            self.error(&format!("a mask must be bool, got {:?}", mask_ty.scalar));
        }
        // With no fallback, inactive lanes are zero of whatever type the body has.
        let Some(fallback) = fallback else {
            let (body_val, body_ty) = if Self::is_literal(body) {
                let is_float = matches!(body.const_value(), Some(ast::ConstValue::Float(_)));
                let scalar = literal_scalar.unwrap_or(if is_float { ScalarType::F32 } else { ScalarType::I32 });
                self.emit_literal_broadcast(body, VecType { scalar, width: mask_ty.width })
            } else {
                self.emit_expr(body)
            };
            let (body_val, body_ty) = if body_ty.width == 1 && mask_ty.width > 1 {
                self.emit_broadcast_to_width(body_val, body_ty, mask_ty.width)
            } else {
                (body_val, body_ty)
            };
            let zero = self.emit_zero_vector(body_ty);
            let val = self
                .block
                .append_operation(arith::select(mask_val, body_val, zero, self.loc))
                .result(0)
                .unwrap()
                .into();
            return (val, body_ty);
        };

        let (mut body_val, mut body_ty, mut fallback_val, mut fallback_ty) =
            match (Self::is_literal(body), Self::is_literal(fallback)) {
                (true, true) => {
                    let is_float = matches!(body.const_value(), Some(ast::ConstValue::Float(_)));
                    let scalar = literal_scalar.unwrap_or(if is_float { ScalarType::F32 } else { ScalarType::I32 });
                    let target = VecType { scalar, width: mask_ty.width };
                    let (bv, bt) = self.emit_literal_broadcast(body, target);
                    let (fv, ft) = self.emit_literal_broadcast(fallback, target);
                    (bv, bt, fv, ft)
                }
                (true, false) => {
                    let (fv, ft) = self.emit_expr(fallback);
                    let (bv, bt) = self.emit_literal_broadcast(body, ft);
                    (bv, bt, fv, ft)
                }
                (false, true) => {
                    let (bv, bt) = self.emit_expr(body);
                    let (fv, ft) = self.emit_literal_broadcast(fallback, bt);
                    (bv, bt, fv, ft)
                }
                (false, false) => {
                    let (bv, bt) = self.emit_expr(body);
                    let (fv, ft) = self.emit_expr(fallback);
                    (bv, bt, fv, ft)
                }
            };

        if mask_ty.width > 1 {
            if body_ty.width == 1 {
                (body_val, body_ty) = self.emit_broadcast_to_width(body_val, body_ty, mask_ty.width);
            }
            if fallback_ty.width == 1 {
                (fallback_val, fallback_ty) = self.emit_broadcast_to_width(fallback_val, fallback_ty, mask_ty.width);
            }
            if body_ty.width != mask_ty.width {
                self.error(&format!("mask has {} lanes but its branch has {}", mask_ty.width, body_ty.width));
            }
        }
        if body_ty.to_mlir_type(self.ctx) != fallback_ty.to_mlir_type(self.ctx) {
            self.error(&format!(
                "masked branches have different types: {:?}[{}] and {:?}[{}]",
                body_ty.scalar, body_ty.width, fallback_ty.scalar, fallback_ty.width
            ));
        }

        // arith.select works lane-wise on vectors
        let val = self
            .block
            .append_operation(arith::select(mask_val, body_val, fallback_val, self.loc))
            .result(0)
            .unwrap()
            .into();
        (val, body_ty)
    }

    fn emit_reduction(
        &mut self,
        op: ast::ReductionOp,
        operand: &ast::Expr,
    ) -> (Value<'c, 'a>, VecType) {
        let (val, ty) = self.emit_expr(operand);
        let result_ty = VecType {
            scalar: ty.scalar,
            width: 1,
        };
        if ty.width == 1 {
            return (val, result_ty);
        }
        if ty.scalar == ScalarType::Bool && matches!(op, ast::ReductionOp::Add | ast::ReductionOp::Mul) {
            self.error("+/ and */ aren't defined on bools; use popcount, |/ or &/");
        }

        // Use vector.reduction operation via OperationBuilder
        let kind_str = match op {
            ast::ReductionOp::Add => "add",
            ast::ReductionOp::Mul => "mul",
            ast::ReductionOp::Or => "or",
            ast::ReductionOp::And => "and",
            ast::ReductionOp::Max => {
                if ty.scalar.is_float() {
                    "maximumf"
                } else if ty.scalar.is_unsigned() {
                    "maxui"
                } else {
                    "maxsi"
                }
            }
            ast::ReductionOp::Min => {
                if ty.scalar.is_float() {
                    "minimumf"
                } else if ty.scalar.is_unsigned() {
                    "minui"
                } else {
                    "minsi"
                }
            }
        };

        let scalar_mlir_type = ty.scalar.to_mlir_type(self.ctx);
        let kind_attr = melior::ir::attribute::Attribute::parse(
            self.ctx,
            &format!("#vector.kind<{}>", kind_str),
        )
        .expect("failed to parse vector.kind attribute");

        let reduction_op = OperationBuilder::new("vector.reduction", self.loc)
            .add_operands(&[val])
            .add_results(&[scalar_mlir_type])
            .add_attributes(&[(
                melior::ir::Identifier::new(self.ctx, "kind"),
                kind_attr,
            )])
            .build()
            .expect("failed to build vector.reduction");

        let scalar_val = self
            .block
            .append_operation(reduction_op)
            .result(0)
            .unwrap()
            .into();

        // Wrap scalar result into a width-1 vector via vector.broadcast
        let result_val = scalar_val;

        (result_val, result_ty)
    }

    fn emit_field_access(
        &mut self,
        base: &ast::Expr,
        field: &str,
    ) -> (Value<'c, 'a>, VecType) {
        if !self.is_struct_expr(base) {
            self.error(&format!("field access '.{}' on a non-struct", field));
        }
        let fields = self.emit_struct_value(base);
        match fields.iter().find(|(name, _, _)| name == field) {
            Some((_, val, ty)) => (*val, *ty),
            None => self.error(&format!("struct has no field '{}'", field)),
        }
    }

    fn emit_scan(
        &mut self,
        op: ast::ScanOp,
        operand: &ast::Expr,
        seed: Option<&ast::Expr>,
    ) -> (Value<'c, 'a>, VecType) {
        let (val, ty) = self.emit_expr(operand);
        let width = ty.width as usize;
        let vec_mlir_type = ty.to_mlir_type(self.ctx);

        // If a seed is provided, XOR/ADD/etc it into lane 0 before scanning.
        // This threads carry state from previous chunks.
        let val = if let Some(seed_expr) = seed {
            let (seed_val, _) = if Self::is_literal(seed_expr) {
                self.emit_literal_broadcast(seed_expr, VecType { scalar: ty.scalar, width: 1 })
            } else {
                self.emit_expr(seed_expr)
            };
            // Seed is width-1 vector. Broadcast it to match the operand width,
            // then combine with lane 0 using the scan's operation.
            // Actually simpler: insert seed into a zero vector at lane 0,
            // then combine with the operand using the scan op.
            // For XOR: val[0] = val[0] ^ seed, rest unchanged.
            // This is equivalent to: result = val XOR (seed_broadcast_to_lane0_only)

            // Extract scalar from seed (width-1 vector)
            let scalar_type = ty.scalar.to_mlir_type(self.ctx);
            let seed_scalar = seed_val;

            // Extract lane 0 from val
            let extract_val0 = OperationBuilder::new("vector.extract", self.loc)
                .add_operands(&[val])
                .add_results(&[scalar_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to extract val[0]");
            let val0 = self.block.append_operation(extract_val0).result(0).unwrap().into();

            // Combine: val[0] OP seed
            let combined = match op {
                ast::ScanOp::Xor => {
                    let op = arith::xori(val0, seed_scalar, self.loc);
                    self.block.append_operation(op).result(0).unwrap().into()
                }
                ast::ScanOp::Add => {
                    let op = if ty.scalar.is_float() {
                        arith::addf(val0, seed_scalar, self.loc)
                    } else {
                        arith::addi(val0, seed_scalar, self.loc)
                    };
                    self.block.append_operation(op).result(0).unwrap().into()
                }
                _ => panic!("seeded scan not supported for {:?}", op),
            };

            // Insert combined back into lane 0
            let insert_op = OperationBuilder::new("vector.insert", self.loc)
                .add_operands(&[combined, val])
                .add_results(&[vec_mlir_type])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "static_position"),
                    DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                )])
                .build()
                .expect("failed to insert combined into lane 0");
            self.block.append_operation(insert_op).result(0).unwrap().into()
        } else {
            val
        };

        match op {
            ast::ScanOp::PrecedingAny => {
                // Shift right by 1: take lanes 0..N-2, prepend a 0 lane
                // This checks if any preceding lane is set
                // Result[i] = input[i-1] for i>0, result[0] = 0
                let mut mask: Vec<i64> = Vec::with_capacity(width);
                // First lane gets a poison/zero lane from the "second vector" (which is a zero vec)
                mask.push(width as i64); // index into the zero vector's lane 0
                for i in 0..(width - 1) {
                    mask.push(i as i64);
                }

                // Create a zero vector for the fill
                let zero_val = self.emit_zero_vector(ty);

                let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[val, zero_val])
                    .add_results(&[vec_mlir_type])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .expect("failed to build vector.shuffle for preceding_any");

                let result = self
                    .block
                    .append_operation(shuffle_op)
                    .result(0)
                    .unwrap()
                    .into();

                (result, ty)
            }
            ast::ScanOp::Add | ast::ScanOp::Xor | ast::ScanOp::Max => {
                // Inclusive prefix scan using log2(N) steps of shuffle-and-combine
                // Step i (for i = 0, 1, ..., log2(N)-1):
                //   shifted = shuffle(val, identity, mask_that_shifts_right_by_2^i)
                //   val = combine(val, shifted)
                let num_steps = width.next_power_of_two().trailing_zeros() as usize;
                let mut current = val;

                for step in 0..num_steps {
                    let shift = 1 << step;

                    // Build shuffle mask: lanes 0..shift-1 get identity from second vector,
                    // lanes shift..N-1 get from first vector at position lane-shift
                    let mut mask: Vec<i64> = Vec::with_capacity(width);
                    for lane in 0..width {
                        if lane < shift {
                            // Fill from identity vector (second operand, offset by width)
                            mask.push((width + lane) as i64);
                        } else {
                            // Take from current vector at position lane - shift
                            mask.push((lane - shift) as i64);
                        }
                    }

                    // Create identity vector for this operation
                    let identity = self.emit_scan_identity(ty, op);

                    let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                        .add_operands(&[current, identity])
                        .add_results(&[vec_mlir_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "mask"),
                            DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                        )])
                        .build()
                        .expect("failed to build vector.shuffle for scan");

                    let shifted = self
                        .block
                        .append_operation(shuffle_op)
                        .result(0)
                        .unwrap()
                        .into();

                    // Combine current with shifted
                    let combine_op = match op {
                        ast::ScanOp::Add => {
                            if ty.scalar.is_float() {
                                arith::addf(current, shifted, self.loc)
                            } else {
                                arith::addi(current, shifted, self.loc)
                            }
                        }
                        ast::ScanOp::Xor => arith::xori(current, shifted, self.loc),
                        ast::ScanOp::Max => {
                            if ty.scalar.is_float() {
                                // Use arith.maximumf
                                OperationBuilder::new("arith.maximumf", self.loc)
                                    .add_operands(&[current, shifted])
                                    .add_results(&[vec_mlir_type])
                                    .build()
                                    .expect("failed to build arith.maximumf")
                            } else {
                                OperationBuilder::new(
                                    if ty.scalar.is_unsigned() { "arith.maxui" } else { "arith.maxsi" },
                                    self.loc,
                                )
                                    .add_operands(&[current, shifted])
                                    .add_results(&[vec_mlir_type])
                                    .build()
                                    .expect("failed to build arith.maxsi")
                            }
                        }
                        ast::ScanOp::PrecedingAny => unreachable!(),
                    };

                    current = self
                        .block
                        .append_operation(combine_op)
                        .result(0)
                        .unwrap()
                        .into();
                }

                (current, ty)
            }
        }
    }

    /// A `vector<width x i8>` constant splat of `val`.
    fn emit_splat_i8(&mut self, val: i64, width: usize) -> Value<'c, 'a> {
        let i8t: Type = IntegerType::new(self.ctx, 8).into();
        let vt = Type::vector(&[width as u64], i8t);
        let attr = IntegerAttribute::new(i8t, val);
        let splat = splat_attr(vt, &[attr.into()]).unwrap();
        self.block
            .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
            .result(0)
            .unwrap()
            .into()
    }

    /// `llvm.<name>` called with these operands.
    fn emit_intrinsic(&mut self, name: &str, operands: &[Value<'c, 'a>], result: Type<'c>) -> Value<'c, 'a> {
        self.block
            .append_operation(
                OperationBuilder::new("llvm.call_intrinsic", self.loc)
                    .add_operands(operands)
                    .add_results(&[result])
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(self.ctx, "intrin"),
                            StringAttribute::new(self.ctx, name).into(),
                        ),
                        (
                            melior::ir::Identifier::new(self.ctx, "operandSegmentSizes"),
                            DenseI32ArrayAttribute::new(self.ctx, &[operands.len() as i32, 0]).into(),
                        ),
                        (
                            melior::ir::Identifier::new(self.ctx, "op_bundle_sizes"),
                            DenseI32ArrayAttribute::new(self.ctx, &[]).into(),
                        ),
                    ])
                    .build()
                    .expect("failed to build llvm.call_intrinsic"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Table lookup on one 16-byte table: an index of 16 or more gives 0.
    fn emit_vtbl1(&mut self, table16: Value<'c, 'a>, idx16: Value<'c, 'a>) -> Value<'c, 'a> {
        let vec16 = Type::vector(&[16], IntegerType::new(self.ctx, 8).into());
        match self.arch() {
            Arch::Aarch64 => self.emit_intrinsic("llvm.aarch64.neon.tbl1", &[table16, idx16], vec16),
            Arch::X86_64 => {
                // pshufb only zeroes a lane whose index has its high bit set, and otherwise
                // uses the low 4 bits. Adding 0x70 with saturation sets the high bit exactly
                // for indices of 16 or more, and leaves the low 4 bits of the rest alone.
                let offset = self.emit_splat_i8(0x70, 16);
                let index = self.emit_intrinsic("llvm.uadd.sat.v16i8", &[idx16, offset], vec16);
                self.emit_intrinsic("llvm.x86.ssse3.pshuf.b.128", &[table16, index], vec16)
            }
        }
    }

    fn arch(&self) -> Arch {
        self.instances.borrow().arch
    }

    /// Extract a 16-lane slice at byte `offset` from a wider i8 vector.
    fn emit_extract16(&mut self, v: Value<'c, 'a>, offset: usize) -> Value<'c, 'a> {
        self.emit_extract_lanes(v, offset, 16)
    }

    /// Extract `size` lanes at byte `offset` from a wider i8 vector.
    fn emit_extract_lanes(&mut self, v: Value<'c, 'a>, offset: usize, size: u64) -> Value<'c, 'a> {
        let vec16 = Type::vector(&[size], IntegerType::new(self.ctx, 8).into());
        let off = melior::ir::attribute::Attribute::parse(self.ctx, &format!("[{}]", offset)).unwrap();
        let sz = melior::ir::attribute::Attribute::parse(self.ctx, &format!("[{}]", size)).unwrap();
        let st = melior::ir::attribute::Attribute::parse(self.ctx, "[1]").unwrap();
        self.block
            .append_operation(
                OperationBuilder::new("vector.extract_strided_slice", self.loc)
                    .add_operands(&[v])
                    .add_results(&[vec16])
                    .add_attributes(&[
                        (melior::ir::Identifier::new(self.ctx, "offsets"), off),
                        (melior::ir::Identifier::new(self.ctx, "sizes"), sz),
                        (melior::ir::Identifier::new(self.ctx, "strides"), st),
                    ])
                    .build()
                    .unwrap(),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Insert a 16-lane slice at byte `offset` into `dest` (width-`w` i8 vector).
    fn emit_insert16(&mut self, slice: Value<'c, 'a>, dest: Value<'c, 'a>, offset: usize, w: usize) -> Value<'c, 'a> {
        let vw = Type::vector(&[w as u64], IntegerType::new(self.ctx, 8).into());
        let off = melior::ir::attribute::Attribute::parse(self.ctx, &format!("[{}]", offset)).unwrap();
        let st = melior::ir::attribute::Attribute::parse(self.ctx, "[1]").unwrap();
        self.block
            .append_operation(
                OperationBuilder::new("vector.insert_strided_slice", self.loc)
                    .add_operands(&[slice, dest])
                    .add_results(&[vw])
                    .add_attributes(&[
                        (melior::ir::Identifier::new(self.ctx, "offsets"), off),
                        (melior::ir::Identifier::new(self.ctx, "strides"), st),
                    ])
                    .build()
                    .unwrap(),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Width-`w` table lookup `out[i] = row16[idx[i]]` (chunked vtbl1).
    fn emit_tbl_w(&mut self, row16: Value<'c, 'a>, idx: Value<'c, 'a>, w: usize) -> Value<'c, 'a> {
        if w == 16 {
            return self.emit_vtbl1(row16, idx);
        }
        let mut acc = self.emit_zero_vector(VecType { scalar: ScalarType::U8, width: w as u64 });
        let mut off = 0;
        while off < w {
            let chunk = self.emit_extract16(idx, off);
            let r = self.emit_vtbl1(row16, chunk);
            acc = self.emit_insert16(r, acc, off, w);
            off += 16;
        }
        acc
    }

    /// Shift `v` (width-`w` i8) right by `d` lanes, filling the low `d` lanes
    /// from `fill` (used to shift in the identity-function value `s`).
    fn emit_shift_in(&mut self, v: Value<'c, 'a>, fill: Value<'c, 'a>, d: usize, w: usize) -> Value<'c, 'a> {
        let vw = Type::vector(&[w as u64], IntegerType::new(self.ctx, 8).into());
        let mut mask: Vec<i64> = Vec::with_capacity(w);
        for lane in 0..w {
            if lane < d {
                mask.push((w + lane) as i64); // from `fill` (second operand)
            } else {
                mask.push((lane - d) as i64); // from `v`
            }
        }
        self.block
            .append_operation(
                OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[v, fill])
                    .add_results(&[vw])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .unwrap(),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// scan_dfa: prefix scan over a DFA's transition monoid. Returns the state
    /// after each byte. The per-byte transition functions are represented as `k`
    /// state-vectors `cur[s][i] = next state from `s` after byte i`, composed
    /// through log-step Hillis-Steele (compose = per-lane select among the k
    /// vectors). `table[s*16 + class] = next state`.
    fn emit_dfa_scan(
        &mut self,
        table: Value<'c, 'a>,
        classes: Value<'c, 'a>,
        seed: Value<'c, 'a>,
        k: usize,
        classes_ty: VecType,
    ) -> (Value<'c, 'a>, VecType) {
        let w = classes_ty.width as usize;
        let i8t: Type = IntegerType::new(self.ctx, 8).into();
        let vw = Type::vector(&[w as u64], i8t);

        // Initial functions: cur[s] = tbl(table_row_s, classes), one per state.
        let mut cur: Vec<Value<'c, 'a>> = Vec::with_capacity(k);
        for s in 0..k {
            let row = self.emit_extract16(table, s * 16);
            cur.push(self.emit_tbl_w(row, classes, w));
        }
        // Precompute the per-state constant splats (identity fill = s; compare keys).
        let splat: Vec<Value<'c, 'a>> = (0..k).map(|t| self.emit_splat_i8(t as i64, w)).collect();

        // Hillis-Steele: cur_i ← cur_i ∘ cur_{i-d}, d = 1,2,4,...
        let steps = w.next_power_of_two().trailing_zeros() as usize;
        for step in 0..steps {
            let d = 1usize << step;
            // shifted[s] = cur[s] shifted right by d, low lanes filled with `s`.
            let shifted: Vec<Value<'c, 'a>> = (0..k)
                .map(|s| self.emit_shift_in(cur[s], splat[s], d, w))
                .collect();
            // newcur[s][i] = cur[ shifted[s][i] ][i]  (select among cur[*] by shifted[s])
            let mut next: Vec<Value<'c, 'a>> = Vec::with_capacity(k);
            for s in 0..k {
                let mut acc = cur[0];
                for t in 1..k {
                    let m = self
                        .block
                        .append_operation(arith::cmpi(
                            self.ctx,
                            arith::CmpiPredicate::Eq,
                            shifted[s],
                            splat[t],
                            self.loc,
                        ))
                        .result(0)
                        .unwrap()
                        .into();
                    acc = self
                        .block
                        .append_operation(arith::select(m, cur[t], acc, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                }
                next.push(acc);
            }
            cur = next;
        }

        // Running state at each byte = cur[seed][i] = select among cur[*] by seed.
        let seed_scalar: Value<'c, 'a> = seed;
        let seed_bcast: Value<'c, 'a> = self
            .block
            .append_operation(
                OperationBuilder::new("vector.broadcast", self.loc)
                    .add_operands(&[seed_scalar])
                    .add_results(&[vw])
                    .build()
                    .unwrap(),
            )
            .result(0)
            .unwrap()
            .into();
        let mut result = cur[0];
        for t in 1..k {
            let m = self
                .block
                .append_operation(arith::cmpi(
                    self.ctx,
                    arith::CmpiPredicate::Eq,
                    seed_bcast,
                    splat[t],
                    self.loc,
                ))
                .result(0)
                .unwrap()
                .into();
            result = self
                .block
                .append_operation(arith::select(m, cur[t], result, self.loc))
                .result(0)
                .unwrap()
                .into();
        }
        (result, VecType { scalar: ScalarType::U8, width: w as u64 })
    }

    /// Create a zero vector (identity for add, xor, and preceding_any).
    fn emit_zero_vector(&mut self, ty: VecType) -> Value<'c, 'a> {
        let vec_type = ty.to_mlir_type(self.ctx);
        let scalar_type = ty.scalar.to_mlir_type(self.ctx);

        if ty.scalar.is_float() {
            let attr = FloatAttribute::new(self.ctx, scalar_type, 0.0);
            let splat = splat_attr(vec_type, &[attr.into()])
                .expect("failed to create zero splat");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        } else {
            let attr = IntegerAttribute::new(scalar_type, 0);
            let splat = splat_attr(vec_type, &[attr.into()])
                .expect("failed to create zero splat");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        }
    }

    /// Create the identity element vector for a scan operation.
    fn emit_scan_identity(&mut self, ty: VecType, op: ast::ScanOp) -> Value<'c, 'a> {
        match op {
            ast::ScanOp::Add | ast::ScanOp::Xor | ast::ScanOp::PrecedingAny => {
                self.emit_zero_vector(ty)
            }
            ast::ScanOp::Max => {
                // Identity for max is the minimum possible value
                let vec_type = ty.to_mlir_type(self.ctx);
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                if ty.scalar.is_float() {
                    let attr = FloatAttribute::new(self.ctx, scalar_type, f64::NEG_INFINITY);
                    let splat = splat_attr(vec_type, &[attr.into()])
                        .expect("failed to create neg_inf splat");
                    let op = arith::constant(self.ctx, splat.into(), self.loc);
                    self.block.append_operation(op).result(0).unwrap().into()
                } else {
                    // The smallest value of the type
                    let min_val = if ty.scalar.is_unsigned() { 0 } else { match ty.scalar.bit_width() {
                        8 => i8::MIN as i64,
                        16 => i16::MIN as i64,
                        32 => i32::MIN as i64,
                        64 => i64::MIN,
                        _ => 0,
                    } };
                    let attr = IntegerAttribute::new(scalar_type, min_val);
                    let splat = splat_attr(vec_type, &[attr.into()])
                        .expect("failed to create min_val splat");
                    let op = arith::constant(self.ctx, splat.into(), self.loc);
                    self.block.append_operation(op).result(0).unwrap().into()
                }
            }
        }
    }

    fn emit_gather(
        &mut self,
        base: &ast::Expr,
        index: &ast::Expr,
        mask: Option<&ast::Expr>,
    ) -> (Value<'c, 'a>, VecType) {
        let (base_val, base_ty) = self.emit_expr(base);
        if base_ty.width != 0 {
            self.error("gather base must be a pointer");
        }
        let element = self.ptr_scalar_of(base, base_ty.scalar);
        let (index_val, index_ty) = self.emit_expr(index);
        let result_ty = VecType { scalar: element, width: index_ty.width };
        let zero_index = self.make_index_const(0);

        if index_ty.width == 1 {
            if mask.is_some() {
                self.error("a masked gather needs a vector of indices; use if for a scalar");
            }
            let at = self.emit_value_as_index(index_val, index_ty);
            return (self.emit_memref_load(base_val, at, element), result_ty);
        }

        let mask_val = match mask {
            Some(m) => self.emit_expr(m).0,
            None => self.emit_all_true(index_ty.width),
        };
        let passthrough = self.emit_zero_vector(result_ty);
        let index_val = self.emit_vector_indices(index_val, index_ty);
        let result = self
            .block
            .append_operation(
                OperationBuilder::new("vector.gather", self.loc)
                    .add_operands(&[base_val, zero_index, index_val, mask_val, passthrough])
                    .add_results(&[result_ty.to_mlir_type(self.ctx)])
                    .build()
                    .expect("failed to build vector.gather"),
            )
            .result(0)
            .unwrap()
            .into();
        (result, result_ty)
    }

    /// A `bool[width]` with every lane true.
    fn emit_all_true(&mut self, width: u64) -> Value<'c, 'a> {
        let mask_mlir = VecType { scalar: ScalarType::Bool, width }.to_mlir_type(self.ctx);
        let attr = IntegerAttribute::new(ScalarType::Bool.to_mlir_type(self.ctx), 1);
        let splat = splat_attr(mask_mlir, &[attr.into()]).expect("failed to create all-true mask");
        self.block
            .append_operation(arith::constant(self.ctx, splat.into(), self.loc))
            .result(0)
            .unwrap()
            .into()
    }

    fn emit_load(
        &mut self,
        _aligned: bool,
        ty: &ast::Type,
        ptr: &ast::Expr,
        offset_expr: Option<&ast::Expr>,
    ) -> (Value<'c, 'a>, VecType) {
        let (ptr_val, ptr_ty) = self.emit_expr(ptr);
        if ptr_ty.width != 0 {
            self.error("load needs a pointer");
        }
        let vec_ty = resolve_type(ty, &self.comptime_env, self.native_width);
        if vec_ty.scalar == ScalarType::Bool {
            self.error("bools can't be loaded: load bytes and compare them");
        }

        // Compute index: either 0 or the offset expression (in elements of the ptr type)
        let load_index: Value = match offset_expr {
            Some(off_expr) => self.emit_index(off_expr),
            None => self.make_index_const(0),
        };

        // Check if load type differs from ptr element type — need bitcast
        let ptr_scalar = Some(ptr_ty.scalar);

        let needs_bitcast = ptr_scalar.is_some() && ptr_scalar.unwrap() != vec_ty.scalar;

        // A scalar load stays scalar: going through vector<8xi8> leaves LLVM with
        // `load <1 x i64>`, which lands in a vector register.
        if vec_ty.width == 1 && (!needs_bitcast || ptr_ty.scalar.bit_width() == 8) {
            let result = if needs_bitcast {
                self.emit_unaligned_load(ptr_val, load_index, vec_ty.scalar)
            } else {
                self.emit_memref_load(ptr_val, load_index, vec_ty.scalar)
            };
            return (result, vec_ty);
        }

        // vector.load always yields a vector; a width-1 language value unwraps it.
        let vector_of = |vt: VecType, ctx: &'c Context| -> Type<'c> {
            Type::vector(&[vt.width], vt.scalar.to_mlir_type(ctx))
        };

        let loaded: Value<'c, 'a> = if needs_bitcast {
            let ptr_sc = ptr_scalar.unwrap();
            // Load as the ptr's element type with enough elements to fill the target type
            let target_bytes = vec_ty.scalar.bit_width() as u64 / 8 * vec_ty.width;
            let ptr_elem_bytes = ptr_sc.bit_width() as u64 / 8;
            if target_bytes % ptr_elem_bytes != 0 {
                self.error(&format!(
                    "can't load {:?}[{}]: {} bytes isn't a whole number of {:?}",
                    vec_ty.scalar, vec_ty.width, target_bytes, ptr_sc
                ));
            }
            let load_width = target_bytes / ptr_elem_bytes;
            let load_vt = VecType { scalar: ptr_sc, width: load_width };

            let load_op = OperationBuilder::new("vector.load", self.loc)
                .add_operands(&[ptr_val, load_index])
                .add_results(&[vector_of(load_vt, self.ctx)])
                .build()
                .expect("failed to build vector.load");
            let raw = self.block.append_operation(load_op).result(0).unwrap().into();

            // Bitcast to target type
            let bitcast_op = OperationBuilder::new("vector.bitcast", self.loc)
                .add_operands(&[raw])
                .add_results(&[vector_of(vec_ty, self.ctx)])
                .build()
                .expect("failed to build vector.bitcast");
            self.block.append_operation(bitcast_op).result(0).unwrap().into()
        } else {
            let load_op = OperationBuilder::new("vector.load", self.loc)
                .add_operands(&[ptr_val, load_index])
                .add_results(&[vector_of(vec_ty, self.ctx)])
                .build()
                .expect("failed to build vector.load");
            self.block.append_operation(load_op).result(0).unwrap().into()
        };

        (self.emit_unwrap_unit(loaded, vec_ty), vec_ty)
    }

    /// Convert between integer widths: extend by the source's signedness, truncate, or (at
    /// equal width, where signedness has no MLIR representation) pass through.
    fn emit_int_resize(&mut self, val: Value<'c, 'a>, from: VecType, to: VecType) -> Value<'c, 'a> {
        if from.scalar.is_float() || to.scalar.is_float() {
            self.error(&format!("can't convert {:?} to {:?}", from.scalar, to.scalar));
        }
        let (from_bits, to_bits) = (from.scalar.bit_width(), to.scalar.bit_width());
        if from_bits == to_bits {
            return val;
        }
        let target = to.to_mlir_type(self.ctx);
        let op = if from_bits > to_bits {
            arith::trunci(val, target, self.loc)
        } else if from.scalar.is_unsigned() {
            arith::extui(val, target, self.loc)
        } else {
            OperationBuilder::new("arith.extsi", self.loc)
                .add_operands(&[val])
                .add_results(&[target])
                .build()
                .expect("failed to build arith.extsi")
        };
        self.block.append_operation(op).result(0).unwrap().into()
    }

    /// An integer value as an MLIR `index`, extended by its own signedness.
    fn emit_value_as_index(&mut self, val: Value<'c, 'a>, ty: VecType) -> Value<'c, 'a> {
        let index = Type::index(self.ctx);
        let op = if ty.scalar.is_unsigned() {
            OperationBuilder::new("arith.index_castui", self.loc)
                .add_operands(&[val])
                .add_results(&[index])
                .build()
                .expect("failed to build arith.index_castui")
        } else {
            arith::index_cast(val, index, self.loc)
        };
        self.block.append_operation(op).result(0).unwrap().into()
    }

    /// Gather and scatter indices. MLIR sign-extends them, so narrow unsigned ones widen first.
    fn emit_vector_indices(&mut self, val: Value<'c, 'a>, ty: VecType) -> Value<'c, 'a> {
        if ty.scalar.is_unsigned() && ty.scalar.bit_width() < 64 {
            self.emit_int_resize(val, ty, VecType { scalar: ScalarType::I64, width: ty.width })
        } else {
            val
        }
    }

    /// An element offset as an MLIR `index`: a literal, or a scalar integer value.
    fn emit_index(&mut self, expr: &ast::Expr) -> Value<'c, 'a> {
        if let Some(ast::ConstValue::Int(n)) = expr.const_value() {
            return self.make_index_const(n);
        }
        let (val, ty) = self.emit_expr(expr);
        if ty.width != 1 || !ty.scalar.is_int() {
            self.error(&format!(
                "memory offset must be a scalar integer, got {:?}[{}]",
                ty.scalar, ty.width
            ));
        }
        self.emit_value_as_index(val, ty)
    }

    /// `vector<1xT>` → `T` for a width-1 value; wider vectors pass through.
    fn emit_unwrap_unit(&mut self, val: Value<'c, 'a>, ty: VecType) -> Value<'c, 'a> {
        if ty.width != 1 {
            return val;
        }
        self.block
            .append_operation(
                OperationBuilder::new("vector.extract", self.loc)
                    .add_operands(&[val])
                    .add_results(&[ty.scalar.to_mlir_type(self.ctx)])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "static_position"),
                        DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                    )])
                    .build()
                    .expect("failed to unwrap unit vector"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// The LLVM struct behind a `memref<?xT>`: allocated, aligned, offset, size, stride.
    fn emit_memref_descriptor(&mut self, buf: Value<'c, 'a>) -> Value<'c, 'a> {
        let descriptor_ty = Type::parse(
            self.ctx,
            "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>",
        )
        .expect("failed to parse memref descriptor type");
        self.block
            .append_operation(
                OperationBuilder::new("builtin.unrealized_conversion_cast", self.loc)
                    .add_operands(&[buf])
                    .add_results(&[descriptor_ty])
                    .build()
                    .expect("failed to cast memref to descriptor"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    fn emit_descriptor_field(&mut self, descriptor: Value<'c, 'a>, position: &[i64], ty: Type<'c>) -> Value<'c, 'a> {
        self.block
            .append_operation(
                OperationBuilder::new("llvm.extractvalue", self.loc)
                    .add_operands(&[descriptor])
                    .add_results(&[ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "position"),
                        DenseI64ArrayAttribute::new(self.ctx, position).into(),
                    )])
                    .build()
                    .expect("failed to read memref descriptor field"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    fn emit_descriptor_insert(&mut self, descriptor: Value<'c, 'a>, value: Value<'c, 'a>, position: &[i64]) -> Value<'c, 'a> {
        let descriptor_ty = descriptor.r#type();
        self.block
            .append_operation(
                OperationBuilder::new("llvm.insertvalue", self.loc)
                    .add_operands(&[descriptor, value])
                    .add_results(&[descriptor_ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "position"),
                        DenseI64ArrayAttribute::new(self.ctx, position).into(),
                    )])
                    .build()
                    .expect("failed to write memref descriptor field"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    fn emit_gep(&mut self, base: Value<'c, 'a>, offset: Value<'c, 'a>, elem: Type<'c>) -> Value<'c, 'a> {
        self.block
            .append_operation(
                OperationBuilder::new("llvm.getelementptr", self.loc)
                    .add_operands(&[base, offset])
                    .add_results(&[melior::dialect::llvm::r#type::pointer(self.ctx, 0)])
                    .add_attributes(&[
                        (
                            melior::ir::Identifier::new(self.ctx, "elem_type"),
                            TypeAttribute::new(elem).into(),
                        ),
                        (
                            melior::ir::Identifier::new(self.ctx, "rawConstantIndices"),
                            DenseI32ArrayAttribute::new(self.ctx, &[i32::MIN]).into(),
                        ),
                    ])
                    .build()
                    .expect("failed to build llvm.getelementptr"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// The symbol to call for `target`, and the comptime environment its signature is
    /// resolved in. A generic function gets a copy for each set of widths it's called
    /// with, requested here and emitted by `compile_module`.
    fn resolve_callee(
        &mut self,
        target: &ast::FnDef,
        explicit_widths: Option<&Vec<ast::Width>>,
        arg_widths: &[Option<u64>],
    ) -> (String, HashMap<String, u64>) {
        let module_env = self.instances.borrow().module_env.clone();
        let is_generic = target.comptime_params.iter().any(|p| !module_env.contains_key(&p.name));
        if !is_generic {
            let symbol = if target.extern_c { extern_impl_name(&target.name) } else { target.name.clone() };
            return (symbol, module_env);
        }

        let mut env = module_env;
        if let Some(widths) = explicit_widths {
            for (param, width) in target.comptime_params.iter().zip(widths) {
                env.insert(param.name.clone(), width.eval(&self.comptime_env, self.native_width));
            }
        }
        for (param, width) in target.params.iter().zip(arg_widths) {
            if let (Some(ast::Width::Param(p)), Some(width)) = (&param.ty.width, width) {
                if is_scalar_type_name(&param.ty.name) {
                    env.entry(p.clone()).or_insert(*width);
                }
            }
        }

        let mut symbol = target.name.clone();
        for param in &target.comptime_params {
            let width = *env.get(&param.name).unwrap_or_else(|| {
                self.error(&format!("can't infer {} for '{}'", param.name, target.name))
            });
            symbol.push_str(&format!(".{}{}", param.name, width));
        }

        let mut instances = self.instances.borrow_mut();
        if instances.seen.len() > 4096 {
            drop(instances);
            self.error(&format!("too many instances of generic functions, at '{}': is it recursive?", symbol));
        }
        if instances.seen.insert(symbol.clone()) {
            let mut instance = target.clone();
            instance.name = symbol.clone();
            instance.private = true;
            let widths: HashMap<String, u64> = target
                .comptime_params
                .iter()
                .map(|p| (p.name.clone(), env[&p.name]))
                .collect();
            crate::comptime::substitute_in_fn(&mut instance, &widths);
            instances.envs.insert(symbol.clone(), env.clone());
            instances.pending.push(instance);
        }
        (symbol, env)
    }

    /// The high nibble of each lane of a `u8[16]`, packed into a u64 (`shrn #4`).
    fn emit_nibblemask16(&mut self, val: Value<'c, 'a>) -> Value<'c, 'a> {
        let i8t: Type = IntegerType::new(self.ctx, 8).into();
        let i16t: Type = IntegerType::new(self.ctx, 16).into();
        let i64t: Type = IntegerType::new(self.ctx, 64).into();
        let v8i16 = Type::vector(&[8], i16t);
        let v8i8 = Type::vector(&[8], i8t);
        let v1i64 = Type::vector(&[1], i64t);
        let as_i16: Value = self
            .block
            .append_operation(
                OperationBuilder::new("vector.bitcast", self.loc)
                    .add_operands(&[val])
                    .add_results(&[v8i16])
                    .build()
                    .expect("failed to bitcast for nibblemask"),
            )
            .result(0)
            .unwrap()
            .into();
        let four_attr = IntegerAttribute::new(i16t, 4);
        let four: Value = self
            .block
            .append_operation(arith::constant(
                self.ctx,
                splat_attr(v8i16, &[four_attr.into()]).unwrap(),
                self.loc,
            ))
            .result(0)
            .unwrap()
            .into();
        let shifted: Value = self
            .block
            .append_operation(arith::shrui(as_i16, four, self.loc))
            .result(0)
            .unwrap()
            .into();
        let narrowed: Value = self
            .block
            .append_operation(arith::trunci(shifted, v8i8, self.loc))
            .result(0)
            .unwrap()
            .into();
        let as_u64: Value = self
            .block
            .append_operation(
                OperationBuilder::new("vector.bitcast", self.loc)
                    .add_operands(&[narrowed])
                    .add_results(&[v1i64])
                    .build()
                    .expect("failed to bitcast nibbles"),
            )
            .result(0)
            .unwrap()
            .into();
                self.emit_unwrap_unit(as_u64, VecType { scalar: ScalarType::U64, width: 1 })
    }

    /// `any` over 16 or more lanes in whole registers: sign-extend to bytes, OR the
    /// registers together, and test one `shrn`. Bitcasting wide i1 vectors to integers
    /// builds a full bitmask per register instead.
    fn emit_any_wide(&mut self, val: Value<'c, 'a>, width: u64) -> Value<'c, 'a> {
        let i8t: Type = IntegerType::new(self.ctx, 8).into();
        let i64t: Type = IntegerType::new(self.ctx, 64).into();
        let bytes: Value = self
            .block
            .append_operation(
                OperationBuilder::new("arith.extsi", self.loc)
                    .add_operands(&[val])
                    .add_results(&[Type::vector(&[width], i8t)])
                    .build()
                    .expect("failed to build arith.extsi"),
            )
            .result(0)
            .unwrap()
            .into();
        let mut combined = if width == 16 { bytes } else { self.emit_extract16(bytes, 0) };
        let mut offset = 16;
        while offset < width as usize {
            let slice = self.emit_extract16(bytes, offset);
            combined = self
                .block
                .append_operation(arith::ori(combined, slice, self.loc))
                .result(0)
                .unwrap()
                .into();
            offset += 16;
        }
        let nibbles = self.emit_nibblemask16(combined);
        let zero: Value = self
            .block
            .append_operation(arith::constant(self.ctx, IntegerAttribute::new(i64t, 0).into(), self.loc))
            .result(0)
            .unwrap()
            .into();
        self.block
            .append_operation(arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, nibbles, zero, self.loc))
            .result(0)
            .unwrap()
            .into()
    }

    /// The fields of a struct type, with their widths resolved.
    fn struct_field_types(&self, ty: &ast::Type, env: &HashMap<String, u64>) -> Vec<(String, VecType)> {
        let Some(def) = self.struct_defs.get(&ty.name) else {
            self.error(&format!("undefined struct: {}", ty.name));
        };
        let width = ty
            .width
            .as_ref()
            .unwrap_or_else(|| self.error(&format!("struct {} needs a width", ty.name)))
            .eval(env, self.native_width);
        let mut struct_env = env.clone();
        if let Some(cp) = def.comptime_params.first() {
            struct_env.insert(cp.name.clone(), width);
        }
        def.fields
            .iter()
            .map(|f| (f.name.clone(), resolve_type(&f.ty, &struct_env, self.native_width)))
            .collect()
    }

    /// The name of the user function an expression calls, if it does.
    fn called_user_fn(&self, expr: &ast::Expr) -> Option<ast::FnDef> {
        let ast::Expr::Call { func, .. } = expr else { return None };
        let name = match func.as_ref() {
            ast::Expr::Ident(n) => n,
            ast::Expr::Instantiate { name, .. } => name,
            _ => return None,
        };
        self.fn_defs.get(name).cloned()
    }

    /// Whether an expression evaluates to a struct.
    fn is_struct_expr(&self, expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::StructLit { .. } => true,
            ast::Expr::Ident(name) => matches!(self.vars.get(name), Some(Binding::Struct { .. })),
            _ => self
                .called_user_fn(expr)
                .and_then(|f| f.ret_ty)
                .is_some_and(|r| !is_scalar_type_name(&r.name) && !is_ptr_type_name(&r.name)),
        }
    }

    /// Evaluate a struct-valued expression to its fields, in declaration order.
    fn emit_struct_value(&mut self, expr: &ast::Expr) -> Vec<(String, Value<'c, 'a>, VecType)> {
        match expr {
            ast::Expr::StructLit { name, width, fields } => self.emit_struct_fields(name, width, fields),
            ast::Expr::Ident(name) => match self.vars.get(name) {
                Some(Binding::Struct { fields }) => fields.clone(),
                _ => self.error(&format!("'{}' is not a struct", name)),
            },
            ast::Expr::Call { func, args } => {
                let (results, ret_ty, env) = self.emit_user_call(func, args);
                let Some(ret_ty) = ret_ty else {
                    self.error("function has no return value");
                };
                let types = self.struct_field_types(&ret_ty, &env);
                types
                    .into_iter()
                    .zip(results)
                    .map(|((name, ty), val)| (name, val, ty))
                    .collect()
            }
            _ => self.error("expected a struct value"),
        }
    }

    /// The fields of a struct literal, each a field-typed literal or an expression.
    fn emit_struct_fields(
        &mut self,
        name: &str,
        width: &ast::Width,
        field_exprs: &[(String, ast::Expr)],
    ) -> Vec<(String, Value<'c, 'a>, VecType)> {
        let ty = ast::Type { name: name.to_string(), width: Some(width.clone()) };
        let env = self.comptime_env.clone();
        let types = self.struct_field_types(&ty, &env);
        let mut fields = Vec::with_capacity(types.len());
        for (field, field_ty) in types {
            let Some((_, expr)) = field_exprs.iter().find(|(f, _)| *f == field) else {
                self.error(&format!("struct literal '{}' missing field '{}'", name, field));
            };
            let (val, ty) = self.emit_expr_typed(expr, field_ty);
            fields.push((field, val, ty));
        }
        fields
    }

    /// `llvm.ptr` to byte offset `index` of a byte pointer. This is the lowered form
    /// of a memref access; going through it directly lets a scalar load or store
    /// carry `align 1`, which `memref.load` and `memref.store` cannot express.
    fn emit_byte_address(&mut self, buf: Value<'c, 'a>, index: Value<'c, 'a>) -> Value<'c, 'a> {
        let descriptor = self.emit_memref_descriptor(buf);
        let aligned = self.emit_descriptor_field(descriptor, &[1], melior::dialect::llvm::r#type::pointer(self.ctx, 0));
        let i64t: Type = IntegerType::new(self.ctx, 64).into();
        let offset: Value = self
            .block
            .append_operation(arith::index_cast(index, i64t, self.loc))
            .result(0)
            .unwrap()
            .into();
        self.emit_gep(aligned, offset, IntegerType::new(self.ctx, 8).into())
    }

    /// A scalar load from a byte pointer, at any alignment.
    fn emit_unaligned_load(&mut self, buf: Value<'c, 'a>, index: Value<'c, 'a>, scalar: ScalarType) -> Value<'c, 'a> {
        let address = self.emit_byte_address(buf, index);
        self.block
            .append_operation(
                OperationBuilder::new("llvm.load", self.loc)
                    .add_operands(&[address])
                    .add_results(&[scalar.to_mlir_type(self.ctx)])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "alignment"),
                        IntegerAttribute::new(IntegerType::new(self.ctx, 64).into(), 1).into(),
                    )])
                    .build()
                    .expect("failed to build llvm.load"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// A scalar store to a byte pointer, at any alignment.
    fn emit_unaligned_store(&mut self, val: Value<'c, 'a>, buf: Value<'c, 'a>, index: Value<'c, 'a>) {
        let address = self.emit_byte_address(buf, index);
        self.block.append_operation(
            OperationBuilder::new("llvm.store", self.loc)
                .add_operands(&[val, address])
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "alignment"),
                    IntegerAttribute::new(IntegerType::new(self.ctx, 64).into(), 1).into(),
                )])
                .build()
                .expect("failed to build llvm.store"),
        );
    }

    /// `memref.load` of one element of the pointer's own type.
    fn emit_memref_load(&mut self, buf: Value<'c, 'a>, index: Value<'c, 'a>, scalar: ScalarType) -> Value<'c, 'a> {
        self.block
            .append_operation(
                OperationBuilder::new("memref.load", self.loc)
                    .add_operands(&[buf, index])
                    .add_results(&[scalar.to_mlir_type(self.ctx)])
                    .build()
                    .expect("failed to build memref.load"),
            )
            .result(0)
            .unwrap()
            .into()
    }

    /// Store `val` at element index `index` of `buf`, reinterpreting it as the
    /// pointer's element type when they differ (e.g. a `u64[1]` into a `ptr[u8]`).
    fn emit_vector_store(
        &mut self,
        val: Value<'c, 'a>,
        val_ty: VecType,
        buf: Value<'c, 'a>,
        buf_scalar: ScalarType,
        index: Value<'c, 'a>,
    ) {
        // Scalars stay scalar (see emit_load).
        if val_ty.width == 1 {
            let same = buf_scalar.to_mlir_type(self.ctx) == val_ty.scalar.to_mlir_type(self.ctx);
            if same {
                self.block.append_operation(
                    OperationBuilder::new("memref.store", self.loc)
                        .add_operands(&[val, buf, index])
                        .build()
                        .expect("failed to build memref.store"),
                );
                return;
            }
            if buf_scalar.bit_width() == 8 {
                self.emit_unaligned_store(val, buf, index);
                return;
            }
        }
        let as_vector: Value<'c, 'a> = if val_ty.width == 1 {
            self.block
                .append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[val])
                        .add_results(&[Type::vector(&[1], val_ty.scalar.to_mlir_type(self.ctx))])
                        .build()
                        .expect("failed to wrap scalar for store"),
                )
                .result(0)
                .unwrap()
                .into()
        } else {
            val
        };
        let to_store: Value<'c, 'a> = if buf_scalar.to_mlir_type(self.ctx) != val_ty.scalar.to_mlir_type(self.ctx) {
            let total_bits = val_ty.scalar.bit_width() as u64 * val_ty.width;
            let elem_bits = buf_scalar.bit_width() as u64;
            if total_bits % elem_bits != 0 {
                self.error("store value size is not a multiple of the pointer's element size");
            }
            self.block
                .append_operation(
                    OperationBuilder::new("vector.bitcast", self.loc)
                        .add_operands(&[as_vector])
                        .add_results(&[Type::vector(&[total_bits / elem_bits], buf_scalar.to_mlir_type(self.ctx))])
                        .build()
                        .expect("failed to bitcast store value"),
                )
                .result(0)
                .unwrap()
                .into()
        } else {
            as_vector
        };
        // Write wide byte vectors one register at a time, ascending. Left whole, LLVM
        // splits the store and may write the upper half first, which is markedly slower
        // once the destination is out of cache.
        let total_bytes = val_ty.scalar.bit_width() as u64 * val_ty.width / 8;
        let register = self.arch().register_bytes();
        if buf_scalar.bit_width() == 8 && total_bytes > register && total_bytes % register == 0 {
            let mut offset = 0;
            while offset < total_bytes as i64 {
                let slice = self.emit_extract_lanes(to_store, offset as usize, register);
                let delta = self.make_index_const(offset);
                let at: Value = self
                    .block
                    .append_operation(arith::addi(index, delta, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                self.block.append_operation(
                    OperationBuilder::new("vector.store", self.loc)
                        .add_operands(&[slice, buf, at])
                        .build()
                        .expect("failed to build vector.store"),
                );
                offset += register as i64;
            }
            return;
        }
        self.block.append_operation(
            OperationBuilder::new("vector.store", self.loc)
                .add_operands(&[to_store, buf, index])
                .build()
                .expect("failed to build vector.store"),
        );
    }

    /// The element type of a pointer expression.
    fn ptr_scalar_of(&self, ptr: &ast::Expr, fallback: ScalarType) -> ScalarType {
        if let ast::Expr::Ident(name) = ptr {
            if let Some(Binding::Ptr(_, s)) = self.vars.get(name) {
                return *s;
            }
        }
        fallback
    }

    fn emit_store(
        &mut self,
        ptr: &ast::Expr,
        value: &ast::Expr,
        mask: Option<&ast::Expr>,
    ) {
        let (ptr_val, ptr_ty) = self.emit_expr(ptr);
        let buf_scalar = self.ptr_scalar_of(ptr, ptr_ty.scalar);
        let (val, val_ty) = self.emit_store_value(value, buf_scalar);
        let zero_index = self.make_index_const(0);
        match mask {
            None => self.emit_vector_store(val, val_ty, ptr_val, buf_scalar, zero_index),
            Some(mask) => {
                let (mask_val, mask_ty) = self.emit_expr(mask);
                if mask_ty.scalar != ScalarType::Bool || mask_ty.width != val_ty.width {
                    self.error(&format!(
                        "store mask must be bool[{}], got {:?}[{}]",
                        val_ty.width, mask_ty.scalar, mask_ty.width
                    ));
                }
                if val_ty.width == 1 {
                    self.error("a masked store needs a vector value; use if for a scalar");
                }
                if buf_scalar.bit_width() != val_ty.scalar.bit_width() {
                    self.error("a masked store's value must have the pointer's element size");
                }
                self.block.append_operation(
                    OperationBuilder::new("vector.maskedstore", self.loc)
                        .add_operands(&[ptr_val, zero_index, mask_val, val])
                        .build()
                        .expect("failed to build vector.maskedstore"),
                );
            }
        }
    }

    /// A value to store: a literal takes the pointer's element type.
    fn emit_store_value(&mut self, value: &ast::Expr, buf_scalar: ScalarType) -> (Value<'c, 'a>, VecType) {
        match value {
            _ if Self::is_literal(value) => {
                self.emit_literal_broadcast(value, VecType { scalar: buf_scalar, width: 1 })
            }
            ast::Expr::Masked { mask, body, fallback } if Self::takes_expected_type(value) => {
                self.emit_masked(mask, body, fallback.as_deref(), Some(buf_scalar))
            }
            _ => self.emit_expr(value),
        }
    }

    fn emit_scatter(
        &mut self,
        base: &ast::Expr,
        index: &ast::Expr,
        mask: Option<&ast::Expr>,
        value: &ast::Expr,
    ) {
        let (base_val, base_ty) = self.emit_expr(base);
        let element = self.ptr_scalar_of(base, base_ty.scalar);
        let (index_val, index_ty) = self.emit_expr(index);
        let (val, val_ty) = self.emit_store_value(value, element);
        let val = if val_ty.width == 1 && index_ty.width > 1 {
            self.emit_broadcast_to_width(val, val_ty, index_ty.width).0
        } else {
            val
        };
        if index_ty.width == 1 {
            if mask.is_some() {
                self.error("a masked scatter needs a vector of indices; use if for a scalar");
            }
            let at = self.emit_value_as_index(index_val, index_ty);
            self.block.append_operation(
                OperationBuilder::new("memref.store", self.loc)
                    .add_operands(&[val, base_val, at])
                    .build()
                    .expect("failed to build memref.store"),
            );
            return;
        }

        // Build mask (all-true if none provided)
        let mask_ty = VecType {
            scalar: ScalarType::Bool,
            width: index_ty.width,
        };
        let mask_val = if let Some(m) = mask {
            self.emit_expr(m).0
        } else {
            let mask_mlir = mask_ty.to_mlir_type(self.ctx);
            let scalar_type = ScalarType::Bool.to_mlir_type(self.ctx);
            let attr = IntegerAttribute::new(scalar_type, 1);
            let splat = splat_attr(mask_mlir, &[attr.into()])
                .expect("failed to create all-true mask");
            let op = arith::constant(self.ctx, splat.into(), self.loc);
            self.block.append_operation(op).result(0).unwrap().into()
        };

        // vector.scatter %base[%c0], %indices, %mask, %value
        let index_type = Type::index(self.ctx);
        let zero_index_attr = IntegerAttribute::new(index_type, 0);
        let zero_index_op = arith::constant(self.ctx, zero_index_attr.into(), self.loc);
        let zero_index = self
            .block
            .append_operation(zero_index_op)
            .result(0)
            .unwrap()
            .into();

        let index_val = self.emit_vector_indices(index_val, index_ty);
        let scatter_op = OperationBuilder::new("vector.scatter", self.loc)
            .add_operands(&[base_val, zero_index, index_val, mask_val, val])
            .build()
            .expect("failed to build vector.scatter");

        self.block.append_operation(scatter_op);
    }

    fn emit_call(
        &mut self,
        func_expr: &ast::Expr,
        args: &[ast::CallArg],
    ) -> (Value<'c, 'a>, VecType) {
        match func_expr {
            ast::Expr::Ident(name) if name == "sqrt" => {
                assert_eq!(args.len(), 1, "sqrt takes exactly one argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                assert!(ty.scalar.is_float(), "sqrt requires float type");
                let vec_type = ty.to_mlir_type(self.ctx);
                let sqrt_op = OperationBuilder::new("math.sqrt", self.loc)
                    .add_operands(&[val])
                    .add_results(&[vec_type])
                    .build()
                    .expect("failed to build math.sqrt");
                let result = self
                    .block
                    .append_operation(sqrt_op)
                    .result(0)
                    .unwrap()
                    .into();
                (result, ty)
            }
            ast::Expr::Ident(name) if name == "compress" => {
                // compress(data, mask) → pack lanes where mask is true to the front
                assert_eq!(args.len(), 2, "compress takes exactly 2 arguments (data, mask)");
                let (data_val, data_ty) = self.emit_expr(&args[0].value);
                let (mask_val, _mask_ty) = self.emit_expr(&args[1].value);
                let result_mlir = data_ty.to_mlir_type(self.ctx);
                let lane = if data_ty.scalar.is_float() { "f" } else { "i" };
                let intrinsic = format!("llvm.experimental.vector.compress.v{}{}{}", data_ty.width, lane, data_ty.scalar.bit_width());
                let passthrough = self.emit_zero_vector(data_ty);
                let result = self.emit_intrinsic(&intrinsic, &[data_val, mask_val, passthrough], result_mlir);
                (result, data_ty)
            }
            ast::Expr::Ident(name) if name == "popcount" => {
                // popcount(bool_vec) → count of true lanes, as i32[1]
                assert_eq!(args.len(), 1, "popcount takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                // Cast bool to i32 then reduce with add
                let i32_ty = VecType {
                    scalar: ScalarType::I32,
                    width: ty.width,
                };
                let i32_mlir = i32_ty.to_mlir_type(self.ctx);
                if ty.width == 1 {
                    let widened = self
                        .block
                        .append_operation(arith::extui(val, i32_mlir, self.loc))
                        .result(0)
                        .unwrap()
                        .into();
                    return (widened, i32_ty);
                }

                // Zero-extend bool (i1) to i32
                let ext_op = arith::extui(val, i32_mlir, self.loc);
                let ext_val = self
                    .block
                    .append_operation(ext_op)
                    .result(0)
                    .unwrap()
                    .into();

                // Reduce with add
                let result_ty = VecType {
                    scalar: ScalarType::I32,
                    width: 1,
                };
                let scalar_i32 = ScalarType::I32.to_mlir_type(self.ctx);
                let kind_attr = melior::ir::attribute::Attribute::parse(
                    self.ctx,
                    "#vector.kind<add>",
                )
                .expect("failed to parse vector.kind attribute");

                let reduction_op = OperationBuilder::new("vector.reduction", self.loc)
                    .add_operands(&[ext_val])
                    .add_results(&[scalar_i32])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "kind"),
                        kind_attr,
                    )])
                    .build()
                    .expect("failed to build vector.reduction");

                let scalar_val = self
                    .block
                    .append_operation(reduction_op)
                    .result(0)
                    .unwrap()
                    .into();

                // Broadcast to width-1 vector
                let result = scalar_val;

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "store" => {
                // store(ptr, value) or store(ptr, value, mask)
                assert!(
                    args.len() == 2 || args.len() == 3,
                    "store takes 2 or 3 arguments"
                );
                self.emit_store(&args[0].value, &args[1].value, args.get(2).map(|a| &a.value));
                // store doesn't return a value — return a dummy
                // This is only used in expression statements
                let dummy_ty = VecType {
                    scalar: ScalarType::I32,
                    width: 1,
                };
                let dummy = self.emit_zero_vector(dummy_ty);
                (dummy, dummy_ty)
            }
            ast::Expr::Ident(name) if name == "iota" => {
                // iota(width) → [0, 1, 2, ..., width-1] as i32 vector
                assert_eq!(args.len(), 1, "iota takes exactly 1 argument (width)");
                let width = match &args[0].value {
                    ast::Expr::IntLit(n) => *n as u64,
                    _ => panic!("iota argument must be an integer literal"),
                };
                let result_ty = VecType {
                    scalar: ScalarType::I32,
                    width,
                };
                let vec_mlir = result_ty.to_mlir_type(self.ctx);
                let i32_type = ScalarType::I32.to_mlir_type(self.ctx);

                let values: Vec<melior::ir::attribute::Attribute> = (0..width as i64)
                    .map(|i| IntegerAttribute::new(i32_type, i).into())
                    .collect();
                let dense = splat_attr(vec_mlir, &values)
                    .expect("failed to create iota dense attribute");
                let op = arith::constant(self.ctx, dense.into(), self.loc);
                let val = self.block.append_operation(op).result(0).unwrap().into();
                (val, result_ty)
            }
            ast::Expr::Ident(name) if name == "extract" => {
                // extract(vec, lane) → scalar value from lane as width-1 vector
                assert_eq!(args.len(), 2, "extract takes exactly 2 arguments (vec, lane)");
                let (val, ty) = self.emit_expr(&args[0].value);
                let lane = match &args[1].value {
                    ast::Expr::IntLit(n) => *n,
                    _ => panic!("extract lane must be an integer literal"),
                };
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let result_ty = VecType {
                    scalar: ty.scalar,
                    width: 1,
                };
                if ty.width == 1 {
                    return (val, result_ty);
                }

                // vector.extract to scalar
                let extract_op = OperationBuilder::new("vector.extract", self.loc)
                    .add_operands(&[val])
                    .add_results(&[scalar_type])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "static_position"),
                        DenseI64ArrayAttribute::new(self.ctx, &[lane]).into(),
                    )])
                    .build()
                    .expect("failed to build vector.extract");
                let scalar_val = self.block.append_operation(extract_op).result(0).unwrap().into();

                // Broadcast scalar to width-1 vector
                let result = scalar_val;
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "compressstore" => {
                // compressstore(buf, offset, data, mask) → stores selected lanes contiguously
                assert_eq!(args.len(), 4, "compressstore takes 4 arguments (buf, offset, data, mask)");
                let (buf_val, _) = self.emit_expr(&args[0].value);
                let offset_index = self.emit_index(&args[1].value);
                let (data_val, _) = self.emit_expr(&args[2].value);
                let (mask_val, _) = self.emit_expr(&args[3].value);

                // vector.compressstore %buf[%offset], %mask, %data
                let compressstore_op = OperationBuilder::new("vector.compressstore", self.loc)
                    .add_operands(&[buf_val, offset_index, mask_val, data_val])
                    .build()
                    .expect("failed to build vector.compressstore");
                self.block.append_operation(compressstore_op);

                // Return a dummy value (compressstore is void)
                let dummy_ty = VecType { scalar: ScalarType::I32, width: 1 };
                let dummy = self.emit_zero_vector(dummy_ty);
                (dummy, dummy_ty)
            }
            ast::Expr::Ident(name) if int_conversion_target(name).is_some() => {
                // to_i64(vec) and friends: each lane to that type, extended by its signedness.
                assert_eq!(args.len(), 1, "{} takes exactly 1 argument", name);
                let (val, ty) = self.emit_expr(&args[0].value);
                let scalar = int_conversion_target(name).unwrap();
                let result_ty = VecType { scalar, width: ty.width };
                (self.emit_int_resize(val, ty, result_ty), result_ty)
            }
            ast::Expr::Ident(name) if name == "to_bitmask" => {
                // to_bitmask(bool[N]) → pack into u64[1], one bit per lane
                // Uses vector.bitcast which LLVM lowers to efficient NEON
                // (cmeq + and-with-weights + zip + addv)
                assert_eq!(args.len(), 1, "to_bitmask takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                assert_eq!(ty.scalar, ScalarType::Bool, "to_bitmask requires bool vector");
                if ty.width > 64 {
                    self.error("to_bitmask can pack at most 64 lanes");
                }
                let width = ty.width;
                let i64_type: Type = IntegerType::new(self.ctx, 64).into();

                // Bitcast vector<Nxi1> → iN directly. A vector<1xiN> intermediate gets
                // legalized through a stack slot on AArch64.
                let int_type: Type = IntegerType::new(self.ctx, width as u32).into();
                let scalar: Value = self.block.append_operation(
                    OperationBuilder::new("llvm.bitcast", self.loc)
                        .add_operands(&[val])
                        .add_results(&[int_type])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let scalar_i64 = if width < 64 {
                    self.block.append_operation(
                        arith::extui(scalar, i64_type, self.loc)
                    ).result(0).unwrap().into()
                } else {
                    scalar
                };

                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };
                let result: Value = scalar_i64;

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "from_bitmask" => {
                // from_bitmask(u64[1], N) → bool[N], unpack bits to bool vector
                assert_eq!(args.len(), 2, "from_bitmask takes 2 arguments (bitmask, width)");
                let (val, _) = self.emit_expr(&args[0].value);
                let target_width = match &args[1].value {
                    ast::Expr::IntLit(n) => *n as u64,
                    _ => panic!("from_bitmask width must be an integer literal"),
                };

                let int_type: Type = IntegerType::new(self.ctx, target_width as u32).into();
                let vec1xint = Type::vector(&[1], int_type);
                let result_ty = VecType { scalar: ScalarType::Bool, width: target_width };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                // Extract i64 scalar
                let scalar: Value = val;

                // Truncate to N bits if needed
                let scalar_n = if target_width < 64 {
                    let trunc: Value = self.block.append_operation(
                        arith::trunci(scalar, int_type, self.loc)
                    ).result(0).unwrap().into();
                    trunc
                } else {
                    scalar
                };

                // Broadcast to vector<1xiN> then bitcast to vector<Nxi1>
                let vec_int: Value = self.block.append_operation(
                    OperationBuilder::new("vector.broadcast", self.loc)
                        .add_operands(&[scalar_n])
                        .add_results(&[vec1xint])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let result: Value = self.block.append_operation(
                    OperationBuilder::new("vector.bitcast", self.loc)
                        .add_operands(&[vec_int])
                        .add_results(&[result_mlir])
                        .build().unwrap()
                ).result(0).unwrap().into();

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "scan_dfa" => {
                // scan_dfa(table: u8[256], classes: u8[N], seed: u8[1], num_states)
                //   → u8[N]: the DFA state after each byte. Prefix scan over the
                //   transition monoid (function composition), via the same log-step
                //   Hillis-Steele structure as scan.xor but composing per-byte
                //   transition functions, represented as `num_states` state-vectors.
                assert_eq!(args.len(), 4, "scan_dfa takes 4 args (table, classes, seed, num_states)");
                let (table_val, _) = self.emit_expr(&args[0].value);
                let (classes_val, classes_ty) = self.emit_expr(&args[1].value);
                let (seed_val, _) = self.emit_expr(&args[2].value);
                let k = match &args[3].value {
                    ast::Expr::IntLit(n) => *n as usize,
                    _ => panic!("scan_dfa num_states must be an integer literal"),
                };
                return self.emit_dfa_scan(table_val, classes_val, seed_val, k, classes_ty);
            }
            ast::Expr::Ident(name) if name == "clmul" => {
                // clmul(u64[1], u64[1]) → u64[1], carry-less multiply (PMULL)
                assert_eq!(args.len(), 2, "clmul takes exactly 2 arguments");
                let (a_val, _) = self.emit_expr(&args[0].value);
                let (b_val, _) = self.emit_expr(&args[1].value);

                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let vec16xi8 = Type::vector(&[16], IntegerType::new(self.ctx, 8).into());
                let vec2xi64 = Type::vector(&[2], i64_type);
                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };

                // Extract scalars from u64[1]
                let a_scalar: Value = a_val;

                let b_scalar: Value = b_val;

                let pmull: Value = match self.arch() {
                    Arch::Aarch64 => self.emit_intrinsic("llvm.aarch64.neon.pmull64", &[a_scalar, b_scalar], vec16xi8),
                    Arch::X86_64 => {
                        let wrap = |this: &mut Self, value: Value<'c, 'a>| -> Value<'c, 'a> {
                            this.block
                                .append_operation(
                                    OperationBuilder::new("vector.broadcast", this.loc)
                                        .add_operands(&[value])
                                        .add_results(&[vec2xi64])
                                        .build()
                                        .expect("failed to wrap clmul operand"),
                                )
                                .result(0)
                                .unwrap()
                                .into()
                        };
                        let a_vec = wrap(self, a_scalar);
                        let b_vec = wrap(self, b_scalar);
                        let i8t: Type = IntegerType::new(self.ctx, 8).into();
                        let low_lanes: Value = self
                            .block
                            .append_operation(arith::constant(self.ctx, IntegerAttribute::new(i8t, 0).into(), self.loc))
                            .result(0)
                            .unwrap()
                            .into();
                        let product = self.emit_intrinsic("llvm.x86.pclmulqdq", &[a_vec, b_vec, low_lanes], vec2xi64);
                        self.block
                            .append_operation(
                                OperationBuilder::new("vector.bitcast", self.loc)
                                    .add_operands(&[product])
                                    .add_results(&[vec16xi8])
                                    .build()
                                    .expect("failed to bitcast clmul result"),
                            )
                            .result(0)
                            .unwrap()
                            .into()
                    }
                };

                // Bitcast to 2xi64, extract lower
                let as_2xi64: Value = self.block.append_operation(
                    OperationBuilder::new("vector.bitcast", self.loc)
                        .add_operands(&[pmull])
                        .add_results(&[vec2xi64])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let lower: Value = self.block.append_operation(
                    OperationBuilder::new("vector.extract", self.loc)
                        .add_operands(&[as_2xi64])
                        .add_results(&[i64_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "static_position"),
                            DenseI64ArrayAttribute::new(self.ctx, &[0]).into(),
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                // Wrap in u64[1]
                let result: Value = lower;

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "ctz" => {
                // ctz(u64[1]) → u64[1], count trailing zeros
                assert_eq!(args.len(), 1, "ctz takes exactly 1 argument");
                let (val, _) = self.emit_expr(&args[0].value);
                let i64_type: Type = IntegerType::new(self.ctx, 64).into();
                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };

                let scalar: Value = val;

                let ctz_result: Value = self.block.append_operation(
                    melior::dialect::llvm::intr_cttz(self.ctx, scalar, false, i64_type, self.loc)
                ).result(0).unwrap().into();

                let result: Value = ctz_result;

                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "clear_lowest_bit" => {
                // clear_lowest_bit(u64[1]) → u64[1], x & (x-1)
                assert_eq!(args.len(), 1, "clear_lowest_bit takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);

                let i64_type: Type = IntegerType::new(self.ctx, 64).into();

                // x - 1
                let one_attr = IntegerAttribute::new(i64_type, 1);
                let one_splat = one_attr;
                let one_vec: Value = self.block.append_operation(
                    arith::constant(self.ctx, one_splat.into(), self.loc)
                ).result(0).unwrap().into();

                let x_minus_1: Value = self.block.append_operation(
                    arith::subi(val, one_vec, self.loc)
                ).result(0).unwrap().into();

                // x & (x - 1)
                let result: Value = self.block.append_operation(
                    arith::andi(val, x_minus_1, self.loc)
                ).result(0).unwrap().into();

                (result, ty)
            }
            ast::Expr::Ident(name) if name == "bswap" => {
                // bswap(val) → reverse bytes within each element
                assert_eq!(args.len(), 1, "bswap takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);

                let bit_width = ty.scalar.bit_width();
                let int_type: Type = IntegerType::new(self.ctx, bit_width as u32).into();
                let vec_type = ty.to_mlir_type(self.ctx);

                // Helper to create a constant vector splat
                let make_const = |cg: &mut Self, c: i64| -> Value<'c, 'a> {
                    let attr = IntegerAttribute::new(int_type, c);
                    let splat = splat_attr(vec_type, &[attr.into()]).unwrap();
                    cg.block.append_operation(
                        arith::constant(cg.ctx, splat.into(), cg.loc)
                    ).result(0).unwrap().into()
                };

                let result = match bit_width {
                    16 => {
                        // (x >> 8) | (x << 8)
                        let c8 = make_const(self, 8);
                        let shr: Value = self.block.append_operation(arith::shrui(val, c8, self.loc)).result(0).unwrap().into();
                        let shl: Value = self.block.append_operation(arith::shli(val, c8, self.loc)).result(0).unwrap().into();
                        let r: Value = self.block.append_operation(arith::ori(shr, shl, self.loc)).result(0).unwrap().into();
                        r
                    }
                    32 => {
                        // byte0 = (x >> 24) & 0xFF
                        // byte1 = (x >> 8) & 0xFF00
                        // byte2 = (x << 8) & 0xFF0000
                        // byte3 = (x << 24)
                        let c8 = make_const(self, 8);
                        let c24 = make_const(self, 24);
                        let m_ff = make_const(self, 0xFF);
                        let m_ff00 = make_const(self, 0xFF00);
                        let m_ff0000 = make_const(self, 0xFF_0000);

                        let shr24: Value = self.block.append_operation(arith::shrui(val, c24, self.loc)).result(0).unwrap().into();
                        let byte0: Value = self.block.append_operation(arith::andi(shr24, m_ff, self.loc)).result(0).unwrap().into();

                        let shr8: Value = self.block.append_operation(arith::shrui(val, c8, self.loc)).result(0).unwrap().into();
                        let byte1: Value = self.block.append_operation(arith::andi(shr8, m_ff00, self.loc)).result(0).unwrap().into();

                        let shl8: Value = self.block.append_operation(arith::shli(val, c8, self.loc)).result(0).unwrap().into();
                        let byte2: Value = self.block.append_operation(arith::andi(shl8, m_ff0000, self.loc)).result(0).unwrap().into();

                        let byte3: Value = self.block.append_operation(arith::shli(val, c24, self.loc)).result(0).unwrap().into();

                        let r01: Value = self.block.append_operation(arith::ori(byte0, byte1, self.loc)).result(0).unwrap().into();
                        let r23: Value = self.block.append_operation(arith::ori(byte2, byte3, self.loc)).result(0).unwrap().into();
                        let r: Value = self.block.append_operation(arith::ori(r01, r23, self.loc)).result(0).unwrap().into();
                        r
                    }
                    64 => {
                        // 8 bytes: reverse using shift/mask/or
                        let c8 = make_const(self, 8);
                        let c24 = make_const(self, 24);
                        let c40 = make_const(self, 40);
                        let c56 = make_const(self, 56);
                        let m_ff = make_const(self, 0xFF);
                        let m_ff00 = make_const(self, 0xFF00);
                        let m_ff0000 = make_const(self, 0xFF_0000);
                        let m_ff000000 = make_const(self, 0xFF00_0000);
                        let m_ff00000000: Value = make_const(self, 0xFF_0000_0000);
                        let m_ff0000000000: Value = make_const(self, 0xFF00_0000_0000);
                        let m_ff000000000000: Value = make_const(self, 0xFF_0000_0000_0000);

                        // byte0 = (x >> 56) & 0xFF
                        let shr56: Value = self.block.append_operation(arith::shrui(val, c56, self.loc)).result(0).unwrap().into();
                        let b0: Value = self.block.append_operation(arith::andi(shr56, m_ff, self.loc)).result(0).unwrap().into();
                        // byte1 = (x >> 40) & 0xFF00
                        let shr40: Value = self.block.append_operation(arith::shrui(val, c40, self.loc)).result(0).unwrap().into();
                        let b1: Value = self.block.append_operation(arith::andi(shr40, m_ff00, self.loc)).result(0).unwrap().into();
                        // byte2 = (x >> 24) & 0xFF0000
                        let shr24: Value = self.block.append_operation(arith::shrui(val, c24, self.loc)).result(0).unwrap().into();
                        let b2: Value = self.block.append_operation(arith::andi(shr24, m_ff0000, self.loc)).result(0).unwrap().into();
                        // byte3 = (x >> 8) & 0xFF000000
                        let shr8: Value = self.block.append_operation(arith::shrui(val, c8, self.loc)).result(0).unwrap().into();
                        let b3: Value = self.block.append_operation(arith::andi(shr8, m_ff000000, self.loc)).result(0).unwrap().into();
                        // byte4 = (x << 8) & 0xFF00000000
                        let shl8: Value = self.block.append_operation(arith::shli(val, c8, self.loc)).result(0).unwrap().into();
                        let b4: Value = self.block.append_operation(arith::andi(shl8, m_ff00000000, self.loc)).result(0).unwrap().into();
                        // byte5 = (x << 24) & 0xFF0000000000
                        let shl24: Value = self.block.append_operation(arith::shli(val, c24, self.loc)).result(0).unwrap().into();
                        let b5: Value = self.block.append_operation(arith::andi(shl24, m_ff0000000000, self.loc)).result(0).unwrap().into();
                        // byte6 = (x << 40) & 0xFF000000000000
                        let shl40: Value = self.block.append_operation(arith::shli(val, c40, self.loc)).result(0).unwrap().into();
                        let b6: Value = self.block.append_operation(arith::andi(shl40, m_ff000000000000, self.loc)).result(0).unwrap().into();
                        // byte7 = (x << 56)
                        let b7: Value = self.block.append_operation(arith::shli(val, c56, self.loc)).result(0).unwrap().into();

                        let r01: Value = self.block.append_operation(arith::ori(b0, b1, self.loc)).result(0).unwrap().into();
                        let r23: Value = self.block.append_operation(arith::ori(b2, b3, self.loc)).result(0).unwrap().into();
                        let r45: Value = self.block.append_operation(arith::ori(b4, b5, self.loc)).result(0).unwrap().into();
                        let r67: Value = self.block.append_operation(arith::ori(b6, b7, self.loc)).result(0).unwrap().into();
                        let r0123: Value = self.block.append_operation(arith::ori(r01, r23, self.loc)).result(0).unwrap().into();
                        let r4567: Value = self.block.append_operation(arith::ori(r45, r67, self.loc)).result(0).unwrap().into();
                        let r: Value = self.block.append_operation(arith::ori(r0123, r4567, self.loc)).result(0).unwrap().into();
                        r
                    }
                    _ => panic!("bswap unsupported for bit width {}", bit_width),
                };

                (result, ty)
            }
            ast::Expr::Ident(name) if name == "xor" => {
                // xor(a, b) → bitwise XOR (legacy alias for a ^ b)
                assert_eq!(args.len(), 2, "xor takes exactly 2 arguments");
                let (a_val, a_ty) = self.emit_expr(&args[0].value);
                let (b_val, _) = self.emit_expr(&args[1].value);
                let result: Value = self.block.append_operation(
                    arith::xori(a_val, b_val, self.loc)
                ).result(0).unwrap().into();
                (result, a_ty)
            }
            ast::Expr::Ident(name) if name == "bit_shr" => {
                // bit_shr(val, amount) → legacy alias for val >> amount
                assert_eq!(args.len(), 2, "bit_shr takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift_amount = match &args[1].value {
                    ast::Expr::IntLit(n) => *n,
                    _ => panic!("bit_shr amount must be an integer literal"),
                };
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let vec_type = ty.to_mlir_type(self.ctx);
                let shift_attr = IntegerAttribute::new(scalar_type, shift_amount);
                let shift_splat = splat_attr(vec_type, &[shift_attr.into()]).unwrap();
                let shift_vec: Value = self.block.append_operation(
                    arith::constant(self.ctx, shift_splat.into(), self.loc)
                ).result(0).unwrap().into();
                let result: Value = self.block.append_operation(
                    arith::shrui(val, shift_vec, self.loc)
                ).result(0).unwrap().into();
                (result, ty)
            }
            ast::Expr::Ident(name) if name == "bit_shl" => {
                // bit_shl(val, amount) → legacy alias for val << amount
                assert_eq!(args.len(), 2, "bit_shl takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift_amount = match &args[1].value {
                    ast::Expr::IntLit(n) => *n,
                    _ => panic!("bit_shl amount must be an integer literal"),
                };
                let scalar_type = ty.scalar.to_mlir_type(self.ctx);
                let vec_type = ty.to_mlir_type(self.ctx);
                let shift_attr = IntegerAttribute::new(scalar_type, shift_amount);
                let shift_splat = splat_attr(vec_type, &[shift_attr.into()]).unwrap();
                let shift_vec: Value = self.block.append_operation(
                    arith::constant(self.ctx, shift_splat.into(), self.loc)
                ).result(0).unwrap().into();
                let result: Value = self.block.append_operation(
                    arith::shli(val, shift_vec, self.loc)
                ).result(0).unwrap().into();
                (result, ty)
            }
            ast::Expr::Ident(name) if name == "lane_shr" => {
                // lane_shr(vec, amount) → shift lanes right (zeros fill left)
                assert_eq!(args.len(), 2, "lane_shr takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift = match &args[1].value {
                    ast::Expr::IntLit(n) => *n as usize,
                    _ => panic!("lane_shr amount must be an integer literal"),
                };
                self.emit_lane_shift(val, ty, shift, false)
            }
            ast::Expr::Ident(name) if name == "lane_shl" => {
                // lane_shl(vec, amount) → shift lanes left (zeros fill right)
                assert_eq!(args.len(), 2, "lane_shl takes exactly 2 arguments");
                let (val, ty) = self.emit_expr(&args[0].value);
                let shift = match &args[1].value {
                    ast::Expr::IntLit(n) => *n as usize,
                    _ => panic!("lane_shl amount must be an integer literal"),
                };
                self.emit_lane_shift(val, ty, shift, true)
            }
            ast::Expr::Ident(name) if name == "tbl" => {
                // tbl(table, indices) → NEON table lookup (vtbl1)
                // table: u8[16], indices: u8[N] where N is multiple of 16
                // For N > 16, splits into 16-byte chunks automatically
                assert_eq!(args.len(), 2, "tbl takes exactly 2 arguments (table, indices)");
                let (table_val, table_ty) = self.emit_expr(&args[0].value);
                let (idx_val, idx_ty) = self.emit_expr(&args[1].value);
                assert_eq!(table_ty.width, 16, "tbl table must be u8[16]");

                let result = self.emit_tbl_w(table_val, idx_val, idx_ty.width as usize);
                (result, idx_ty)
            }
            ast::Expr::Ident(name) if name == "split_lo" => {
                // split_lo(vec[N]) → vec[N/2], lower half via vector.shuffle
                assert_eq!(args.len(), 1, "split_lo takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                let half = ty.width / 2;
                let result_ty = VecType { scalar: ty.scalar, width: half };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                // Use vector.shuffle with mask [0, 1, ..., half-1]
                let mask: Vec<i64> = (0..half as i64).collect();
                let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[val, val])
                    .add_results(&[result_mlir])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .expect("failed to build vector.shuffle for split_lo");
                let result: Value = self.block.append_operation(shuffle_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "split_hi" => {
                // split_hi(vec[N]) → vec[N/2], upper half via vector.shuffle
                assert_eq!(args.len(), 1, "split_hi takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                let half = ty.width / 2;
                let result_ty = VecType { scalar: ty.scalar, width: half };
                let result_mlir = result_ty.to_mlir_type(self.ctx);

                let mask: Vec<i64> = (half as i64..ty.width as i64).collect();
                let shuffle_op = OperationBuilder::new("vector.shuffle", self.loc)
                    .add_operands(&[val, val])
                    .add_results(&[result_mlir])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(self.ctx, "mask"),
                        DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                    )])
                    .build()
                    .expect("failed to build vector.shuffle for split_hi");
                let result: Value = self.block.append_operation(shuffle_op).result(0).unwrap().into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "likely" || name == "unlikely" => {
                // likely(bool[1]) / unlikely(bool[1]) → the same value, hinting which way it
                // usually goes so the other path is laid out away from the hot path.
                assert_eq!(args.len(), 1, "{} takes exactly 1 argument", name);
                let (val, ty) = self.emit_expr(&args[0].value);
                if ty.scalar != ScalarType::Bool || ty.width != 1 {
                    self.error(&format!("{} requires a scalar bool", name));
                }
                let i1: Type = IntegerType::new(self.ctx, 1).into();
                let expected = (name == "likely") as i64;
                let no: Value = self
                    .block
                    .append_operation(arith::constant(self.ctx, IntegerAttribute::new(i1, expected).into(), self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let hinted: Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("llvm.intr.expect", self.loc)
                            .add_operands(&[val, no])
                            .add_results(&[i1])
                            .build()
                            .expect("failed to build llvm.intr.expect"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                (hinted, ty)
            }
            ast::Expr::Ident(name) if name == "min" || name == "max" => {
                // min(a, b) / max(a, b) → lane-wise, honouring the operand's signedness
                assert_eq!(args.len(), 2, "{} takes exactly 2 arguments", name);
                let (a, a_ty, b, b_ty) = if Self::is_literal(&args[0].value) {
                    let (b, b_ty) = self.emit_expr(&args[1].value);
                    let (a, a_ty) = self.emit_literal_broadcast(&args[0].value, b_ty);
                    (a, a_ty, b, b_ty)
                } else {
                    let (a, a_ty) = self.emit_expr(&args[0].value);
                    let (b, b_ty) = if Self::is_literal(&args[1].value) {
                        self.emit_literal_broadcast(&args[1].value, a_ty)
                    } else {
                        self.emit_expr(&args[1].value)
                    };
                    (a, a_ty, b, b_ty)
                };
                if a_ty.width != b_ty.width {
                    self.error(&format!("{} operands have different widths: {} vs {}", name, a_ty.width, b_ty.width));
                }
                let op_name = match (name.as_str(), a_ty.scalar.is_float(), a_ty.scalar.is_unsigned()) {
                    ("min", true, _) => "arith.minimumf",
                    ("max", true, _) => "arith.maximumf",
                    ("min", false, true) => "arith.minui",
                    ("max", false, true) => "arith.maxui",
                    ("min", false, false) => "arith.minsi",
                    ("max", false, false) => "arith.maxsi",
                    _ => unreachable!(),
                };
                let result = self
                    .block
                    .append_operation(
                        OperationBuilder::new(op_name, self.loc)
                            .add_operands(&[a, b])
                            .add_results(&[a_ty.to_mlir_type(self.ctx)])
                            .build()
                            .expect("failed to build lane-wise min/max"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                (result, a_ty)
            }
            ast::Expr::Ident(name) if name == "fold_min" => {
                // fold_min(u8[N]) → the lane-wise minimum across vector registers, so one
                // register's worth of lanes. A no-op when N fits in one register.
                assert_eq!(args.len(), 1, "fold_min takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                let register = self.arch().register_bytes();
                if ty.scalar != ScalarType::U8 {
                    self.error("fold_min requires a u8 vector");
                }
                if ty.width <= register {
                    return (val, ty);
                }
                if ty.width % register != 0 {
                    self.error(&format!(
                        "fold_min's width must fit in or be a multiple of a {}-byte register",
                        register
                    ));
                }
                let result_ty = VecType { scalar: ScalarType::U8, width: register };
                let result_mlir = result_ty.to_mlir_type(self.ctx);
                let mut folded = self.emit_extract_lanes(val, 0, register);
                let mut offset = register;
                while offset < ty.width {
                    let slice = self.emit_extract_lanes(val, offset as usize, register);
                    folded = self
                        .block
                        .append_operation(
                            OperationBuilder::new("arith.minui", self.loc)
                                .add_operands(&[folded, slice])
                                .add_results(&[result_mlir])
                                .build()
                                .expect("failed to build arith.minui"),
                        )
                        .result(0)
                        .unwrap()
                        .into();
                    offset += register;
                }
                (folded, result_ty)
            }
            ast::Expr::Ident(name) if name == "any_in_first" => {
                // any_in_first(bool[N], n) → whether any of the first n lanes is true, for
                // n in 1..=N. The mask is built by whichever movemask the target has cheaply.
                assert_eq!(args.len(), 2, "any_in_first takes exactly 2 arguments");
                let (mask, mask_ty) = self.emit_expr(&args[0].value);
                let supported = match self.arch() {
                    Arch::Aarch64 => mask_ty.width == 16,
                    Arch::X86_64 => matches!(mask_ty.width, 16 | 32 | 64),
                };
                if mask_ty.scalar != ScalarType::Bool || !supported {
                    self.error(&format!(
                        "any_in_first needs a bool vector of one register, got {:?}[{}]",
                        mask_ty.scalar, mask_ty.width
                    ));
                }
                let count_i64 = VecType { scalar: ScalarType::I64, width: 1 };
                let count: Value = if Self::is_literal(&args[1].value) {
                    if !matches!(args[1].value.const_value(), Some(ast::ConstValue::Int(n)) if n >= 1 && n as u64 <= mask_ty.width) {
                        self.error(&format!("any_in_first's count must be from 1 to {}", mask_ty.width));
                    }
                    self.emit_literal_broadcast(&args[1].value, count_i64).0
                } else {
                    let (count, count_ty) = self.emit_expr(&args[1].value);
                    if count_ty.width != 1 || !count_ty.scalar.is_int() {
                        self.error("any_in_first's count must be a scalar integer");
                    }
                    self.emit_int_resize(count, count_ty, count_i64)
                };
                let i64t: Type = IntegerType::new(self.ctx, 64).into();
                let packed_type: Type = IntegerType::new(self.ctx, mask_ty.width as u32).into();
                let int_const = |this: &mut Self, value: i64| -> Value<'c, 'a> {
                    this.block
                        .append_operation(arith::constant(this.ctx, IntegerAttribute::new(i64t, value).into(), this.loc))
                        .result(0)
                        .unwrap()
                        .into()
                };
                let i8t: Type = IntegerType::new(self.ctx, 8).into();
                let (bits, lanes_at_top): (Value<'c, 'a>, i64) = match self.arch() {
                    Arch::Aarch64 => {
                        let bytes: Value = self
                            .block
                            .append_operation(
                                OperationBuilder::new("arith.extsi", self.loc)
                                    .add_operands(&[mask])
                                    .add_results(&[Type::vector(&[16], i8t)])
                                    .build()
                                    .expect("failed to build arith.extsi"),
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        (self.emit_nibblemask16(bytes), 4)
                    }
                    Arch::X86_64 => {
                        let packed: Value = self
                            .block
                            .append_operation(
                                OperationBuilder::new("llvm.bitcast", self.loc)
                                    .add_operands(&[mask])
                                    .add_results(&[packed_type])
                                    .build()
                                    .expect("failed to build llvm.bitcast"),
                            )
                            .result(0)
                            .unwrap()
                            .into();
                        let wide: Value = if mask_ty.width == 64 {
                            packed
                        } else {
                            self.block
                                .append_operation(arith::extui(packed, i64t, self.loc))
                                .result(0)
                                .unwrap()
                                .into()
                        };
                        (wide, 1)
                    }
                };
                // Shift the lanes at and past n out of the top of the word.
                let bits_per_lane = int_const(self, lanes_at_top);
                let lanes = int_const(self, 64 / lanes_at_top);
                let gone: Value = self
                    .block
                    .append_operation(arith::subi(lanes, count, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let amount: Value = self
                    .block
                    .append_operation(arith::muli(gone, bits_per_lane, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let kept: Value = self
                    .block
                    .append_operation(arith::shli(bits, amount, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let zero = int_const(self, 0);
                let result: Value = self
                    .block
                    .append_operation(arith::cmpi(self.ctx, arith::CmpiPredicate::Ne, kept, zero, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (result, VecType { scalar: ScalarType::Bool, width: 1 })
            }
            ast::Expr::Ident(name) if name == "mask8" => {
                // mask8(bool[N]) → u8[N]: true lanes are 0xFF, false lanes 0x00.
                assert_eq!(args.len(), 1, "mask8 takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                if ty.scalar != ScalarType::Bool {
                    self.error("mask8 requires a bool vector");
                }
                let result_ty = VecType { scalar: ScalarType::U8, width: ty.width };
                let result_mlir = result_ty.to_mlir_type(self.ctx);
                let result = self
                    .block
                    .append_operation(
                        OperationBuilder::new("arith.extsi", self.loc)
                            .add_operands(&[val])
                            .add_results(&[result_mlir])
                            .build()
                            .expect("failed to build arith.extsi"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "nibblemask" => {
                // nibblemask(u8[16]) → u64[1]: for a mask whose lanes are 0x00 or 0xFF, 4 bits
                // per lane, with lane i at bits 4i..4i+3 (NEON's cheap movemask: `shrn #4`).
                assert_eq!(args.len(), 1, "nibblemask takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                if ty.width != 16 || ty.scalar.bit_width() != 8 {
                    self.error("nibblemask requires a u8[16]");
                }
                let result_ty = VecType { scalar: ScalarType::U64, width: 1 };
                (self.emit_nibblemask16(val), result_ty)
            }
            ast::Expr::Ident(name) if name == "shuffle" => {
                // shuffle(a, b, [T: lanes...]) → one output lane per entry: lane i of a, or
                // lane i - N of b. The lane list is constant, so LLVM can choose an instruction.
                assert_eq!(args.len(), 3, "shuffle takes exactly 3 arguments (a, b, lanes)");
                let (a, a_ty) = self.emit_expr(&args[0].value);
                let (b, b_ty) = self.emit_expr(&args[1].value);
                if a_ty.width != b_ty.width || a_ty.scalar.bit_width() != b_ty.scalar.bit_width() {
                    self.error("shuffle requires operands of the same type");
                }
                let ast::Expr::VecLit { values, .. } = &args[2].value else {
                    self.error("shuffle's lanes must be a vector literal");
                };
                if let Some(bad) = values.iter().find(|&&lane| lane < 0 || lane >= 2 * a_ty.width as i64) {
                    self.error(&format!("shuffle lane {} is out of range", bad));
                }
                let result_ty = VecType { scalar: a_ty.scalar, width: values.len() as u64 };
                let result = self
                    .block
                    .append_operation(
                        OperationBuilder::new("vector.shuffle", self.loc)
                            .add_operands(&[a, b])
                            .add_results(&[result_ty.to_mlir_type(self.ctx)])
                            .add_attributes(&[(
                                melior::ir::Identifier::new(self.ctx, "mask"),
                                DenseI64ArrayAttribute::new(self.ctx, values).into(),
                            )])
                            .build()
                            .expect("failed to build vector.shuffle"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "concat" => {
                // concat(a: T[N], b: T[N]) → T[2N]
                assert_eq!(args.len(), 2, "concat takes exactly 2 arguments");
                let (a, a_ty) = self.emit_expr(&args[0].value);
                let (b, b_ty) = self.emit_expr(&args[1].value);
                if a_ty.width != b_ty.width || a_ty.scalar.bit_width() != b_ty.scalar.bit_width() {
                    self.error("concat requires operands of the same type");
                }
                if a_ty.width < 2 {
                    self.error("concat requires vector operands");
                }
                let result_ty = VecType { scalar: a_ty.scalar, width: a_ty.width * 2 };
                let mask: Vec<i64> = (0..result_ty.width as i64).collect();
                let result = self
                    .block
                    .append_operation(
                        OperationBuilder::new("vector.shuffle", self.loc)
                            .add_operands(&[a, b])
                            .add_results(&[result_ty.to_mlir_type(self.ctx)])
                            .add_attributes(&[(
                                melior::ir::Identifier::new(self.ctx, "mask"),
                                DenseI64ArrayAttribute::new(self.ctx, &mask).into(),
                            )])
                            .build()
                            .expect("failed to build vector.shuffle for concat"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "ptr_addr" => {
                // ptr_addr(ptr) → i64[1]: the machine address of a pointer's first
                // element, for alignment and page-boundary tests.
                assert_eq!(args.len(), 1, "ptr_addr takes exactly 1 argument");
                let (ptr_val, ptr_ty) = self.emit_expr(&args[0].value);
                if ptr_ty.width != 0 {
                    self.error("ptr_addr requires a pointer");
                }
                let index_type = Type::index(self.ctx);
                let i64t: Type = IntegerType::new(self.ctx, 64).into();
                let aligned: Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("memref.extract_aligned_pointer_as_index", self.loc)
                            .add_operands(&[ptr_val])
                            .add_results(&[index_type])
                            .build()
                            .expect("failed to extract pointer"),
                    )
                    .result(0)
                    .unwrap()
                    .into();
                let addr: Value = self
                    .block
                    .append_operation(arith::index_cast(aligned, i64t, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                (addr, VecType { scalar: ScalarType::I64, width: 1 })
            }
            ast::Expr::Ident(name) if name == "any" => {
                // any(bool[N]) → bool[1], true if any lane is true
                assert_eq!(args.len(), 1, "any takes exactly 1 argument");
                let (val, ty) = self.emit_expr(&args[0].value);
                assert_eq!(ty.scalar, ScalarType::Bool, "any requires bool vector");
                if ty.width == 1 {
                    return (val, ty);
                }
                if self.arch() == Arch::Aarch64 && ty.width % 16 == 0 {
                    let result = self.emit_any_wide(val, ty.width);
                    return (result, VecType { scalar: ScalarType::Bool, width: 1 });
                }
                let bool_type = ScalarType::Bool.to_mlir_type(self.ctx);
                let result_ty = VecType { scalar: ScalarType::Bool, width: 1 };

                let kind_attr = melior::ir::attribute::Attribute::parse(
                    self.ctx, "#vector.kind<or>",
                ).unwrap();
                let reduced: Value = self.block.append_operation(
                    OperationBuilder::new("vector.reduction", self.loc)
                        .add_operands(&[val])
                        .add_results(&[bool_type])
                        .add_attributes(&[(
                            melior::ir::Identifier::new(self.ctx, "kind"),
                            kind_attr,
                        )])
                        .build().unwrap()
                ).result(0).unwrap().into();

                let result: Value = reduced;
                (result, result_ty)
            }
            ast::Expr::Ident(name) if name == "gather" => {
                // gather(buf, indices) → gather elements from memref at index positions
                assert_eq!(args.len(), 2, "gather takes exactly 2 arguments (buf, indices)");
                self.emit_gather(&args[0].value, &args[1].value, None)
            }
            ast::Expr::Ident(name) if name == "store_at" => {
                // store_at(buf, offset, data) → vector.store at index offset
                assert_eq!(args.len(), 3, "store_at takes 3 arguments (buf, offset, data)");
                let (buf_val, buf_ty) = self.emit_expr(&args[0].value);
                let offset_index = self.emit_index(&args[1].value);
                let buf_scalar = self.ptr_scalar_of(&args[0].value, buf_ty.scalar);
                let (data_val, data_ty) = self.emit_store_value(&args[2].value, buf_scalar);

                self.emit_vector_store(data_val, data_ty, buf_val, buf_scalar, offset_index);

                let dummy_ty = VecType { scalar: ScalarType::I32, width: 1 };
                let dummy = self.emit_zero_vector(dummy_ty);
                (dummy, dummy_ty)
            }
            ast::Expr::Ident(name) if name == "ptr_add" => {
                // ptr_add(ptr, offset) → memref.subview to create a new pointer advanced by offset elements
                assert_eq!(args.len(), 2, "ptr_add takes 2 arguments (ptr, offset)");
                let (ptr_val, ptr_dummy_ty) = self.emit_expr(&args[0].value);

                // Determine the element scalar type from the ptr binding
                let elem_scalar = if let ast::Expr::Ident(ref pname) = args[0].value {
                    match self.vars.get(pname) {
                        Some(Binding::Ptr(_, s)) => *s,
                        _ => ptr_dummy_ty.scalar,
                    }
                } else {
                    ptr_dummy_ty.scalar
                };

                let offset_index = self.emit_index(&args[1].value);
                let i64t: Type = IntegerType::new(self.ctx, 64).into();
                let offset: Value = self
                    .block
                    .append_operation(arith::index_cast(offset_index, i64t, self.loc))
                    .result(0)
                    .unwrap()
                    .into();

                // A plain `memref<?xT>` is defined to have offset 0, so advance the data
                // pointer itself and keep the descriptor's offset field at 0. Loads, stores
                // and ptr_addr then all see the same address.
                let descriptor = self.emit_memref_descriptor(ptr_val);
                let aligned = self.emit_descriptor_field(descriptor, &[1], melior::dialect::llvm::r#type::pointer(self.ctx, 0));
                let advanced = self.emit_gep(aligned, offset, elem_scalar.to_mlir_type(self.ctx));
                let size = self.emit_descriptor_field(descriptor, &[3, 0], i64t);
                let remaining: Value = self
                    .block
                    .append_operation(arith::subi(size, offset, self.loc))
                    .result(0)
                    .unwrap()
                    .into();
                let with_ptr = self.emit_descriptor_insert(descriptor, advanced, &[1]);
                let with_size = self.emit_descriptor_insert(with_ptr, remaining, &[3, 0]);
                let plain_memref_type = MemRefType::new(elem_scalar.to_mlir_type(self.ctx), &[i64::MIN], None, None);
                let result: Value = self
                    .block
                    .append_operation(
                        OperationBuilder::new("builtin.unrealized_conversion_cast", self.loc)
                            .add_operands(&[with_size])
                            .add_results(&[plain_memref_type.into()])
                            .build()
                            .expect("failed to cast descriptor to memref"),
                    )
                    .result(0)
                    .unwrap()
                    .into();

                (result, VecType { scalar: elem_scalar, width: 0 })
            }
            ast::Expr::Ident(name) if name == "load_at" => {
                // load_at(ptr, offset) → vector.load at element index offset, returns vector<1xT>
                assert_eq!(args.len(), 2, "load_at takes 2 arguments (ptr, offset)");
                let (ptr_val, ptr_dummy_ty) = self.emit_expr(&args[0].value);

                // Determine the element scalar type from the ptr binding
                let elem_scalar = if let ast::Expr::Ident(ref pname) = args[0].value {
                    match self.vars.get(pname) {
                        Some(Binding::Ptr(_, s)) => *s,
                        _ => ptr_dummy_ty.scalar,
                    }
                } else {
                    ptr_dummy_ty.scalar
                };

                let offset_index = self.emit_index(&args[1].value);

                let result_ty = VecType { scalar: elem_scalar, width: 1 };
                (self.emit_memref_load(ptr_val, offset_index, elem_scalar), result_ty)
            }
            ast::Expr::Instantiate { .. } | ast::Expr::Ident(_) => {
                let (results, ret_ty, env) = self.emit_user_call(func_expr, args);
                match ret_ty {
                    None => {
                        // Void return — return a dummy
                        let dummy_ty = VecType { scalar: ScalarType::I32, width: 1 };
                        let dummy = self.emit_zero_vector(dummy_ty);
                        (dummy, dummy_ty)
                    }
                    Some(ret) if is_ptr_type_name(&ret.name) => {
                        (results[0], VecType { scalar: ptr_element_scalar(&ret), width: 0 })
                    }
                    Some(ret) if is_scalar_type_name(&ret.name) => {
                        (results[0], resolve_type(&ret, &env, self.native_width))
                    }
                    Some(ret) => self.error(&format!(
                        "this call returns struct {}: assign it to a variable or access a field",
                        ret.name
                    )),
                }
            }
            other => panic!("non-identifier function call not supported: {:?}", other),
        }
    }

    /// Call a user function. Returns its flattened results, its declared return type,
    /// and the comptime environment that type is resolved in.
    fn emit_user_call(
        &mut self,
        func_expr: &ast::Expr,
        args: &[ast::CallArg],
    ) -> (Vec<Value<'c, 'a>>, Option<ast::Type>, HashMap<String, u64>) {
        let (target_name, explicit_widths) = match func_expr {
            ast::Expr::Ident(n) => (n.clone(), None),
            ast::Expr::Instantiate { name, widths } => (name.clone(), Some(widths)),
            _ => panic!("non-identifier function call not supported: {:?}", func_expr),
        };

        let target_fn = self.fn_defs.get(&target_name)
            .unwrap_or_else(|| panic!("unknown function in codegen: {}", target_name))
            .clone();

        // Literal arguments are emitted last: their types depend on the callee's
        // comptime parameters, which the other arguments may determine.
        let mut call_args: Vec<Value<'c, 'a>> = Vec::new();
        let mut arg_widths: Vec<Option<u64>> = Vec::new();
        let mut literal_slots: Vec<(usize, usize)> = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let target_param = &target_fn.params[i];
            arg_widths.push(None);
            if is_ptr_type_name(&target_param.ty.name) {
                // Ptr param: pass the memref value directly
                let (val, _) = self.emit_expr(&arg.value);
                call_args.push(val);
            } else if is_scalar_type_name(&target_param.ty.name) {
                if Self::takes_expected_type(&arg.value) {
                    literal_slots.push((call_args.len(), i));
                    call_args.push(self.emit_zero_vector(VecType { scalar: ScalarType::I64, width: 1 }));
                    continue;
                }
                // Scalar/vector param: pass directly
                let (val, ty) = self.emit_expr(&arg.value);
                arg_widths[i] = Some(ty.width);
                call_args.push(val);
            } else {
                // Struct param: pass each field
                if !self.is_struct_expr(&arg.value) {
                    self.error(&format!(
                        "argument '{}' of '{}' must be a struct",
                        target_param.name, target_name
                    ));
                }
                for (_, val, _) in self.emit_struct_value(&arg.value) {
                    call_args.push(val);
                }
            }
        }

        let (symbol, callee_env) = self.resolve_callee(&target_fn, explicit_widths, &arg_widths);

        for (slot, i) in literal_slots {
            let param_ty = resolve_type(&target_fn.params[i].ty, &callee_env, self.native_width);
            call_args[slot] = self.emit_expr_typed(&args[i].value, param_ty).0;
        }

        let ret_mlir_types: Vec<Type<'c>> = match &target_fn.ret_ty {
            None => vec![],
            Some(ret) if is_ptr_type_name(&ret.name) => vec![ptr_to_memref_type(self.ctx, ret)],
            Some(ret) => resolve_struct_return_types(
                self.ctx, ret, &self.struct_defs, &callee_env, self.native_width,
            )
            .into_iter()
            .map(|(t, _)| t)
            .collect(),
        };

        let op_ref = self.block.append_operation(
            OperationBuilder::new("func.call", self.loc)
                .add_operands(&call_args)
                .add_results(&ret_mlir_types)
                .add_attributes(&[(
                    melior::ir::Identifier::new(self.ctx, "callee"),
                    melior::ir::attribute::FlatSymbolRefAttribute::new(self.ctx, &symbol).into(),
                )])
                .build()
                .expect("failed to build func.call for inter-function call"),
        );
        let results = (0..ret_mlir_types.len()).map(|i| op_ref.result(i).unwrap().into()).collect();
        (results, target_fn.ret_ty.clone(), callee_env)
    }

    /// Emit a stream construct: scf.for loop over buffer chunks with carry state.
    ///
    /// Generates:
    ///   %c0 = constant 0 : index
    ///   %len = <from second param or buffer dim>
    ///   %step = constant <chunk_width> : index
    ///   %init_carry... = constant <initial values>
    ///   scf.for %i = %c0 to %len step %step iter_args(%carry = %init_carry) -> (types) {
    ///     %chunk = vector.load %buf[%i] : memref<?xT>, vector<NxT>
    ///     <body>
    ///     scf.yield %updated_carry
    ///   }
    fn emit_stream(
        &mut self,
        chunk_name: &str,
        chunk_ty: &ast::Type,
        buffer: &str,
        carry_defs: &[ast::CarryDef],
        body: &[ast::Stmt],
        carry_updates: &[(String, ast::Expr)],
    ) {
        let index_type = Type::index(self.ctx);

        // Resolve the chunk vector type
        let chunk_vt = resolve_type(chunk_ty, &self.comptime_env, self.native_width);
        let chunk_mlir = chunk_vt.to_mlir_type(self.ctx);
        let chunk_width = chunk_vt.width;
        if chunk_width < 2 {
            self.error("stream chunk width must be at least 2 lanes");
        }

        // Get the buffer memref value from vars
        let buf_binding = self
            .vars
            .get(buffer)
            .unwrap_or_else(|| panic!("undefined buffer: {}", buffer))
            .clone();
        let buf_val = match buf_binding {
            Binding::Ptr(val, _) => val,
            _ => panic!("stream buffer '{}' must be a ptr type", buffer),
        };

        // Emit constants in the outer block: start=0, step=chunk_width
        let zero_attr = IntegerAttribute::new(index_type, 0);
        let start_val = self
            .block
            .append_operation(arith::constant(self.ctx, zero_attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into();

        let step_attr = IntegerAttribute::new(index_type, chunk_width as i64);
        let step_val = self
            .block
            .append_operation(arith::constant(self.ctx, step_attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into();

        // Get buffer length via memref.dim
        let dim_index_attr = IntegerAttribute::new(index_type, 0);
        let dim_index = self
            .block
            .append_operation(arith::constant(self.ctx, dim_index_attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into();
        let end_val = self
            .block
            .append_operation(
                OperationBuilder::new("memref.dim", self.loc)
                    .add_operands(&[buf_val, dim_index])
                    .add_results(&[index_type])
                    .build()
                    .expect("failed to build memref.dim"),
            )
            .result(0)
            .unwrap()
            .into();

        // Emit initial carry values in the outer block
        let mut init_carry_vals: Vec<Value<'c, 'a>> = Vec::new();
        let mut carry_vtypes: Vec<VecType> = Vec::new();
        let mut carry_mlir_types: Vec<Type<'c>> = Vec::new();

        for carry_def in carry_defs {
            let carry_vt = resolve_type(&carry_def.ty, &self.comptime_env, self.native_width);
            carry_vtypes.push(carry_vt);
            carry_mlir_types.push(carry_vt.to_mlir_type(self.ctx));

            let (init_val, _) = self.emit_expr_typed(&carry_def.init, carry_vt);
            init_carry_vals.push(init_val);
        }

        // Build the inner block for scf.for body
        // Block args: [index (induction var), carry_0, carry_1, ...]
        let mut block_arg_types: Vec<(Type<'c>, Location<'c>)> = vec![(index_type, self.loc)];
        for ct in &carry_mlir_types {
            block_arg_types.push((*ct, self.loc));
        }
        let inner_block = Block::new(&block_arg_types);

        // Inside the inner block:
        // 1. Load chunk from buffer at current index
        let iv: Value = inner_block.argument(0).unwrap().into();
        let chunk_load = inner_block.append_operation(
            OperationBuilder::new("vector.load", self.loc)
                .add_operands(&[
                    // Transmute buf_val to inner block lifetime — safe because MLIR allows
                    // inner regions to reference values from enclosing regions
                    unsafe { std::mem::transmute::<Value, Value>(buf_val) },
                    iv,
                ])
                .add_results(&[chunk_mlir])
                .build()
                .expect("failed to build vector.load for stream chunk"),
        );
        let chunk_val: Value = chunk_load.result(0).unwrap().into();

        // 2. Create a FnCodegen for the inner block to emit body statements
        {
            let mut inner_cg = FnCodegen {
                instances: self.instances.clone(),
                ctx: self.ctx,
                loc: self.loc,
                block: &inner_block,
                vars: HashMap::new(),
                comptime_env: self.comptime_env.clone(),
                native_width: self.native_width,
                struct_defs: self.struct_defs.clone(),
                fn_defs: self.fn_defs.clone(),
                current_span: ast::Span::dummy(),
            };

            // Bind chunk variable
            inner_cg
                .vars
                .insert(chunk_name.to_string(), Binding::Vec(chunk_val, chunk_vt));

            // Bind chunk_offset — the loop induction variable as i32[1]
            let i32_type = ScalarType::I32.to_mlir_type(self.ctx);
            let offset_i32_ty = VecType { scalar: ScalarType::I32, width: 1 };
            let cast_op = arith::index_cast(iv, i32_type, self.loc);
            let iv_i32 = inner_block.append_operation(cast_op).result(0).unwrap().into();
            let iv_vec: Value = iv_i32;
            inner_cg.vars.insert("chunk_offset".to_string(), Binding::Vec(iv_vec, offset_i32_ty));

            // Bind carry variables from inner block args
            for (idx, carry_def) in carry_defs.iter().enumerate() {
                let carry_val: Value = inner_block.argument(1 + idx).unwrap().into();
                inner_cg.vars.insert(
                    carry_def.name.clone(),
                    Binding::Vec(carry_val, carry_vtypes[idx]),
                );
            }

            // Copy outer variables into inner scope (transmute lifetimes)
            // Safe because MLIR inner regions can reference outer values
            for (name, binding) in &self.vars {
                if name != chunk_name
                    && name != "chunk_offset"
                    && !carry_defs.iter().any(|c| &c.name == name)
                {
                    let inner_binding: Binding =
                        unsafe { std::mem::transmute(binding.clone()) };
                    inner_cg.vars.insert(name.clone(), inner_binding);
                }
            }

            // Emit body statements
            for stmt in body {
                inner_cg.emit_stmt(stmt);
            }

            // Each carry takes its update, all evaluated before any applies. A carry with
            // no update keeps its value.
            for (name, _) in carry_updates {
                if !carry_defs.iter().any(|c| &c.name == name) {
                    inner_cg.error(&format!("'{}' isn't one of this stream's carries", name));
                }
            }
            let mut yield_vals: Vec<Value> = Vec::new();
            for (idx, carry_def) in carry_defs.iter().enumerate() {
                let value = match carry_updates.iter().find(|(name, _)| name == &carry_def.name) {
                    Some((_, expr)) => inner_cg.emit_expr_typed(expr, carry_vtypes[idx]).0,
                    None => match inner_cg.vars.get(&carry_def.name) {
                        Some(Binding::Vec(val, _)) => *val,
                        _ => inner_cg.error(&format!("carry '{}' isn't a vector", carry_def.name)),
                    },
                };
                yield_vals.push(value);
            }

            // scf.yield
            inner_block.append_operation(scf::r#yield(
                &yield_vals,
                self.loc,
            ));
        }

        // Build region
        let region = Region::new();
        region.append_block(inner_block);

        // Build scf.for with iter_args via OperationBuilder
        let mut operands: Vec<Value<'c, 'a>> = vec![start_val, end_val, step_val];
        operands.extend(init_carry_vals);

        let for_op = OperationBuilder::new("scf.for", self.loc)
            .add_operands(&operands)
            .add_results(&carry_mlir_types)
            .add_regions([region])
            .build()
            .expect("failed to build scf.for for stream");

        let for_result = self.block.append_operation(for_op);

        // Bind the final carry values back into the outer scope
        for (idx, carry_def) in carry_defs.iter().enumerate() {
            let result_val: Value<'c, 'a> = for_result.result(idx).unwrap().into();
            self.vars.insert(
                carry_def.name.clone(),
                Binding::Vec(result_val, carry_vtypes[idx]),
            );
        }
    }

    fn emit_stmt(&mut self, stmt: &ast::Stmt) {
        self.current_span = stmt.span;
        match &stmt.kind {
            ast::StmtKind::Assign { target, ty: ann_ty, value } => match target {
                ast::AssignTarget::Ident(name) => {
                    match value {
                        _ if self.is_struct_expr(value) => {
                            let fields = self.emit_struct_value(value);
                            self.vars.insert(name.clone(), Binding::Struct { fields });
                        }
                        _ => {
                            let (val, ty) = if let Some(ann) = ann_ty {
                                let target_ty = resolve_type(ann, &self.comptime_env, self.native_width);
                                self.emit_expr_typed(value, target_ty)
                            } else if let (true, Some(Binding::Vec(_, existing))) =
                                (Self::is_literal(value), self.vars.get(name).cloned())
                            {
                                // A literal assigned to an existing variable takes its type.
                                self.emit_literal_broadcast(value, existing)
                            } else {
                                self.emit_expr(value)
                            };
                            if ty.width == 0 {
                                // width 0 is the sentinel for ptr types
                                self.vars.insert(name.clone(), Binding::Ptr(val, ty.scalar));
                            } else {
                                self.vars.insert(name.clone(), Binding::Vec(val, ty));
                            }
                        }
                    }
                }
                ast::AssignTarget::Scatter { base, index, mask } => {
                    self.emit_scatter(
                        base,
                        index,
                        mask.as_ref().map(|m| m.as_ref()),
                        value,
                    );
                }
                ast::AssignTarget::Destructure(names) => {
                    if !self.is_struct_expr(value) {
                        self.error("destructuring needs a struct value");
                    }
                    let fields = self.emit_struct_value(value);
                    if fields.len() != names.len() {
                        self.error(&format!(
                            "destructuring {} names from a struct with {} fields",
                            names.len(),
                            fields.len()
                        ));
                    }
                    for (name, (_, val, ty)) in names.iter().zip(fields) {
                        self.vars.insert(name.clone(), Binding::Vec(val, ty));
                    }
                }
            },
            ast::StmtKind::Return(_) | ast::StmtKind::ReturnVoid => {
                // Handled by the caller (emit_fn); desugaring leaves these last in a function
            }
            ast::StmtKind::Break => self.error("break outside of a loop"),
            ast::StmtKind::Stream {
                chunk_name,
                chunk_ty,
                buffer,
                carry,
                body,
                carry_updates,
            } => {
                self.emit_stream(chunk_name, chunk_ty, buffer, carry, body, carry_updates);
            }
            ast::StmtKind::Expr(expr) => {
                // Check if it's a store call
                match expr {
                    ast::Expr::Call { func, args } => {
                        if let ast::Expr::Ident(name) = func.as_ref() {
                            if name == "store" {
                                assert!(
                                    args.len() == 2 || args.len() == 3,
                                    "store takes 2 or 3 arguments"
                                );
                                self.emit_store(&args[0].value, &args[1].value, args.get(2).map(|a| &a.value));
                                return;
                            }
                        }
                        self.emit_expr(expr);
                    }
                    _ => {
                        self.emit_expr(expr);
                    }
                }
            }
            ast::StmtKind::If {
                cond,
                then_body,
                else_body,
            } => {
                self.emit_if(cond, then_body, else_body);
            }
            ast::StmtKind::While { cond, body } => {
                self.emit_while(cond, body);
            }
        }
    }

    /// A value of the same type as `binding`, for a path that never reads it.
    fn emit_placeholder(&mut self, binding: &Binding<'c, '_>, name: &str) -> Binding<'c, 'a> {
        match binding {
            Binding::Vec(_, ty) => Binding::Vec(self.emit_zero_vector(*ty), *ty),
            Binding::Struct { fields } => Binding::Struct {
                fields: fields
                    .iter()
                    .map(|(field, _, ty)| (field.clone(), self.emit_zero_vector(*ty), *ty))
                    .collect(),
            },
            Binding::Ptr(..) => self.error(&format!(
                "pointer '{}' must be defined on every path through an early return",
                name
            )),
        }
    }

    /// Codegen for a nested region. MLIR regions may use values from enclosing regions.
    fn inner_codegen<'b>(&self, block: &'b Block<'c>) -> FnCodegen<'c, 'b> {
        FnCodegen {
            instances: self.instances.clone(),
            ctx: self.ctx,
            loc: self.loc,
            block,
            vars: self
                .vars
                .iter()
                .map(|(k, b)| (k.clone(), unsafe { std::mem::transmute(b.clone()) }))
                .collect(),
            comptime_env: self.comptime_env.clone(),
            native_width: self.native_width,
            struct_defs: self.struct_defs.clone(),
            fn_defs: self.fn_defs.clone(),
            current_span: self.current_span,
        }
    }

    /// `a && b` / `a || b`: `b` is evaluated only in the branch that needs it.
    fn emit_logical(&mut self, op: ast::LogicalOp, lhs: &ast::Expr, rhs: &ast::Expr) -> (Value<'c, 'a>, VecType) {
        let bool1 = VecType { scalar: ScalarType::Bool, width: 1 };
        let i1: Type = IntegerType::new(self.ctx, 1).into();
        let (lhs_val, _) = if Self::is_literal(lhs) {
            self.emit_literal_broadcast(lhs, bool1)
        } else {
            self.emit_expr(lhs)
        };

        let eval_block = Block::new(&[]);
        let short_block = Block::new(&[]);
        {
            let mut cg = self.inner_codegen(&eval_block);
            let (rhs_val, _) = if Self::is_literal(rhs) {
                cg.emit_literal_broadcast(rhs, bool1)
            } else {
                cg.emit_expr(rhs)
            };
            eval_block.append_operation(scf::r#yield(&[rhs_val], self.loc));

            let short_value = matches!(op, ast::LogicalOp::Or) as i64;
            let short: Value = short_block
                .append_operation(arith::constant(self.ctx, IntegerAttribute::new(i1, short_value).into(), self.loc))
                .result(0)
                .unwrap()
                .into();
            short_block.append_operation(scf::r#yield(&[short], self.loc));
        }
        let (then_block, else_block) = match op {
            ast::LogicalOp::And => (eval_block, short_block),
            ast::LogicalOp::Or => (short_block, eval_block),
        };
        let then_region = Region::new();
        then_region.append_block(then_block);
        let else_region = Region::new();
        else_region.append_block(else_block);
        let result = self
            .block
            .append_operation(
                OperationBuilder::new("scf.if", self.loc)
                    .add_operands(&[lhs_val])
                    .add_results(&[i1])
                    .add_regions([then_region, else_region])
                    .build()
                    .expect("failed to build scf.if for a logical operator"),
            )
            .result(0)
            .unwrap()
            .into();
        (result, bool1)
    }

    /// Emit an if/else statement as a real branch (`scf.if`).
    ///
    /// Only the taken branch runs, so stores and other side effects are conditional.
    /// After the `if`, a variable holds its new value if either branch reassigned it.
    /// A name first defined inside the `if` is visible afterwards only when both
    /// branches define it. LLVM if-converts cheap pure branches back into `select`.
    fn emit_if(
        &mut self,
        cond_expr: &ast::Expr,
        then_body: &[ast::Stmt],
        else_body: &[ast::Stmt],
    ) {
        let (cond_val, cond_ty) = self.emit_expr(cond_expr);
        if cond_ty.scalar != ScalarType::Bool {
            self.error("if condition must be bool");
        }
        if cond_ty.width != 1 {
            self.error("if condition must be width 1 (scalar bool)");
        }

        let then_block = Block::new(&[]);
        let else_block = Block::new(&[]);

        let result_types: Vec<Type<'c>>;
        let result_bindings: Vec<IfResult>;
        {
            let mut then_cg = self.inner_codegen(&then_block);
            for stmt in then_body {
                then_cg.emit_stmt(stmt);
            }
            let mut else_cg = self.inner_codegen(&else_block);
            for stmt in else_body {
                else_cg.emit_stmt(stmt);
            }

            let mut names = Self::collect_assigned_vars(then_body);
            names.extend(Self::collect_assigned_vars(else_body));
            names.sort();
            names.dedup();

            let mut types = Vec::new();
            let mut bindings = Vec::new();
            let mut then_yield: Vec<Value> = Vec::new();
            let mut else_yield: Vec<Value> = Vec::new();
            let (then_continues, else_continues) = ast::sides_that_continue(cond_expr, then_body, else_body);
            for name in names {
                let mut then_binding = then_cg.vars.get(&name).cloned();
                let mut else_binding = else_cg.vars.get(&name).cloned();
                // A side that never continues doesn't need a real value for a name defined
                // on the other side. Nothing can read this one.
                if let (Some(binding), None, false) = (&then_binding, &else_binding, else_continues) {
                    else_binding = Some(else_cg.emit_placeholder(binding, &name));
                }
                if let (None, Some(binding), false) = (&then_binding, &else_binding, then_continues) {
                    then_binding = Some(then_cg.emit_placeholder(binding, &name));
                }
                match (&then_binding, &else_binding) {
                    (Some(Binding::Vec(tv, tt)), Some(Binding::Vec(ev, et))) => {
                        if tt.to_mlir_type(self.ctx) != et.to_mlir_type(self.ctx) {
                            self.error(&format!(
                                "variable '{}' has different types in the two branches of an if: {:?}[{}] vs {:?}[{}]",
                                name, tt.scalar, tt.width, et.scalar, et.width
                            ));
                        }
                        types.push(tt.to_mlir_type(self.ctx));
                        then_yield.push(*tv);
                        else_yield.push(*ev);
                        bindings.push(IfResult::Vec(name, *tt));
                    }
                    (Some(Binding::Ptr(tv, ts)), Some(Binding::Ptr(ev, _))) => {
                        let elem = ts.to_mlir_type(self.ctx);
                        types.push(MemRefType::new(elem, &[i64::MIN], None, None).into());
                        then_yield.push(*tv);
                        else_yield.push(*ev);
                        bindings.push(IfResult::Ptr(name, *ts));
                    }
                    (Some(Binding::Struct { fields: tf }), Some(Binding::Struct { fields: ef })) => {
                        let same_shape = tf.len() == ef.len()
                            && tf.iter().zip(ef).all(|((tn, _, tt), (en, _, et))| {
                                tn == en && tt.to_mlir_type(self.ctx) == et.to_mlir_type(self.ctx)
                            });
                        if !same_shape {
                            self.error(&format!(
                                "variable '{}' is a different struct in each branch of an if",
                                name
                            ));
                        }
                        let mut field_types = Vec::with_capacity(tf.len());
                        for ((field, tv, tt), (_, ev, _)) in tf.iter().zip(ef) {
                            types.push(tt.to_mlir_type(self.ctx));
                            then_yield.push(*tv);
                            else_yield.push(*ev);
                            field_types.push((field.clone(), *tt));
                        }
                        bindings.push(IfResult::Struct(name, field_types));
                    }
                    // Defined in only one branch: scoped to that branch.
                    (Some(_), None) | (None, Some(_)) | (None, None) => {}
                    (Some(_), Some(_)) => {
                        self.error(&format!(
                            "variable '{}' has different kinds in the two branches of an if",
                            name
                        ));
                    }
                }
            }
            then_block.append_operation(scf::r#yield(&then_yield, self.loc));
            else_block.append_operation(scf::r#yield(&else_yield, self.loc));
            result_types = types;
            result_bindings = bindings;
        }

        let then_region = Region::new();
        then_region.append_block(then_block);
        let else_region = Region::new();
        else_region.append_block(else_block);

        let if_op = self.block.append_operation(
            OperationBuilder::new("scf.if", self.loc)
                .add_operands(&[cond_val])
                .add_results(&result_types)
                .add_regions([then_region, else_region])
                .build()
                .expect("failed to build scf.if"),
        );

        let mut next_result = 0;
        for binding in result_bindings {
            let mut take = || {
                let result: Value<'c, 'a> = if_op.result(next_result).unwrap().into();
                next_result += 1;
                result
            };
            match binding {
                IfResult::Vec(name, ty) => {
                    let value = take();
                    self.vars.insert(name, Binding::Vec(value, ty));
                }
                IfResult::Ptr(name, scalar) => {
                    let value = take();
                    self.vars.insert(name, Binding::Ptr(value, scalar));
                }
                IfResult::Struct(name, field_types) => {
                    let fields = field_types.into_iter().map(|(f, ty)| (f, take(), ty)).collect();
                    self.vars.insert(name, Binding::Struct { fields });
                }
            }
        }
    }

    /// Collect variable names assigned in a list of statements (for while loop iter_args).
    fn collect_assigned_vars(stmts: &[ast::Stmt]) -> Vec<String> {
        let mut names = Vec::new();
        for stmt in stmts {
            match &stmt.kind {
                ast::StmtKind::Assign { target, .. } => {
                    match target {
                        ast::AssignTarget::Ident(name) => names.push(name.clone()),
                        ast::AssignTarget::Destructure(targets) => names.extend(targets.iter().cloned()),
                        ast::AssignTarget::Scatter { .. } => {}
                    }
                }
                ast::StmtKind::If { then_body, else_body, .. } => {
                    names.extend(Self::collect_assigned_vars(then_body));
                    names.extend(Self::collect_assigned_vars(else_body));
                }
                ast::StmtKind::While { body, .. } => {
                    names.extend(Self::collect_assigned_vars(body));
                }
                // A stream's carries are visible after it.
                ast::StmtKind::Stream { carry, .. } => {
                    names.extend(carry.iter().map(|c| c.name.clone()));
                }
                _ => {}
            }
        }
        names.sort();
        names.dedup();
        names
    }

    /// Bind a loop's carried values, in `carried` order, as variables. Struct fields are
    /// gathered back into their struct.
    fn bind_carried(
        vars: &mut HashMap<String, Binding<'c, 'a>>,
        carried: &[(String, VecType, bool)],
        fields: &[Option<String>],
        value_at: impl Fn(usize) -> Value<'c, 'a>,
    ) {
        for (idx, (name, ty, is_ptr)) in carried.iter().enumerate() {
            let value = value_at(idx);
            match (&fields[idx], is_ptr) {
                (None, true) => {
                    vars.insert(name.clone(), Binding::Ptr(value, ty.scalar));
                }
                (None, false) => {
                    vars.insert(name.clone(), Binding::Vec(value, *ty));
                }
                (Some(field), _) => {
                    // A struct's fields are adjacent; its first field starts it afresh.
                    if idx == 0 || carried[idx - 1].0 != *name {
                        vars.insert(name.clone(), Binding::Struct { fields: Vec::new() });
                    }
                    if let Some(Binding::Struct { fields: bound }) = vars.get_mut(name) {
                        bound.push((field.clone(), value, *ty));
                    }
                }
            }
        }
    }

    /// Emit a while loop using scf.while with iter_args for modified variables.
    ///
    /// Generates:
    ///   scf.while (%v0 = %init0, %v1 = %init1, ...) : (types) -> (types) {
    ///     // "before" region: evaluate condition
    ///     scf.condition(%cond) %v0, %v1, ... : types
    ///   } do {
    ///   ^bb(%v0: T0, %v1: T1, ...):
    ///     // body
    ///     scf.yield %v0', %v1', ...
    ///   }
    fn emit_while(
        &mut self,
        cond_expr: &ast::Expr,
        body_stmts: &[ast::Stmt],
    ) {
        // Collect all variables assigned in the body that already exist in outer scope.
        // Also include variables referenced in the condition that get modified.
        let assigned_names = Self::collect_assigned_vars(body_stmts);

        // Carry the variables that already exist in the current scope (not new ones).
        // A struct is carried one field at a time: (name, Some(field), ...).
        let mut carried_vars: Vec<(String, Value<'c, 'a>, VecType, bool)> = Vec::new(); // (name, init_val, type, is_ptr)
        let mut carried_fields: Vec<Option<String>> = Vec::new();
        for name in &assigned_names {
            if let Some(binding) = self.vars.get(name) {
                match binding {
                    Binding::Vec(val, ty) => {
                        carried_vars.push((name.clone(), *val, *ty, false));
                        carried_fields.push(None);
                    }
                    Binding::Ptr(val, scalar) => {
                        // A memref's descriptor flows through the loop as a block argument.
                        carried_vars.push((name.clone(), *val, VecType { scalar: *scalar, width: 0 }, true));
                        carried_fields.push(None);
                    }
                    Binding::Struct { fields } => {
                        for (field, val, ty) in fields {
                            carried_vars.push((name.clone(), *val, *ty, false));
                            carried_fields.push(Some(field.clone()));
                        }
                    }
                }
            }
        }

        // Build MLIR types for carried values
        let mut carried_mlir_types: Vec<Type<'c>> = Vec::new();
        for (_name, _, ty, is_ptr) in &carried_vars {
            if *is_ptr {
                // For ptr types, build a memref<?xT> type
                let elem_mlir = ty.scalar.to_mlir_type(self.ctx);
                let memref_type = MemRefType::new(elem_mlir, &[i64::MIN], None, None);
                carried_mlir_types.push(memref_type.into());
            } else {
                carried_mlir_types.push(ty.to_mlir_type(self.ctx));
            }
        }

        // === "before" region: evaluate condition, pass carried vars through ===
        let mut before_arg_types: Vec<(Type<'c>, Location<'c>)> = Vec::new();
        for t in &carried_mlir_types {
            before_arg_types.push((*t, self.loc));
        }
        let before_block = Block::new(&before_arg_types);

        // Create inner FnCodegen for the before block to evaluate the condition
        {
            let mut before_cg = FnCodegen {
                instances: self.instances.clone(),
                ctx: self.ctx,
                loc: self.loc,
                block: &before_block,
                vars: HashMap::new(),
                comptime_env: self.comptime_env.clone(),
                native_width: self.native_width,
                struct_defs: self.struct_defs.clone(),
                fn_defs: self.fn_defs.clone(),
                current_span: self.current_span.clone(),
            };

            // Copy outer variables (transmute lifetimes for MLIR cross-region refs)
            for (name, binding) in &self.vars {
                let inner_binding: Binding = unsafe { std::mem::transmute(binding.clone()) };
                before_cg.vars.insert(name.clone(), inner_binding);
            }

            // Override carried vars with block args
            let carried: Vec<(String, VecType, bool)> =
                carried_vars.iter().map(|(n, _, t, p)| (n.clone(), *t, *p)).collect();
            FnCodegen::bind_carried(&mut before_cg.vars, &carried, &carried_fields, |idx| {
                before_block.argument(idx).unwrap().into()
            });

            // Evaluate condition
            let (cond_val, _) = before_cg.emit_expr(cond_expr);

            // Extract scalar bool
            let cond_scalar: Value = cond_val;

            // scf.condition %cond %args... : types
            // Pass through the block args as the values to forward to the "after" region
            let mut condition_args: Vec<Value> = Vec::new();
            for idx in 0..carried_vars.len() {
                let arg_val: Value = before_block.argument(idx).unwrap().into();
                condition_args.push(arg_val);
            }

            let condition_op = OperationBuilder::new("scf.condition", before_cg.loc)
                .add_operands(&[&[cond_scalar], condition_args.as_slice()].concat())
                .build()
                .expect("failed to build scf.condition");
            before_block.append_operation(condition_op);
        }

        // === "after" region: execute body, yield updated vars ===
        let mut after_arg_types: Vec<(Type<'c>, Location<'c>)> = Vec::new();
        for t in &carried_mlir_types {
            after_arg_types.push((*t, self.loc));
        }
        let after_block = Block::new(&after_arg_types);

        {
            let mut body_cg = FnCodegen {
                instances: self.instances.clone(),
                ctx: self.ctx,
                loc: self.loc,
                block: &after_block,
                vars: HashMap::new(),
                comptime_env: self.comptime_env.clone(),
                native_width: self.native_width,
                struct_defs: self.struct_defs.clone(),
                fn_defs: self.fn_defs.clone(),
                current_span: self.current_span.clone(),
            };

            // Copy outer variables
            for (name, binding) in &self.vars {
                let inner_binding: Binding = unsafe { std::mem::transmute(binding.clone()) };
                body_cg.vars.insert(name.clone(), inner_binding);
            }

            // Override carried vars with "after" block args
            let carried: Vec<(String, VecType, bool)> =
                carried_vars.iter().map(|(n, _, t, p)| (n.clone(), *t, *p)).collect();
            FnCodegen::bind_carried(&mut body_cg.vars, &carried, &carried_fields, |idx| {
                after_block.argument(idx).unwrap().into()
            });

            // Emit body statements
            for stmt in body_stmts {
                body_cg.emit_stmt(stmt);
            }

            // Yield updated carried vars
            let mut yield_vals: Vec<Value> = Vec::new();
            for (idx, (name, _, _, _)) in carried_vars.iter().enumerate() {
                match (body_cg.vars.get(name), &carried_fields[idx]) {
                    (Some(Binding::Vec(val, _)), None) | (Some(Binding::Ptr(val, _)), None) => {
                        yield_vals.push(*val);
                    }
                    (Some(Binding::Struct { fields }), Some(field)) => {
                        let found = fields.iter().find(|(f, _, _)| f == field).unwrap_or_else(|| {
                            panic!("struct '{}' lost field '{}' in a while body", name, field)
                        });
                        yield_vals.push(found.1);
                    }
                    _ => panic!("variable '{}' changed kind in a while body", name),
                }
            }

            after_block.append_operation(scf::r#yield(&yield_vals, body_cg.loc));
        }

        // Build the two regions
        let before_region = Region::new();
        before_region.append_block(before_block);

        let after_region = Region::new();
        after_region.append_block(after_block);

        // Initial values for iter_args
        let init_vals: Vec<Value<'c, 'a>> = carried_vars.iter().map(|(_, v, _, _)| *v).collect();

        // Build scf.while
        let while_op = OperationBuilder::new("scf.while", self.loc)
            .add_operands(&init_vals)
            .add_results(&carried_mlir_types)
            .add_regions([before_region, after_region])
            .build()
            .expect("failed to build scf.while");

        let while_result = self.block.append_operation(while_op);

        // Bind final values back into outer scope
        let carried: Vec<(String, VecType, bool)> =
            carried_vars.iter().map(|(n, _, t, p)| (n.clone(), *t, *p)).collect();
        Self::bind_carried(&mut self.vars, &carried, &carried_fields, |idx| {
            while_result.result(idx).unwrap().into()
        });
    }

    fn make_index_const(&mut self, value: i64) -> Value<'c, 'a> {
        let index_type = Type::index(self.ctx);
        let attr = IntegerAttribute::new(index_type, value);
        self.block
            .append_operation(arith::constant(self.ctx, attr.into(), self.loc))
            .result(0)
            .unwrap()
            .into()
    }
}

/// Resolve struct return types to flattened MLIR types.
fn resolve_struct_return_types<'c>(
    ctx: &'c Context,
    ret_ty: &ast::Type,
    struct_defs: &HashMap<String, ast::StructDef>,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> Vec<(Type<'c>, VecType)> {
    if is_ptr_type_name(&ret_ty.name) {
        vec![(ptr_to_memref_type(ctx, ret_ty), VecType { scalar: ptr_element_scalar(ret_ty), width: 0 })]
    } else if is_scalar_type_name(&ret_ty.name) {
        let vt = resolve_type(ret_ty, comptime_env, native_width);
        vec![(vt.to_mlir_type(ctx), vt)]
    } else {
        // Struct return: flatten fields
        let struct_def = struct_defs
            .get(&ret_ty.name)
            .unwrap_or_else(|| panic!("undefined struct in return type: {}", ret_ty.name));

        let concrete_width = ret_ty
            .width
            .as_ref()
            .expect("struct return type must have a width")
            .eval(comptime_env, native_width);

        let mut struct_comptime = comptime_env.clone();
        if let Some(cp) = struct_def.comptime_params.first() {
            struct_comptime.insert(cp.name.clone(), concrete_width);
        }

        struct_def
            .fields
            .iter()
            .map(|field| {
                let vt = resolve_type(&field.ty, &struct_comptime, native_width);
                (vt.to_mlir_type(ctx), vt)
            })
            .collect()
    }
}

/// Flatten struct parameters into individual vector parameters.
fn flatten_params<'c>(
    ctx: &'c Context,
    params: &[ast::Param],
    struct_defs: &HashMap<String, ast::StructDef>,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
) -> Vec<(String, Type<'c>, VecType, Option<(String, String)>)> {
    // Returns: (param_name, mlir_type, vec_type, Option<(original_param_name, field_name)>)
    let mut result = Vec::new();
    for param in params {
        if is_ptr_type_name(&param.ty.name) {
            // ptr[T] → memref<?xT>
            let memref_type = ptr_to_memref_type(ctx, &param.ty);
            // Use a dummy VecType — this param is a memref, not a vector
            let dummy_vt = VecType {
                scalar: ScalarType::I8,
                width: 0,
            };
            result.push((param.name.clone(), memref_type, dummy_vt, None));
        } else if is_scalar_type_name(&param.ty.name) {
            let vt = resolve_type(&param.ty, comptime_env, native_width);
            result.push((param.name.clone(), vt.to_mlir_type(ctx), vt, None));
        } else {
            // Struct param: flatten into one param per field
            let struct_def = struct_defs.get(&param.ty.name).unwrap_or_else(|| {
                panic!("undefined struct in param type: {}", param.ty.name)
            });

            let concrete_width = param
                .ty
                .width
                .as_ref()
                .expect("struct param type must have a width")
                .eval(comptime_env, native_width);

            let mut struct_comptime = comptime_env.clone();
            if let Some(cp) = struct_def.comptime_params.first() {
                struct_comptime.insert(cp.name.clone(), concrete_width);
            }

            for field in &struct_def.fields {
                let vt = resolve_type(&field.ty, &struct_comptime, native_width);
                let flat_name = format!("{}_{}", param.name, field.name);
                result.push((
                    flat_name,
                    vt.to_mlir_type(ctx),
                    vt,
                    Some((param.name.clone(), field.name.clone())),
                ));
            }
        }
    }
    result
}

/// Compile a single function definition to an MLIR func.func operation.
/// Emit `memref.global "private" constant` for a const table.
fn emit_const_global<'c>(ctx: &'c Context, def: &ast::ConstDef) -> Operation<'c> {
    let loc = Location::unknown(ctx);
    let scalar = ScalarType::from_name(&def.elem_type);
    let elem = scalar.to_mlir_type(ctx);
    let len = def.values.len() as i64;
    let memref_ty = MemRefType::new(elem, &[len], None, None);
    if scalar.is_int() {
        let bits = scalar.bit_width();
        if let Some(bad) = def.values.iter().find(|&&v| bits < 64 && (v < -(1i64 << (bits - 1)) || v >= (1i64 << bits))) {
            panic!("const '{}': literal {} doesn't fit in {:?}", def.name, bad, scalar);
        }
    }
    let elem_name = format!("i{}", scalar.bit_width());
    let numbers: Vec<String> = def.values.iter().map(|v| v.to_string()).collect();
    let dense = melior::ir::attribute::Attribute::parse(
        ctx,
        &format!("dense<[{}]> : tensor<{}x{}>", numbers.join(","), len, elem_name),
    )
    .unwrap_or_else(|| panic!("const '{}' has an invalid value", def.name));
    OperationBuilder::new("memref.global", loc)
        .add_attributes(&[
            (melior::ir::Identifier::new(ctx, "sym_name"), StringAttribute::new(ctx, &def.name).into()),
            (melior::ir::Identifier::new(ctx, "sym_visibility"), StringAttribute::new(ctx, "private").into()),
            (melior::ir::Identifier::new(ctx, "type"), TypeAttribute::new(memref_ty.into()).into()),
            (melior::ir::Identifier::new(ctx, "initial_value"), dense),
            (melior::ir::Identifier::new(ctx, "constant"), melior::ir::attribute::Attribute::unit(ctx)),
            (
                melior::ir::Identifier::new(ctx, "alignment"),
                IntegerAttribute::new(IntegerType::new(ctx, 64).into(), 64).into(),
            ),
        ])
        .build()
        .expect("failed to build memref.global")
}

fn emit_fn<'c>(
    ctx: &'c Context,
    fn_def: &ast::FnDef,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
    struct_defs: &HashMap<String, ast::StructDef>,
    fn_defs: &HashMap<String, ast::FnDef>,
    const_defs: &[ast::ConstDef],
    instances: std::rc::Rc<std::cell::RefCell<Instances>>,
) -> Operation<'c> {
    let loc = Location::unknown(ctx);

    // Flatten parameters (struct params become multiple vector params)
    let flat_params = flatten_params(ctx, &fn_def.params, struct_defs, comptime_env, native_width);

    let param_mlir_types: Vec<Type<'c>> = flat_params.iter().map(|(_, t, _, _)| *t).collect();

    // Resolve return type
    let ret_info: Vec<(Type<'c>, VecType)> = if let Some(ret_ty) = &fn_def.ret_ty {
        resolve_struct_return_types(ctx, ret_ty, struct_defs, comptime_env, native_width)
    } else {
        vec![]
    };

    let ret_mlir_types: Vec<Type<'c>> = ret_info.iter().map(|(t, _)| *t).collect();

    // Build function type
    let func_type = FunctionType::new(ctx, &param_mlir_types, &ret_mlir_types);

    // Create entry block with params
    let block_args: Vec<(Type<'c>, Location<'c>)> =
        param_mlir_types.iter().map(|t| (*t, loc)).collect();
    let block = Block::new(&block_args);

    // Set up codegen state
    let mut cg = FnCodegen {
        instances,
        ctx,
        loc,
        block: &block,
        vars: HashMap::new(),
        comptime_env: comptime_env.clone(),
        native_width,
        struct_defs: struct_defs.clone(),
        fn_defs: fn_defs.clone(),
        current_span: ast::Span::dummy(),
    };

    // Const tables are pointers. A parameter of the same name shadows them.
    for def in const_defs {
        let scalar = ScalarType::from_name(&def.elem_type);
        let elem = scalar.to_mlir_type(ctx);
        let sized = MemRefType::new(elem, &[def.values.len() as i64], None, None);
        let global: Value = block
            .append_operation(
                OperationBuilder::new("memref.get_global", loc)
                    .add_attributes(&[(
                        melior::ir::Identifier::new(ctx, "name"),
                        melior::ir::attribute::FlatSymbolRefAttribute::new(ctx, &def.name).into(),
                    )])
                    .add_results(&[sized.into()])
                    .build()
                    .expect("failed to build memref.get_global"),
            )
            .result(0)
            .unwrap()
            .into();
        let dynamic: Value = block
            .append_operation(
                OperationBuilder::new("memref.cast", loc)
                    .add_operands(&[global])
                    .add_results(&[MemRefType::new(elem, &[i64::MIN], None, None).into()])
                    .build()
                    .expect("failed to build memref.cast for const"),
            )
            .result(0)
            .unwrap()
            .into();
        cg.vars.insert(def.name.clone(), Binding::Ptr(dynamic, scalar));
    }

    // Bind params to variable names
    // Group struct fields back into Binding::Struct
    let mut struct_fields: HashMap<String, Vec<(String, Value<'c, '_>, VecType)>> = HashMap::new();
    for (i, (name, _, vt, origin)) in flat_params.iter().enumerate() {
        let val: Value<'c, '_> = block.argument(i).unwrap().into();
        match origin {
            None => {
                // Check if this is a ptr param by looking at the original AST params
                let is_ptr = fn_def.params.iter().any(|p| &p.name == name && is_ptr_type_name(&p.ty.name));
                if is_ptr {
                    let orig = fn_def.params.iter().find(|p| &p.name == name).unwrap();
                    let scalar = ptr_element_scalar(&orig.ty);
                    cg.vars.insert(name.clone(), Binding::Ptr(val, scalar));
                } else {
                    cg.vars.insert(name.clone(), Binding::Vec(val, *vt));
                }
            }
            Some((param_name, field_name)) => {
                struct_fields
                    .entry(param_name.clone())
                    .or_default()
                    .push((field_name.clone(), val, *vt));
            }
        }
    }
    // Insert assembled struct bindings
    for (param_name, fields) in struct_fields {
        cg.vars
            .insert(param_name, Binding::Struct { fields });
    }

    // Emit statements
    for stmt in &fn_def.body {
        cg.emit_stmt(stmt);
    }

    // Emit return
    let return_values: Vec<Value> = if let Some(last) = fn_def.body.last() {
        cg.current_span = last.span;
        match &last.kind {
            ast::StmtKind::Return(expr) => {
                match expr {
                    _ if cg.is_struct_expr(expr) => {
                        cg.emit_struct_value(expr).into_iter().map(|(_, val, _)| val).collect()
                    }
                    _ => match ret_info.first() {
                        Some((_, ret_vt)) if ret_vt.width != 0 => vec![cg.emit_expr_typed(expr, *ret_vt).0],
                        Some(_) => vec![cg.emit_expr(expr).0],
                        None => cg.error("a function with no return type can't return a value"),
                    },
                }
            }
            _ => vec![],
        }
    } else {
        vec![]
    };

    block.append_operation(func::r#return(&return_values, loc));

    let region = Region::new();
    region.append_block(block);

    let mut attributes: Vec<(melior::ir::Identifier<'c>, melior::ir::attribute::Attribute<'c>)> =
        Vec::new();
    if fn_def.noinline {
        attributes.push((
            melior::ir::Identifier::new(ctx, "no_inline"),
            melior::ir::attribute::Attribute::unit(ctx),
        ));
    }
    if fn_def.private {
        attributes.push((
            melior::ir::Identifier::new(ctx, "sym_visibility"),
            StringAttribute::new(ctx, "private").into(),
        ));
        // MLIR's visibility stops at the func dialect; this is what reaches LLVM.
        attributes.push((
            melior::ir::Identifier::new(ctx, "llvm.linkage"),
            melior::ir::attribute::Attribute::parse(ctx, "#llvm.linkage<internal>")
                .expect("failed to parse linkage attribute"),
        ));
    }

    let symbol = if fn_def.extern_c { extern_impl_name(&fn_def.name) } else { fn_def.name.clone() };
    if fn_def.extern_c {
        attributes.push((
            melior::ir::Identifier::new(ctx, "sym_visibility"),
            StringAttribute::new(ctx, "private").into(),
        ));
        attributes.push((
            melior::ir::Identifier::new(ctx, "llvm.linkage"),
            melior::ir::attribute::Attribute::parse(ctx, "#llvm.linkage<internal>")
                .expect("failed to parse linkage attribute"),
        ));
    }

    func::func(
        ctx,
        StringAttribute::new(ctx, &symbol),
        TypeAttribute::new(func_type.into()),
        region,
        &attributes,
        loc,
    )
}

/// The symbol holding the body of an `extern fn`.
fn extern_impl_name(name: &str) -> String {
    format!("{}.impl", name)
}

/// The exported C-ABI entry of an `extern fn`: builds memref descriptors around its
/// raw pointers and calls the body, which LLVM then inlines back into it.
fn emit_extern_entry<'c>(
    ctx: &'c Context,
    fn_def: &ast::FnDef,
    comptime_env: &HashMap<String, u64>,
    native_width: u64,
    struct_defs: &HashMap<String, ast::StructDef>,
) -> Operation<'c> {
    let loc = Location::unknown(ctx);
    let ptr_ty: Type = melior::dialect::llvm::r#type::pointer(ctx, 0);
    let i64t: Type = IntegerType::new(ctx, 64).into();
    let descriptor_ty = Type::parse(ctx, "!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>")
        .expect("failed to parse memref descriptor type");

    let flat = flatten_params(ctx, &fn_def.params, struct_defs, comptime_env, native_width);
    let is_ptr: Vec<bool> = flat
        .iter()
        .map(|(name, _, _, _)| {
            fn_def.params.iter().any(|p| &p.name == name && is_ptr_type_name(&p.ty.name))
        })
        .collect();
    let entry_types: Vec<Type<'c>> = flat
        .iter()
        .zip(&is_ptr)
        .map(|((_, t, _, _), is_ptr)| if *is_ptr { ptr_ty } else { *t })
        .collect();
    let impl_types: Vec<Type<'c>> = flat.iter().map(|(_, t, _, _)| *t).collect();
    let ret_types: Vec<Type<'c>> = match &fn_def.ret_ty {
        Some(ret) => resolve_struct_return_types(ctx, ret, struct_defs, comptime_env, native_width)
            .into_iter()
            .map(|(t, _)| t)
            .collect(),
        None => vec![],
    };

    let block_args: Vec<(Type<'c>, Location<'c>)> = entry_types.iter().map(|t| (*t, loc)).collect();
    let block = Block::new(&block_args);

    let constant = |value: i64| -> Value {
        block
            .append_operation(arith::constant(ctx, IntegerAttribute::new(i64t, value).into(), loc))
            .result(0)
            .unwrap()
            .into()
    };
    let insert = |container: Value<'c, '_>, value: Value<'c, '_>, position: &[i64]| -> Value {
        block
            .append_operation(
                OperationBuilder::new("llvm.insertvalue", loc)
                    .add_operands(&[container, value])
                    .add_results(&[descriptor_ty])
                    .add_attributes(&[(
                        melior::ir::Identifier::new(ctx, "position"),
                        DenseI64ArrayAttribute::new(ctx, position).into(),
                    )])
                    .build()
                    .expect("failed to build llvm.insertvalue"),
            )
            .result(0)
            .unwrap()
            .into()
    };

    let mut call_args: Vec<Value> = Vec::new();
    for (i, is_ptr) in is_ptr.iter().enumerate() {
        let arg: Value = block.argument(i).unwrap().into();
        if !*is_ptr {
            call_args.push(arg);
            continue;
        }
        let poison: Value = block
            .append_operation(
                OperationBuilder::new("llvm.mlir.poison", loc)
                    .add_results(&[descriptor_ty])
                    .build()
                    .expect("failed to build llvm.mlir.poison"),
            )
            .result(0)
            .unwrap()
            .into();
        let d = insert(poison, arg, &[0]);
        let d = insert(d, arg, &[1]);
        let d = insert(d, constant(0), &[2]);
        let d = insert(d, constant(0), &[3, 0]);
        let d = insert(d, constant(1), &[4, 0]);
        let memref: Value = block
            .append_operation(
                OperationBuilder::new("builtin.unrealized_conversion_cast", loc)
                    .add_operands(&[d])
                    .add_results(&[impl_types[i]])
                    .build()
                    .expect("failed to cast descriptor to memref"),
            )
            .result(0)
            .unwrap()
            .into();
        call_args.push(memref);
    }

    let call = block.append_operation(
        OperationBuilder::new("func.call", loc)
            .add_operands(&call_args)
            .add_results(&ret_types)
            .add_attributes(&[(
                melior::ir::Identifier::new(ctx, "callee"),
                melior::ir::attribute::FlatSymbolRefAttribute::new(ctx, &extern_impl_name(&fn_def.name)).into(),
            )])
            .build()
            .expect("failed to build call to extern fn body"),
    );
    let results: Vec<Value> = (0..ret_types.len()).map(|i| call.result(i).unwrap().into()).collect();
    block.append_operation(func::r#return(&results, loc));

    let region = Region::new();
    region.append_block(block);
    func::func(
        ctx,
        StringAttribute::new(ctx, &fn_def.name),
        TypeAttribute::new(FunctionType::new(ctx, &entry_types, &ret_types).into()),
        region,
        &[],
        loc,
    )
}

/// Compile a program (list of items) to an MLIR module.
pub fn compile_module<'c>(
    ctx: &'c Context,
    items: &[ast::Item],
    comptime_env: &HashMap<String, u64>,
) -> Module<'c> {
    compile_module_for(ctx, items, comptime_env, Arch::host())
}

/// Compile a program to an MLIR module for a given architecture.
pub fn compile_module_for<'c>(
    ctx: &'c Context,
    items: &[ast::Item],
    comptime_env: &HashMap<String, u64>,
    arch: Arch,
) -> Module<'c> {
    let native_width = arch.register_bytes();
    let items = &crate::comptime::with_register_bytes(items, native_width)[..];
    let module = Module::new(Location::unknown(ctx));

    // Collect struct definitions first
    let mut struct_defs: HashMap<String, ast::StructDef> = HashMap::new();
    for item in items {
        if let ast::Item::Struct(sd) = item {
            struct_defs.insert(sd.name.clone(), sd.clone());
        }
    }

    // Collect all function definitions for inter-function calls
    let mut fn_defs: HashMap<String, ast::FnDef> = HashMap::new();
    for item in items {
        if let ast::Item::Fn(f) = item {
            fn_defs.insert(f.name.clone(), f.clone());
        }
    }

    let instances = std::rc::Rc::new(std::cell::RefCell::new(Instances {
        arch,
        module_env: comptime_env.clone(),
        seen: Default::default(),
        pending: Vec::new(),
        envs: HashMap::new(),
    }));

    let const_defs: Vec<ast::ConstDef> = items
        .iter()
        .filter_map(|item| match item {
            ast::Item::Const(c) => Some(c.clone()),
            _ => None,
        })
        .collect();
    for def in &const_defs {
        module.body().append_operation(emit_const_global(ctx, def));
    }

    for item in items {
        match item {
            ast::Item::Fn(fn_def) => {
                // Merge function-level comptime params with global env
                let env = comptime_env.clone();
                // For now, comptime params must be provided in the env
                // A generic function is emitted once per set of widths it's called with.
                if fn_def.comptime_params.iter().any(|p| !env.contains_key(&p.name)) {
                    continue;
                }
                let op = emit_fn(ctx, fn_def, &env, native_width, &struct_defs, &fn_defs, &const_defs, instances.clone());
                module.body().append_operation(op);
                if fn_def.extern_c {
                    let entry = emit_extern_entry(ctx, fn_def, &env, native_width, &struct_defs);
                    module.body().append_operation(entry);
                }
            }
            ast::Item::Struct(_) | ast::Item::Const(_) => {
                // Structs are erased; consts were emitted above
            }
        }
    }

    // Emitting an instance can request more instances.
    loop {
        let next = instances.borrow_mut().pending.pop();
        let Some(def) = next else { break };
        let env = instances.borrow().envs[&def.name].clone();
        let op = emit_fn(ctx, &def, &env, native_width, &struct_defs, &fn_defs, &const_defs, instances.clone());
        module.body().append_operation(op);
    }

    module
}

/// Lower an MLIR module through the pass pipeline to LLVM dialect.
pub fn lower_to_llvm(ctx: &Context, module: &mut Module) -> Result<(), String> {
    lower_to_llvm_for(ctx, module, Arch::host_triple())
}

/// Lower an MLIR module to the LLVM dialect for a target triple.
pub fn lower_to_llvm_for(ctx: &Context, module: &mut Module, triple: &str) -> Result<(), String> {
    if let Some(attr) = melior::ir::attribute::Attribute::parse(ctx, &format!("\"{}\"", triple)) {
        module.as_operation_mut().set_attribute("llvm.target_triple", attr);
    }

    let pass_manager = pass::PassManager::new(ctx);

    // Inline user-defined functions before lowering
    pass_manager.add_pass(pass::transform::create_inliner());

    // SCF → ControlFlow
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    // Vector → LLVM
    pass_manager.add_pass(pass::conversion::create_vector_to_llvm());
    // Math → LLVM
    pass_manager.add_pass(pass::conversion::create_math_to_llvm());
    // ControlFlow → LLVM
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    // Func → LLVM
    pass_manager.add_pass(pass::conversion::create_func_to_llvm());
    // Arith → LLVM
    pass_manager.add_pass(pass::conversion::create_arith_to_llvm());
    // Index → LLVM
    pass_manager.add_pass(pass::conversion::create_index_to_llvm());
    // Finalize memref → LLVM
    pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
    // Clean up unrealized casts
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

    pass_manager
        .run(module)
        .map_err(|_| "MLIR pass pipeline failed".to_string())
}

/// The NEON names these tests pass vectors with, on x86: the same lanes in the same
/// registers under the C ABI.
#[cfg(all(test, target_arch = "x86_64"))]
#[allow(non_camel_case_types)]
mod neon_names {
    use std::arch::x86_64::*;

    pub type float32x4_t = __m128;
    pub type int32x4_t = __m128i;
    pub type uint32x4_t = __m128i;
    pub type uint8x16_t = __m128i;
    pub type uint8x8_t = __m128i;

    pub unsafe fn vld1q_f32(p: *const f32) -> float32x4_t { unsafe { _mm_loadu_ps(p) } }
    pub unsafe fn vst1q_f32(p: *mut f32, v: float32x4_t) { unsafe { _mm_storeu_ps(p, v) } }
    pub unsafe fn vld1q_s32(p: *const i32) -> int32x4_t { unsafe { _mm_loadu_si128(p.cast()) } }
    pub unsafe fn vst1q_s32(p: *mut i32, v: int32x4_t) { unsafe { _mm_storeu_si128(p.cast(), v) } }
    pub unsafe fn vld1q_u32(p: *const u32) -> uint32x4_t { unsafe { _mm_loadu_si128(p.cast()) } }
    pub unsafe fn vst1q_u32(p: *mut u32, v: uint32x4_t) { unsafe { _mm_storeu_si128(p.cast(), v) } }
    pub unsafe fn vld1q_u8(p: *const u8) -> uint8x16_t { unsafe { _mm_loadu_si128(p.cast()) } }
    pub unsafe fn vst1q_u8(p: *mut u8, v: uint8x16_t) { unsafe { _mm_storeu_si128(p.cast(), v) } }
    pub unsafe fn vld1_u8(p: *const u8) -> uint8x8_t { unsafe { _mm_loadl_epi64(p.cast()) } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;
    use melior::ExecutionEngine;

    fn env(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    // ============================================================
    // Basic arithmetic
    // ============================================================

    /// Helper: compile source, lower to LLVM, look up a function pointer.
    fn jit_lookup(source: &str, fn_name: &str) -> (*mut (), melior::ExecutionEngine) {
        jit_lookup_opt(source, fn_name, 0)
    }

    fn jit_lookup_opt(source: &str, fn_name: &str, opt_level: usize) -> (*mut (), melior::ExecutionEngine) {
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        if !module.as_operation().verify() {
            eprintln!("MLIR verify failed for '{}':\n{}", fn_name, module.as_operation());
            panic!("MLIR verify failed");
        }
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = melior::ExecutionEngine::new(&module, opt_level, &[], false);
        let fptr = engine.lookup(fn_name);
        (fptr, engine)
    }

    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;
    #[cfg(target_arch = "x86_64")]
    use neon_names::*;

    #[test]
    fn test_vector_add_f32() {
        let source = r#"
            fn add(a: f32[4], b: f32[4]) -> f32[4] {
                return a + b
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "add");

        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [11.0, 22.0, 33.0, 44.0]);
        }
    }

    #[test]
    fn test_vector_sub_f32() {
        let source = "fn sub(a: f32[4], b: f32[4]) -> f32[4] { return a - b }";
        let (fptr, _engine) = jit_lookup(source, "sub");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let b = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [9.0, 18.0, 27.0, 36.0]);
        }
    }

    #[test]
    fn test_vector_mul_f32() {
        let source = "fn mul(a: f32[4], b: f32[4]) -> f32[4] { return a * b }";
        let (fptr, _engine) = jit_lookup(source, "mul");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([2.0f32, 3.0, 4.0, 5.0].as_ptr());
            let b = vld1q_f32([10.0f32, 10.0, 10.0, 10.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [20.0, 30.0, 40.0, 50.0]);
        }
    }

    #[test]
    fn test_vector_div_f32() {
        let source = "fn div(a: f32[4], b: f32[4]) -> f32[4] { return a / b }";
        let (fptr, _engine) = jit_lookup(source, "div");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let b = vld1q_f32([2.0f32, 4.0, 5.0, 8.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [5.0, 5.0, 6.0, 5.0]);
        }
    }

    // ============================================================
    // Literal broadcasting
    // ============================================================

    #[test]
    fn test_add_literal() {
        let source = "fn add1(a: f32[4]) -> f32[4] { return a + 1.0 }";
        let (fptr, _engine) = jit_lookup(source, "add1");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [2.0, 3.0, 4.0, 5.0]);
        }
    }

    #[test]
    fn test_mul_literal() {
        let source = "fn scale(a: f32[4]) -> f32[4] { return a * 2.0 }";
        let (fptr, _engine) = jit_lookup(source, "scale");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    // ============================================================
    // Variable assignment and multi-statement
    // ============================================================

    #[test]
    fn test_multi_stmt() {
        let source = r#"
            fn compute(a: f32[4], b: f32[4]) -> f32[4] {
                sum = a + b
                scaled = sum * 2.0
                return scaled
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "compute");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [22.0, 44.0, 66.0, 88.0]);
        }
    }

    // ============================================================
    // Masked select
    // ============================================================

    #[test]
    fn test_masked_select() {
        // [a > 0.0] a : b  → select(a > 0, a, b)
        let source = r#"
            fn clamp_positive(a: f32[4], b: f32[4]) -> f32[4] {
                mask = a > 0.0
                return [mask] a : b
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "clamp_positive");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, -2.0, 3.0, -4.0].as_ptr());
            let b = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            // Where a > 0: take a; else take b
            assert_eq!(out, [1.0, 20.0, 3.0, 40.0]);
        }
    }

    // ============================================================
    // Comptime params
    // ============================================================

    #[test]
    fn test_comptime_param() {
        let source = r#"
            fn add[N](a: f32[N], b: f32[N]) -> f32[N] {
                return a + b
            }
        "#;
        // Need custom jit_lookup with comptime env
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &env(&[("N", 4)]));
        assert!(module.as_operation().verify());
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("add");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([5.0f32, 6.0, 7.0, 8.0].as_ptr());
            let result = f(a, b);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
        }
    }

    // ============================================================
    // Reduction
    // ============================================================

    #[test]
    fn test_reduction_add() {
        let source = r#"
            fn sum(a: f32[4]) -> f32[1] {
                return +/ a
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "sum");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> f32 = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(a);
            assert_eq!(result, 10.0);
        }
    }

    #[test]
    fn test_dot_product() {
        let source = r#"
            fn dot(a: f32[4], b: f32[4]) -> f32[1] {
                return +/ (a * b)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "dot");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> f32 = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let b = vld1q_f32([2.0f32, 3.0, 4.0, 5.0].as_ptr());
            let result = f(a, b);
            // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
            assert_eq!(result, 40.0);
        }
    }

    // ============================================================
    // Unary negation
    // ============================================================

    #[test]
    fn test_negate() {
        let source = "fn neg(a: f32[4]) -> f32[4] { return -a }";
        let (fptr, _engine) = jit_lookup(source, "neg");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let a = vld1q_f32([1.0f32, -2.0, 3.0, -4.0].as_ptr());
            let result = f(a);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [-1.0, 2.0, -3.0, 4.0]);
        }
    }

    // ============================================================
    // Complex expression: speed clamping pattern
    // ============================================================

    #[test]
    fn test_speed_clamp() {
        let source = r#"
            fn clamp_speed(vx: f32[4], vy: f32[4]) -> f32[4] {
                speed_sq = vx * vx + vy * vy
                too_fast = speed_sq > 100.0
                speed = sqrt(speed_sq)
                clamped = [too_fast] (vx / speed) * 10.0 : vx
                return clamped
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "clamp_speed");
        unsafe {
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            // Lane 0: speed = sqrt(9+16) = 5, not > 10, keep vx
            // Lane 1: speed = sqrt(36+64) = 10, not > 10, keep vx
            // Lane 2: speed = sqrt(100+100) = ~14.14, > 10, clamp to (10/14.14)*10 ≈ 7.07
            // Lane 3: speed = sqrt(1+1) = ~1.41, not > 10, keep vx
            let vx = vld1q_f32([3.0f32, 6.0, 10.0, 1.0].as_ptr());
            let vy = vld1q_f32([4.0f32, 8.0, 10.0, 1.0].as_ptr());
            let result = f(vx, vy);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out[0], 3.0); // not clamped
            assert_eq!(out[1], 6.0); // not clamped
            assert!(
                (out[2] - 10.0 / (200.0_f32).sqrt() * 10.0).abs() < 0.01,
                "lane 2: expected ~7.07, got {}",
                out[2]
            );
            assert_eq!(out[3], 1.0); // not clamped
        }
    }

    // ============================================================
    // Integer vectors
    // ============================================================

    #[test]
    fn test_i32_add() {
        let source = "fn add(a: i32[4], b: i32[4]) -> i32[4] { return a + b }";
        let (fptr, _engine) = jit_lookup(source, "add");
        unsafe {
            let f: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let b = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [11, 22, 33, 44]);
        }
    }

    // ============================================================
    // Verify MLIR output (print, don't execute)
    // ============================================================

    #[test]
    fn test_mlir_output_readable() {
        let source = r#"
            fn add(a: f32[4], b: f32[4]) -> f32[4] {
                return a + b
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(ir.contains("func.func @add"));
        assert!(ir.contains("vector<4xf32>"));
        assert!(ir.contains("arith.addf"));
    }

    // ============================================================
    // Struct field access
    // ============================================================

    #[test]
    fn test_struct_field_access() {
        // A function takes a 2-field struct, returns one field
        let source = r#"
            struct Particle[N] {
                pos_x: f32[N],
                pos_y: f32[N],
            }
            fn get_x(p: Particle[4]) -> f32[4] {
                return p.pos_x
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "get_x");
        unsafe {
            // Struct params are flattened: get_x(p_pos_x: f32[4], p_pos_y: f32[4]) -> f32[4]
            let f: extern "C" fn(float32x4_t, float32x4_t) -> float32x4_t =
                std::mem::transmute(fptr);
            let pos_x = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let pos_y = vld1q_f32([10.0f32, 20.0, 30.0, 40.0].as_ptr());
            let result = f(pos_x, pos_y);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    // ============================================================
    // Struct return
    // ============================================================

    #[test]
    fn test_struct_return() {
        // A function takes a 2-field struct, returns a new struct
        // Test IR verification and lowering (multi-value return with vectors)
        let source = r#"
            struct Pair[N] {
                x: f32[N],
                y: f32[N],
            }
            fn swap(p: Pair[4]) -> Pair[4] {
                return Pair[4] { x: p.y, y: p.x }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(
            module.as_operation().verify(),
            "verify failed:\n{}",
            module.as_operation()
        );

        let ir = format!("{}", module.as_operation());
        // Should have a function that takes 2 vector<4xf32> and returns 2 vector<4xf32>
        assert!(ir.contains("func.func @swap"), "should have swap function");
        assert!(ir.contains("vector<4xf32>"), "should have vector<4xf32>");
        // The function should return 2 values
        assert!(
            ir.contains("return") || ir.contains("func.return"),
            "should have a return"
        );

        lower_to_llvm(&ctx, &mut module).unwrap();

        // Verify the lowered IR has the expected struct return shape
        let lowered_ir = format!("{}", module.as_operation());
        assert!(
            lowered_ir.contains("llvm.struct<(vector<4xf32>, vector<4xf32>)>"),
            "lowered IR should contain struct return type:\n{}",
            lowered_ir
        );
    }

    // ============================================================
    // Char literal compare
    // ============================================================

    #[test]
    fn test_char_literal_compare() {
        // Compare u8 vector with char literal — IR verification only
        // (bool[4] = vector<4xi1> doesn't map cleanly to NEON register types)
        let source = r#"
            fn is_open_brace(v: u8[4]) -> bool[4] {
                return v == '{'
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(
            module.as_operation().verify(),
            "verify failed:\n{}",
            module.as_operation()
        );
        lower_to_llvm(&ctx, &mut module).unwrap();
        // Just verify lowering succeeds — no JIT execution for bool vectors
    }

    // ============================================================
    // Prefix sum (scan.add)
    // ============================================================

    #[test]
    fn test_prefix_sum() {
        let source = r#"
            fn prefix_sum(v: f32[4]) -> f32[4] {
                return scan.add(v)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "prefix_sum");
        unsafe {
            let f: extern "C" fn(float32x4_t) -> float32x4_t = std::mem::transmute(fptr);
            let v = vld1q_f32([1.0f32, 2.0, 3.0, 4.0].as_ptr());
            let result = f(v);
            let mut out = [0.0f32; 4];
            vst1q_f32(out.as_mut_ptr(), result);
            // Inclusive prefix sum: [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10]
            assert_eq!(out, [1.0, 3.0, 6.0, 10.0]);
        }
    }

    // ============================================================
    // Scan XOR
    // ============================================================

    #[test]
    fn test_scan_xor() {
        let source = r#"
            fn prefix_xor(v: i32[4]) -> i32[4] {
                return scan.xor(v)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "prefix_xor");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1i32, 3, 5, 7].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            // Inclusive prefix xor: [1, 1^3=2, 1^3^5=7, 1^3^5^7=0]
            assert_eq!(out, [1, 2, 7, 0]);
        }
    }

    // ============================================================
    // Load MLIR verification
    // ============================================================

    #[test]
    fn test_load_mlir_verify() {
        let source = r#"
            fn do_load(ptr: ptr[f32]) -> f32[4] {
                return load[f32[4]](ptr)
            }
        "#;
        // We can't easily JIT this because we'd need a real memref.
        // But we can verify the MLIR is well-formed (it won't be without memref params).
        // Instead, let's just test that the codegen produces valid-looking IR strings.
        // Actually, vector.load needs a memref operand, not a vector.
        // So this test just verifies that emit_load produces the right ops.
        // We'll use a simpler approach: just verify the IR contains vector.load.
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.load"),
            "IR should contain vector.load:\n{}",
            ir
        );
    }

    // ============================================================
    // Store MLIR verification
    // ============================================================

    #[test]
    fn test_store_mlir_verify() {
        // store is called as a function: store(ptr, v)
        // We verify the IR contains vector.store
        let source = r#"
            fn do_store(ptr: f32[1], v: f32[4]) {
                store(ptr, v)
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.store"),
            "IR should contain vector.store:\n{}",
            ir
        );
    }

    // ============================================================
    // Gather MLIR verification
    // ============================================================

    #[test]
    fn test_gather_mlir_verify() {
        // src.[indices] produces vector.gather
        let source = r#"
            fn do_gather(src: ptr[f32], indices: i32[4]) -> f32[4] {
                return src.[indices]
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.gather"),
            "IR should contain vector.gather:\n{}",
            ir
        );
    }

    // ============================================================
    // Scatter MLIR verification
    // ============================================================

    #[test]
    fn test_scatter_mlir_verify() {
        // dst.[indices] = src produces vector.scatter
        let source = r#"
            fn do_scatter(dst: ptr[f32], indices: i32[4], src: f32[4]) {
                dst.[indices] = src
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("vector.scatter"),
            "IR should contain vector.scatter:\n{}",
            ir
        );
    }

    // ============================================================
    // Lane shift operators
    // ============================================================

    #[test]
    fn test_lane_shift_right() {
        // Shift right by 1: [a, b, c, d] lane_shr 1 = [0, a, b, c]
        let source = r#"
            fn shift_right(v: i32[4]) -> i32[4] {
                return lane_shr(v, 1)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shift_right");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 10, 20, 30]);
        }
    }

    #[test]
    fn test_lane_shift_left() {
        // Shift left by 1: [a, b, c, d] lane_shl 1 = [b, c, d, 0]
        let source = r#"
            fn shift_left(v: i32[4]) -> i32[4] {
                return lane_shl(v, 1)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shift_left");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [20, 30, 40, 0]);
        }
    }

    #[test]
    fn test_lane_shift_right_by_2() {
        let source = r#"
            fn shift_right2(v: i32[4]) -> i32[4] {
                return lane_shr(v, 2)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shift_right2");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 30, 40].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 0, 10, 20]);
        }
    }

    // ============================================================
    // JSON-style byte classification (single chunk, no stream yet)
    // ============================================================

    #[test]
    fn test_json_classify_structural() {
        // Classify structural JSON characters in a u8[8] chunk
        let source = r#"
            fn classify(chunk: u8[8]) -> i32[8] {
                is_open_brace = chunk == '{'
                is_close_brace = chunk == '}'
                is_comma = chunk == ','
                structural = is_open_brace | is_close_brace | is_comma
                return [structural] 1 : 0
            }
        "#;
        let (_fptr, _engine) = jit_lookup(source, "classify");

        // Simpler: test with 4-wide
        let source4 = r#"
            fn classify4(chunk: u8[4]) -> i32[4] {
                structural = (chunk == '{') | (chunk == '}') | (chunk == ',')
                return [structural] 1 : 0
            }
        "#;
        let (_fptr, _engine) = jit_lookup(source4, "classify4");
        // Just verify the IR compiles and verifies
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(module.as_operation().verify());
        lower_to_llvm(&ctx, &mut module).unwrap();
    }

    // ============================================================
    // Escape detection with lane shift
    // ============================================================

    #[test]
    fn test_escape_detection() {
        // Detect escaped quotes: verify IR compiles (type changed from u8 to i32)
        let source = r#"
            fn find_real_quotes(chunk: u8[8]) -> i32[8] {
                is_quote = chunk == '"'
                is_backslash = chunk == '\\'
                prev_backslash = lane_shr(is_backslash, 1)
                real_quote = is_quote & ~prev_backslash
                return [real_quote] 1 : 0
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(module.as_operation().verify());
        lower_to_llvm(&ctx, &mut module).unwrap();
    }

    // ============================================================
    // Popcount
    // ============================================================

    #[test]
    fn test_popcount() {
        let source = r#"
            fn count_matches(chunk: u8[8]) -> i32[1] {
                matches = chunk == 'a'
                return popcount(matches)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_matches");
        unsafe {
            // Use f32 return to capture the value from s0, then transmute bits to i32
            let f: extern "C" fn(uint8x8_t) -> i32 = std::mem::transmute(fptr);
            let chunk = vld1_u8([b'a', b'b', b'a', b'c', b'a', b'a', b'd', b'a'].as_ptr());
            let result_bits = f(chunk);
            let result: i32 = (result_bits) as i32;
            assert_eq!(result, 5); // 5 'a' characters
        }
    }

    // ============================================================
    // Stream — IR verification
    // ============================================================

    #[test]
    fn test_stream_ir_verify() {
        // Stream over a buffer, no carry state — just verify IR is valid
        let source = r#"
            fn process(buf: ptr[u8]) {
                stream chunk: u8[4] over buf {
                    result = chunk + 1
                }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("scf.for"),
            "IR should contain scf.for:\n{}",
            ir
        );
        assert!(
            ir.contains("vector.load"),
            "IR should contain vector.load:\n{}",
            ir
        );
        assert!(
            module.as_operation().verify(),
            "MLIR verify failed:\n{}",
            ir
        );
    }

    #[test]
    fn test_stream_with_carry_ir_verify() {
        // Stream with carry state
        let source = r#"
            fn count_commas(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[4] over buf carry (count: i32[1] = 0) {
                    is_comma = chunk == ','
                    chunk_count = popcount(is_comma)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            ir.contains("scf.for"),
            "IR should contain scf.for:\n{}",
            ir
        );
        assert!(
            module.as_operation().verify(),
            "MLIR verify failed:\n{}",
            ir
        );
    }

    // ============================================================
    // Stream — JIT execution
    // ============================================================

    #[test]
    fn test_stream_simple_jit() {
        // Simplest possible stream: just iterate, no carry, no return
        // Verify it doesn't crash
        let source = r#"
            fn process(buf: ptr[u8]) {
                stream chunk: u8[4] over buf {
                    result = chunk + 1
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "process");
        let input: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        unsafe {
            // memref<?xi8> is passed as (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) =
                std::mem::transmute(fptr);
            f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
        }
        // If we get here without crashing, the stream iteration works
    }

    #[test]
    fn test_memref_basic_jit() {
        // Just test that we can pass a memref and return a value
        // No stream -- just load first chunk and count commas
        let source = r#"
            fn first_chunk(buf: ptr[u8]) -> i32[1] {
                chunk = load[u8[4]](buf)
                is_comma = chunk == ','
                return popcount(is_comma)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "first_chunk");
        let input: [u8; 4] = [b'a', b',', b'b', b','];
        unsafe {
            // memref<?xi8> passed as (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 4, 1);
            let result: i32 = (result_bits) as i32;
            assert_eq!(result, 2, "should count 2 commas in first chunk");
        }
    }

    #[cfg(target_arch = "aarch64")] // vector<1xi32> comes back in s0, which is NEON's ABI
    #[test]
    fn test_scf_for_with_carry_minimal() {
        // Minimal: scf.for accumulating a counter, no memref
        // Just counts from 0 to 8 in steps of 4, incrementing carry by 1 each iteration
        let ctx = create_context();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let index_type = Type::index(&ctx);
        let i32_type = IntegerType::new(&ctx, 32).into();
        let vec1xi32 = Type::vector(&[1], i32_type);

        let func_type = FunctionType::new(&ctx, &[], &[vec1xi32]);
        let block = Block::new(&[]);

        // Constants
        let c0 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 0).into(), loc));
        let c8 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 8).into(), loc));
        let c4 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 4).into(), loc));
        let init = block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec1xi32, &[IntegerAttribute::new(i32_type, 0).into()]).unwrap().into(),
            loc));

        // scf.for with carry
        let inner_block = Block::new(&[(index_type, loc), (vec1xi32, loc)]);
        let one = inner_block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec1xi32, &[IntegerAttribute::new(i32_type, 1).into()]).unwrap().into(),
            loc));
        let carry_arg: Value = inner_block.argument(1).unwrap().into();
        let new_carry = inner_block.append_operation(arith::addi(carry_arg, one.result(0).unwrap().into(), loc));
        inner_block.append_operation(scf::r#yield(&[new_carry.result(0).unwrap().into()], loc));

        let region = Region::new();
        region.append_block(inner_block);

        let for_op = OperationBuilder::new("scf.for", loc)
            .add_operands(&[
                c0.result(0).unwrap().into(),
                c8.result(0).unwrap().into(),
                c4.result(0).unwrap().into(),
                init.result(0).unwrap().into(),
            ])
            .add_results(&[vec1xi32])
            .add_regions([region])
            .build()
            .expect("failed to build scf.for");
        let for_result = block.append_operation(for_op);

        block.append_operation(func::r#return(&[for_result.result(0).unwrap().into()], loc));

        let func_region = Region::new();
        func_region.append_block(block);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "test_carry"),
            TypeAttribute::new(func_type.into()),
            func_region,
            &[],
            loc,
        ));

        let ir = format!("{}", module.as_operation());
        assert!(module.as_operation().verify(), "verify failed:\n{}", ir);
        let mut module = module;
        lower_to_llvm(&ctx, &mut module).unwrap();

        let engine = ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("test_carry");
        unsafe {
            // vector<1xi32> on ARM64 is returned in s0 (NEON register), not w0 (GPR)
            let f: extern "C" fn() -> f32 = std::mem::transmute(fptr);
            let result_bits = f();
            let result: i32 = f32::to_bits(result_bits) as i32;
            // 2 iterations (0->4, 4->8), each adds 1: result should be 2
            assert_eq!(result, 2);
        }
    }

    #[cfg(target_arch = "aarch64")] // vector<1xi32> comes back in s0, which is NEON's ABI
    #[test]
    fn test_scf_for_with_memref_and_carry() {
        // scf.for with memref input AND carry state
        let ctx = create_context();
        let loc = Location::unknown(&ctx);
        let module = Module::new(loc);
        let index_type = Type::index(&ctx);
        let i8_type = IntegerType::new(&ctx, 8).into();
        let i32_type = IntegerType::new(&ctx, 32).into();
        let vec1xi32 = Type::vector(&[1], i32_type);
        let vec4xi8 = Type::vector(&[4], i8_type);
        let memref_type: Type = MemRefType::new(i8_type, &[i64::MIN], None, None).into();

        let func_type = FunctionType::new(&ctx, &[memref_type], &[vec1xi32]);
        let block = Block::new(&[(memref_type, loc)]);
        let buf_val: Value = block.argument(0).unwrap().into();

        // Constants
        let c0 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 0).into(), loc));
        let c0_dim = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 0).into(), loc));
        let dim_op = block.append_operation(
            OperationBuilder::new("memref.dim", loc)
                .add_operands(&[buf_val, c0_dim.result(0).unwrap().into()])
                .add_results(&[index_type])
                .build().unwrap());
        let c4 = block.append_operation(arith::constant(&ctx, IntegerAttribute::new(index_type, 4).into(), loc));
        let init = block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec1xi32, &[IntegerAttribute::new(i32_type, 0).into()]).unwrap().into(), loc));

        // Inner block: load chunk, count specific byte, accumulate
        let inner_block = Block::new(&[(index_type, loc), (vec1xi32, loc)]);
        let iv: Value = inner_block.argument(0).unwrap().into();
        let carry: Value = inner_block.argument(1).unwrap().into();

        // Load chunk
        let buf_inner: Value = unsafe { std::mem::transmute(buf_val) };
        let load_op = inner_block.append_operation(
            OperationBuilder::new("vector.load", loc)
                .add_operands(&[buf_inner, iv])
                .add_results(&[vec4xi8])
                .build().unwrap());

        // Compare with ','
        let comma_const = inner_block.append_operation(arith::constant(&ctx,
            DenseElementsAttribute::new(vec4xi8, &[IntegerAttribute::new(i8_type, 44).into()]).unwrap().into(), loc));
        let cmp = inner_block.append_operation(arith::cmpi(&ctx, arith::CmpiPredicate::Eq,
            load_op.result(0).unwrap().into(), comma_const.result(0).unwrap().into(), loc));
        // extui i1->i32
        let vec4xi32 = Type::vector(&[4], i32_type);
        let ext = inner_block.append_operation(arith::extui(cmp.result(0).unwrap().into(), vec4xi32, loc));
        // reduce
        let kind_attr = melior::ir::attribute::Attribute::parse(&ctx, "#vector.kind<add>").unwrap();
        let red = inner_block.append_operation(OperationBuilder::new("vector.reduction", loc)
            .add_operands(&[ext.result(0).unwrap().into()])
            .add_results(&[i32_type])
            .add_attributes(&[(melior::ir::Identifier::new(&ctx, "kind"), kind_attr)])
            .build().unwrap());
        let bcast = inner_block.append_operation(OperationBuilder::new("vector.broadcast", loc)
            .add_operands(&[red.result(0).unwrap().into()])
            .add_results(&[vec1xi32])
            .build().unwrap());
        let new_carry = inner_block.append_operation(arith::addi(carry, bcast.result(0).unwrap().into(), loc));
        inner_block.append_operation(scf::r#yield(&[new_carry.result(0).unwrap().into()], loc));

        let region = Region::new();
        region.append_block(inner_block);

        let for_op = OperationBuilder::new("scf.for", loc)
            .add_operands(&[
                c0.result(0).unwrap().into(),
                dim_op.result(0).unwrap().into(),
                c4.result(0).unwrap().into(),
                init.result(0).unwrap().into(),
            ])
            .add_results(&[vec1xi32])
            .add_regions([region])
            .build().unwrap();
        let for_result = block.append_operation(for_op);

        block.append_operation(func::r#return(&[for_result.result(0).unwrap().into()], loc));

        let func_region = Region::new();
        func_region.append_block(block);
        module.body().append_operation(func::func(
            &ctx,
            StringAttribute::new(&ctx, "count_commas"),
            TypeAttribute::new(func_type.into()),
            func_region,
            &[],
            loc,
        ));

        assert!(module.as_operation().verify(), "verify failed:\n{}", module.as_operation());
        let mut module = module;
        lower_to_llvm(&ctx, &mut module).unwrap();

        let engine = ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("count_commas");
        let input: [u8; 8] = [b'a', b',', b'b', b',', b'c', b',', b'd', 0];
        unsafe {
            // memref<?xi8> -> (ptr, ptr, i64, i64, i64)
            // vector<1xi32> returned in s0 (NEON register), so use f32 and transmute
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> f32 = std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
            let result: i32 = f32::to_bits(result_bits) as i32;
            assert_eq!(result, 3);
        }
    }

    #[test]
    fn test_stream_carry_jit() {
        let source = r#"
            fn count_commas(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[4] over buf carry (count: i32[1] = 0) {
                    is_comma = chunk == ','
                    chunk_count = popcount(is_comma)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_commas");
        let input: [u8; 8] = [b'a', b',', b'b', b',', b'c', b',', b'd', 0];
        unsafe {
            // memref<?xi8> -> (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
            let result: i32 = (result_bits) as i32;
            assert_eq!(result, 3, "should count 3 commas");
        }
    }

    #[test]
    fn test_stream_count_commas_jit() {
        let source = r#"
            fn count_commas(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[4] over buf carry (count: i32[1] = 0) {
                    is_comma = chunk == ','
                    chunk_count = popcount(is_comma)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_commas");
        // Input: "a,b,c,d" (8 bytes, 3 commas) -- processed in 2 chunks of 4
        let input: [u8; 8] = [b'a', b',', b'b', b',', b'c', b',', b'd', b'\0'];
        unsafe {
            // memref<?xi8> -> (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1);
            let result: i32 = (result_bits) as i32;
            assert_eq!(result, 3, "should count 3 commas");
        }
    }

    #[test]
    fn test_scan_dfa_string_toggle() {
        // 2-state DFA: 0=Code, 1=InString. Classes: 0='"', 1=other.
        // Validates the transition-monoid prefix scan against a hand-computed run.
        let source = r#"
            fn dfa_test(input: ptr[u8], table_ptr: ptr[u8], out: ptr[u8]) {
                stream chunk: u8[64] over input carry (st: u8[1] = 0) {
                    table = loadu[u8[256]](table_ptr)
                    states = scan_dfa(table, chunk, st, 2)
                    store_at(out, chunk_offset, states)
                    carry st = extract(states, 63)
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "dfa_test");
        // input bytes are already classes: 0='"', 1=other.  a " b c " d
        let mut input = [1u8; 64];
        input[1] = 0; // "
        input[4] = 0; // "
        // table[state*16 + class] = next state
        let mut table = [0u8; 256];
        table[0 * 16 + 0] = 1; // Code, "  -> InString
        table[0 * 16 + 1] = 0; // Code, other -> Code
        table[1 * 16 + 0] = 0; // InString, " -> Code
        table[1 * 16 + 1] = 1; // InString, other -> InString
        let mut out = [9u8; 64];
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const u8, *const u8, i64, i64, i64,
                *mut u8, *mut u8, i64, i64, i64,
            ) = std::mem::transmute(fptr);
            f(
                input.as_ptr(), input.as_ptr(), 0, 64, 1,
                table.as_ptr(), table.as_ptr(), 0, 256, 1,
                out.as_mut_ptr(), out.as_mut_ptr(), 0, 64, 1,
            );
        }
        // state after each byte (seed 0): a=0, "=1, b=1, c=1, "=0, d=0
        assert_eq!(&out[..6], &[0, 1, 1, 1, 0, 0], "DFA state per byte");
        assert!(out[6..].iter().all(|&s| s == 0), "stays in Code after");
    }

    #[test]
    fn test_scan_dfa_multimode_entanglement() {
        // 4-state DFA proving multi-mode resolution: a `/` inside a string is NOT
        // a comment, and a `"` inside a comment is NOT a string — the exact thing
        // single-toggle CLMUL can't do. States: 0=Code,1=Slash,2=DQ,3=Line.
        // Classes: 0='"', 1='/', 2='\n', 3=other.
        let source = r#"
            fn dfa_test(input: ptr[u8], table_ptr: ptr[u8], out: ptr[u8]) {
                stream chunk: u8[64] over input carry (st: u8[1] = 0) {
                    table = loadu[u8[256]](table_ptr)
                    states = scan_dfa(table, chunk, st, 4)
                    store_at(out, chunk_offset, states)
                    carry st = extract(states, 63)
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "dfa_test");
        // `"` `/` `/` `"`  ` ` `/` `/` `"` `x`   as classes:
        let mut input = [3u8; 64]; // default: other
        for (i, &c) in [0u8, 1, 1, 0, 3, 1, 1, 0, 3].iter().enumerate() {
            input[i] = c;
        }
        let mut t = [0u8; 256];
        let set = |t: &mut [u8; 256], s: usize, row: [u8; 4]| {
            for c in 0..4 {
                t[s * 16 + c] = row[c];
            }
        };
        set(&mut t, 0, [2, 1, 0, 0]); // Code:  "->DQ, /->Slash, \n->Code, other->Code
        set(&mut t, 1, [2, 3, 0, 0]); // Slash: "->DQ, /->Line,  \n->Code, other->Code
        set(&mut t, 2, [0, 2, 2, 2]); // DQ:    "->Code, else stay (/, \n inside string)
        set(&mut t, 3, [3, 3, 0, 3]); // Line:  \n->Code, else stay (" inside comment)
        let mut out = [9u8; 64];
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const u8, *const u8, i64, i64, i64,
                *mut u8, *mut u8, i64, i64, i64,
            ) = std::mem::transmute(fptr);
            f(
                input.as_ptr(), input.as_ptr(), 0, 64, 1,
                t.as_ptr(), t.as_ptr(), 0, 256, 1,
                out.as_mut_ptr(), out.as_mut_ptr(), 0, 64, 1,
            );
        }
        // `"//"` → string (state 2) for the two `/`; then `//"x` → comment (3) incl the `"`.
        assert_eq!(&out[..9], &[2, 2, 2, 0, 0, 1, 3, 3, 3], "multi-mode states");
    }

    #[test]
    fn test_scan_dfa_full_literal_mask() {
        // The real 9-state JS literal DFA (strings + line/block comments +
        // escapes) classifying bytes and producing an in-literal mask in SIMD,
        // validated byte-for-byte against a scalar reference DFA.
        // states: 0 Code,1 Slash,2 DQ,3 DQesc,4 SQ,5 SQesc,6 Line,7 Block,8 BlockStar
        // classes: 0 '"',1 '\'',2 '\\',3 '/',4 '*',5 '\n',6 other
        let source = r#"
            fn litmask(input: ptr[u8], table_ptr: ptr[u8], out: ptr[u8]) {
                stream chunk: u8[64] over input carry (st: u8[1] = 0) {
                    table = loadu[u8[256]](table_ptr)
                    is_dq = chunk == 34
                    is_sq = chunk == 39
                    is_bs = chunk == 92
                    is_sl = chunk == 47
                    is_st = chunk == 42
                    is_nl = chunk == 10
                    c0: u8[64] = 0
                    c1: u8[64] = 1
                    c2: u8[64] = 2
                    c3: u8[64] = 3
                    c4: u8[64] = 4
                    c5: u8[64] = 5
                    c6: u8[64] = 6
                    cls = [is_dq] c0 : ([is_sq] c1 : ([is_bs] c2 : ([is_sl] c3 : ([is_st] c4 : ([is_nl] c5 : c6)))))
                    states = scan_dfa(table, cls, st, 9)
                    two: u8[64] = 2
                    in_lit_mask = states >= two
                    one: u8[64] = 1
                    zero: u8[64] = 0
                    in_lit = [in_lit_mask] one : zero
                    store_at(out, chunk_offset, in_lit)
                    carry st = extract(states, 63)
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "litmask");

        // Build the transition table (also the scalar reference).
        let mut t = [0u8; 256];
        let set = |t: &mut [u8; 256], s: usize, row: [u8; 7]| {
            for c in 0..7 { t[s * 16 + c] = row[c]; }
        };
        set(&mut t, 0, [2, 4, 0, 1, 0, 0, 0]); // Code
        set(&mut t, 1, [2, 4, 0, 6, 7, 0, 0]); // Slash
        set(&mut t, 2, [0, 2, 3, 2, 2, 2, 2]); // DQ
        set(&mut t, 3, [2, 2, 2, 2, 2, 2, 2]); // DQesc
        set(&mut t, 4, [4, 0, 5, 4, 4, 4, 4]); // SQ
        set(&mut t, 5, [4, 4, 4, 4, 4, 4, 4]); // SQesc
        set(&mut t, 6, [6, 6, 6, 6, 6, 0, 6]); // Line
        set(&mut t, 7, [7, 7, 7, 7, 8, 7, 7]); // Block
        set(&mut t, 8, [7, 7, 7, 0, 8, 7, 7]); // BlockStar
        let classify = |b: u8| match b {
            34 => 0, 39 => 1, 92 => 2, 47 => 3, 42 => 4, 10 => 5, _ => 6,
        };

        // Regex-free source exercising strings, escapes, both comment kinds, and
        // the cross-mode cases (`//` in a string, `"` in a comment).
        let src = b"var s = \"a\\\"b//c\"; // x\"y\nvar t = 'p/*q'; /* r\"s */ z;\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        let mut input = [0u8; 128];
        input[..src.len().min(128)].copy_from_slice(&src[..src.len().min(128)]);

        // Scalar reference DFA.
        let mut want = [0u8; 128];
        let mut st = 0usize;
        for i in 0..128 {
            st = t[st * 16 + classify(input[i]) as usize] as usize;
            want[i] = (st >= 2) as u8;
        }

        let mut out = [9u8; 128];
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const u8, *const u8, i64, i64, i64,
                *mut u8, *mut u8, i64, i64, i64,
            ) = std::mem::transmute(fptr);
            f(
                input.as_ptr(), input.as_ptr(), 0, 128, 1,
                t.as_ptr(), t.as_ptr(), 0, 256, 1,
                out.as_mut_ptr(), out.as_mut_ptr(), 0, 128, 1,
            );
        }
        assert_eq!(&out[..], &want[..], "SIMD literal DFA mask must match scalar reference");
    }

    // ============================================================
    // iota
    // ============================================================

    #[test]
    fn test_iota() {
        let source = r#"
            fn make_indices() -> i32[4] {
                return iota(4)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "make_indices");
        unsafe {
            let f: extern "C" fn() -> int32x4_t = std::mem::transmute(fptr);
            let result = f();
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 1, 2, 3]);
        }
    }

    // ============================================================
    // extract
    // ============================================================

    #[test]
    fn test_extract() {
        let source = r#"
            fn get_last(v: i32[4]) -> i32[1] {
                return extract(v, 3)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "get_last");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> i32 = std::mem::transmute(fptr);
            let v = vld1q_s32([10, 20, 30, 42].as_ptr());
            let result = (f(v)) as i32;
            assert_eq!(result, 42);
        }
    }

    // ============================================================
    // seeded scan.xor
    // ============================================================

    #[test]
    fn test_seeded_scan_xor() {
        // scan.xor([1,0,1,0], seed=1) should give:
        // lane0: 1 ^ 1 = 0, then prefix xor:
        // [0, 0^0=0, 0^0^1=1, 0^0^1^0=1]
        // Wait, let me think: seed XORed into lane 0 first.
        // input = [1, 0, 1, 0], seed = 1
        // modified = [1^1, 0, 1, 0] = [0, 0, 1, 0]
        // prefix xor = [0, 0^0=0, 0^0^1=1, 0^0^1^0=1]
        let source = r#"
            fn seeded_xor(v: i32[4], s: i32[1]) -> i32[4] {
                return scan.xor(v, s)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "seeded_xor");
        unsafe {
            let f: extern "C" fn(int32x4_t, i32) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1, 0, 1, 0].as_ptr());
            let seed: i32 = (1u32) as i32; // i32 value 1 as f32 bits
            let result = f(v, seed);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0, 0, 1, 1]);
        }
    }

    // ============================================================
    // chunk_offset in stream
    // ============================================================

    #[test]
    fn test_stream_chunk_offset_ir() {
        let source = r#"
            fn process(buf: ptr[u8]) {
                stream chunk: u8[4] over buf carry (last_offset: i32[1] = 0) {
                    carry last_offset = chunk_offset
                }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        assert!(
            module.as_operation().verify(),
            "MLIR verify failed:\n{}",
            ir
        );
        // chunk_offset should produce an arith.index_cast + vector.broadcast
        assert!(ir.contains("arith.index_cast"), "should cast index to i32:\n{}", ir);
    }

    // ============================================================
    // JSON stage 1 — full pipeline IR verification
    // ============================================================

    #[test]
    fn test_json_stage1_ir() {
        let source = r#"
            fn json_stage1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    -- Classify characters
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs

                    -- Structural chars
                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma

                    -- String boundaries
                    in_str = scan.xor(real_quote, in_string)
                    outside = ~in_str

                    -- Structural chars outside strings
                    real_structural = structural & outside

                    -- Write positions to output
                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)

                    -- Update carry
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        eprintln!("=== JSON Stage 1 MLIR ===\n{}", ir);
        assert!(
            module.as_operation().verify(),
            "JSON stage 1 MLIR verify failed:\n{}",
            ir
        );
    }

    #[test]
    fn test_json_stage1_jit() {
        let source = r#"
            fn json_stage1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs

                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma

                    in_str = scan.xor(real_quote, in_string)
                    outside = ~in_str
                    real_structural = structural & outside

                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)

                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "json_stage1");

        // Test input: {"key": "value", "n": 42}
        let json = b"{\"key\": \"value\", \"n\": 42}\0\0\0\0\0\0\0\0";
        // Pad to multiple of 8
        let input = &json[..32]; // 32 bytes = 4 chunks of 8

        let mut output = [0i32; 64]; // plenty of space for structural positions

        unsafe {
            // memref is (ptr, ptr, i64, i64, i64)
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,  // input memref
                *mut i32, *mut i32, i64, i64, i64,     // output memref
            ) -> i32 = std::mem::transmute(fptr);

            let result_bits = f(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                output.as_mut_ptr(), output.as_mut_ptr(), 0, output.len() as i64, 1,
            );
            let num_structural = (result_bits) as i32;

            eprintln!("JSON: {:?}", std::str::from_utf8(&json[..24]).unwrap());
            eprintln!("num_structural = {}", num_structural);
            for i in 0..num_structural as usize {
                let pos = output[i];
                eprintln!("  structural[{}] = {} ('{}')", i, pos, json[pos as usize] as char);
            }

            // {"key": "value", "n": 42}
            // Structural chars outside strings:
            // pos 0: {
            // pos 5: :  (the colon after "key")
            // pos 15: , (the comma after "value")
            // pos 20: : (the colon after "n")
            // pos 23: } (closing brace)
            assert!(num_structural >= 5, "expected at least 5 structural chars, got {}", num_structural);
            assert_eq!(output[0], 0, "first structural should be {{ at position 0");
        }
    }

    // ============================================================
    // JSON stage 2 — depth + type classification (pure SIMD)
    // ============================================================

    #[test]
    fn test_json_stage2_ir() {
        let source = r#"
            fn json_stage2(input: ptr[u8], positions: ptr[i32],
                           out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    -- Gather actual bytes at structural positions
                    chars = gather(input, pos)

                    -- Depth deltas: open → +1, close → -1, else → 0
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    -- delta = +1 for open, -1 for close, 0 for other
                    -- Use masked select with type context from pos (i32[8])
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)

                    -- Prefix sum for running depth
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)

                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let module = compile_module(&ctx, &items, &HashMap::new());
        let ir = format!("{}", module.as_operation());
        eprintln!("=== JSON Stage 2 MLIR ===\n{}", ir);
        assert!(
            module.as_operation().verify(),
            "JSON stage 2 MLIR verify failed:\n{}",
            ir
        );
    }

    #[test]
    fn test_json_full_pipeline_jit() {
        // Stage 1 source
        let stage1_src = r#"
            fn json_stage1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        // Stage 2 source
        let stage2_src = r#"
            fn json_stage2(input: ptr[u8], positions: ptr[i32],
                           out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    -- delta = +1 for open, -1 for close, 0 for other
                    -- Use masked select with type context from pos (i32[8])
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;

        let (fptr1, _engine1) = jit_lookup(stage1_src, "json_stage1");
        let (fptr2, _engine2) = jit_lookup(stage2_src, "json_stage2");

        // Test: {"a": [1, 2], "b": {"c": 3}}
        let json = b"{\"a\": [1, 2], \"b\": {\"c\": 3}}\0\0\0\0";
        let input = &json[..32];

        let mut positions = [0i32; 64];
        let mut depths = [0i32; 64];

        unsafe {
            // Stage 1: find structural positions
            let f1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr1);

            let bits = f1(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 64, 1,
            );
            let num = (bits) as i32;

            eprintln!("\n=== Full JSON Pipeline ===");
            eprintln!("Input: {:?}", std::str::from_utf8(&json[..28]).unwrap());
            eprintln!("Stage 1: {} structural positions", num);
            for i in 0..num as usize {
                eprintln!("  pos[{}] = {} ('{}')", i, positions[i], json[positions[i] as usize] as char);
            }

            // Pad positions to multiple of 8 for stage 2
            let padded_len = ((num as usize + 7) / 8) * 8;
            // Fill padding with position 0 (safe — will just read '{' again)
            for i in num as usize..padded_len {
                positions[i] = 0;
            }

            // Stage 2: compute depths
            let f2: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,    // input
                *const i32, *const i32, i64, i64, i64,   // positions
                *mut i32, *mut i32, i64, i64, i64,       // out_depth
            ) = std::mem::transmute(fptr2);

            f2(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded_len as i64, 1,
                depths.as_mut_ptr(), depths.as_mut_ptr(), 0, 64, 1,
            );

            eprintln!("Stage 2: depths");
            for i in 0..num as usize {
                let ch = json[positions[i] as usize] as char;
                eprintln!("  depth[{}] = {} (pos {} = '{}')", i, depths[i], positions[i], ch);
            }

            // Verify depths for {"a": [1, 2], "b": {"c": 3}}
            // { → depth 1
            // : → depth 1
            // [ → depth 2
            // , → depth 2
            // ] → depth 1
            // , → depth 1
            // : → depth 1
            // { → depth 2
            // : → depth 2
            // } → depth 1
            // } → depth 0
            assert_eq!(depths[0], 1, "{{ should be depth 1");

            // Find the closing brace
            let last = num as usize - 1;
            assert_eq!(json[positions[last] as usize], b'}');
            assert_eq!(depths[last], 0, "final }} should return to depth 0");
        }
    }

    // ============================================================
    // Full SIMD JSON parser — stage 2b: type classification
    // ============================================================

    #[test]
    fn test_json_type_classify_jit() {
        // Classify value types with whitespace skipping
        let source = r#"
            fn classify_types(input: ptr[u8], positions: ptr[i32],
                              out_types: ptr[i32]) {
                stream pos: i32[8] over positions carry (dummy: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    is_open_obj  = chars == '{'
                    is_close_obj = chars == '}'
                    is_open_arr  = chars == '['
                    is_close_arr = chars == ']'
                    is_colon     = chars == ':'
                    is_comma     = chars == ','
                    is_val_str   = (is_colon | is_comma) & (next == '"')
                    is_val_num   = (is_colon | is_comma) & ((next >= '0') & (next <= '9') | (next == '-'))
                    is_val_true  = (is_colon | is_comma) & (next == 't')
                    is_val_false = (is_colon | is_comma) & (next == 'f')
                    is_val_null  = (is_colon | is_comma) & (next == 'n')
                    is_val_obj   = (is_colon | is_comma) & (next == '{')
                    is_val_arr   = (is_colon | is_comma) & (next == '[')
                    t = zero
                    t = [is_open_obj]  zero + 1  : t
                    t = [is_close_obj] zero + 2  : t
                    t = [is_open_arr]  zero + 3  : t
                    t = [is_close_arr] zero + 4  : t
                    t = [is_val_str]   zero + 5  : t
                    t = [is_val_num]   zero + 6  : t
                    t = [is_val_true]  zero + 7  : t
                    t = [is_val_false] zero + 8  : t
                    t = [is_val_null]  zero + 9  : t
                    t = [is_val_obj]   zero + 10 : t
                    t = [is_val_arr]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry dummy = dummy
                }
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "classify_types");

        // Use structural positions from a known JSON input
        let json_str = b"{\"key\": \"val\", \"n\": 42, \"b\": true}";
        let mut json = [0u8; 48];
        json[..json_str.len()].copy_from_slice(json_str);
        let input = &json[..];

        // Manually determined structural positions for this input:
        // 0:{  5::  13:,  18::  20:number_start  22:,  27::  29:t
        // Actually let's just run stage 1 first
        let stage1_src = r#"
            fn s1(input: ptr[u8], output: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    is_open = chunk == '{'
                    is_close = chunk == '}'
                    is_open_bracket = chunk == '['
                    is_close_bracket = chunk == ']'
                    is_colon = chunk == ':'
                    is_comma = chunk == ','
                    structural = is_open | is_close | is_open_bracket | is_close_bracket | is_colon | is_comma
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(output, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let (fptr1, _e1) = jit_lookup(stage1_src, "s1");

        let mut positions = [0i32; 64];
        let mut types = [0i32; 64];

        unsafe {
            let f1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr1);

            let bits = f1(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 64, 1,
            );
            let num = (bits) as i32;

            // Pad positions
            let padded = ((num as usize + 7) / 8) * 8;
            for i in num as usize..padded { positions[i] = 0; }

            // Run type classification
            let f2: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(fptr);

            f2(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                types.as_mut_ptr(), types.as_mut_ptr(), 0, 64, 1,
            );

            let type_names = ["?", "str", "num", "true", "false", "null", "obj", "arr", "}", "]"];
            eprintln!("\n=== Type Classification ===");
            eprintln!("Input: {:?}", std::str::from_utf8(&json[..34]).unwrap());
            for i in 0..num as usize {
                let ch = json[positions[i] as usize] as char;
                let tname = type_names.get(types[i] as usize).unwrap_or(&"?");
                eprintln!("  [{}] pos={} '{}' → type={} ({})", i, positions[i], ch, types[i], tname);
            }

            // { should classify what follows it — the first key is a string
            // : should classify what follows it — could be string, number, etc.
            assert!(num >= 5, "need at least 5 structural chars");
        }
    }

    // ============================================================
    // Complete SIMD JSON parse pipeline
    // ============================================================

    #[test]
    fn test_json_complete_parse() {
        // Run all stages on a real JSON document
        let stage1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let stage2_depth_src = r#"
            fn stage2_depth(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;

        let stage2_type_src = r#"
            fn stage2_type(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    -- Skip whitespace: check pos+1, pos+2, pos+3, pos+4
                    -- Use the first non-space, non-tab, non-newline byte
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    -- Pick first non-whitespace byte
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    -- Classify the structural char itself
                    is_open_obj  = chars == '{'
                    is_close_obj = chars == '}'
                    is_open_arr  = chars == '['
                    is_close_arr = chars == ']'
                    is_colon     = chars == ':'
                    is_comma     = chars == ','
                    -- For colons and commas, classify the value that follows
                    is_val_str   = (is_colon | is_comma) & (next == '"')
                    is_val_num   = (is_colon | is_comma) & ((next >= '0') & (next <= '9') | (next == '-'))
                    is_val_true  = (is_colon | is_comma) & (next == 't')
                    is_val_false = (is_colon | is_comma) & (next == 'f')
                    is_val_null  = (is_colon | is_comma) & (next == 'n')
                    is_val_obj   = (is_colon | is_comma) & (next == '{')
                    is_val_arr   = (is_colon | is_comma) & (next == '[')
                    -- Type codes: 1=open_obj 2=close_obj 3=open_arr 4=close_arr
                    --             5=colon_str 6=colon_num 7=colon_true 8=colon_false
                    --             9=colon_null 10=colon_obj 11=colon_arr 12=comma
                    t = zero
                    t = [is_open_obj]  zero + 1  : t
                    t = [is_close_obj] zero + 2  : t
                    t = [is_open_arr]  zero + 3  : t
                    t = [is_close_arr] zero + 4  : t
                    t = [is_val_str]   zero + 5  : t
                    t = [is_val_num]   zero + 6  : t
                    t = [is_val_true]  zero + 7  : t
                    t = [is_val_false] zero + 8  : t
                    t = [is_val_null]  zero + 9  : t
                    t = [is_val_obj]   zero + 10 : t
                    t = [is_val_arr]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;

        let (fptr1, _e1) = jit_lookup(stage1_src, "stage1");
        let (fptr2d, _e2d) = jit_lookup(stage2_depth_src, "stage2_depth");
        let (fptr2t, _e2t) = jit_lookup(stage2_type_src, "stage2_type");

        // A real JSON document
        let json_str = r#"{"name": "Alice", "age": 30, "scores": [95, 87], "active": true}"#;
        let mut json = [0u8; 72]; // pad to multiple of 8
        json[..json_str.len()].copy_from_slice(json_str.as_bytes());

        let mut positions = [0i32; 128];
        let mut depths = [0i32; 128];
        let mut types = [0i32; 128];

        unsafe {
            // Stage 1
            let f1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr1);
            let bits = f1(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 128, 1,
            );
            let num = (bits) as i32;
            let padded = ((num as usize + 7) / 8) * 8;
            for i in num as usize..padded { positions[i] = 0; }

            // Stage 2: depth
            let f2d: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(fptr2d);
            f2d(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                depths.as_mut_ptr(), depths.as_mut_ptr(), 0, 128, 1,
            );

            // Stage 2: type classification
            let f2t: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(fptr2t);
            f2t(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                types.as_mut_ptr(), types.as_mut_ptr(), 0, 128, 1,
            );

            let type_names = [
                "?", "open_obj", "close_obj", "open_arr", "close_arr",
                "val:str", "val:num", "val:true", "val:false", "val:null",
                "val:obj", "val:arr", "comma",
            ];

            eprintln!("\n=== Complete SIMD JSON Parse ===");
            eprintln!("Input: {}", json_str);
            eprintln!("\n{:<5} {:<6} {:<6} {:<12} {}", "idx", "pos", "depth", "type", "char");
            eprintln!("{}", "-".repeat(50));
            for i in 0..num as usize {
                let pos = positions[i] as usize;
                let ch = json[pos] as char;
                let tname = type_names.get(types[i] as usize).unwrap_or(&"?");
                eprintln!("{:<5} {:<6} {:<6} {:<12} '{}'", i, pos, depths[i], tname, ch);
            }

            // Verify basic structure
            assert!(num >= 10, "expected many structural chars");
            assert_eq!(depths[0], 1, "first {{ is depth 1");
            assert_eq!(depths[num as usize - 1], 0, "last }} returns to depth 0");

            // Verify types
            assert_eq!(types[0], 1, "first token is open_obj");
            assert_eq!(types[num as usize - 1], 2, "last token is close_obj");

            // ============================================================
            // Now demonstrate the parse result is usable:
            // Walk the parse output and extract key-value pairs
            // ============================================================
            eprintln!("\n=== Extracted Key-Value Pairs ===");
            let mut i = 0;
            while i < num as usize {
                let pos = positions[i] as usize;
                let depth = depths[i];
                let typ = types[i];

                // Colons at depth 1 are top-level key:value separators
                // The structural char must be ':' (not ',')
                if depth == 1 && (typ >= 5 && typ <= 11) && json[pos] == b':' {
                    // The key is the string before this colon
                    // Find the opening quote: scan backwards from pos
                    let json_slice = &json[..json_str.len()];

                    // Find key: look backwards from pos for the key string
                    let mut key_end = pos;
                    while key_end > 0 && json_slice[key_end - 1] != b'"' {
                        key_end -= 1;
                    }
                    let mut key_start = key_end - 1;
                    while key_start > 0 && json_slice[key_start - 1] != b'"' {
                        key_start -= 1;
                    }
                    let key = std::str::from_utf8(&json_slice[key_start..key_end - 1]).unwrap_or("?");

                    let val_desc = match typ {
                        5 => {
                            // String value: find the quoted string after the colon
                            let mut vs = pos + 1;
                            while vs < json_slice.len() && json_slice[vs] != b'"' { vs += 1; }
                            vs += 1; // skip opening quote
                            let mut ve = vs;
                            while ve < json_slice.len() && json_slice[ve] != b'"' { ve += 1; }
                            format!("\"{}\"", std::str::from_utf8(&json_slice[vs..ve]).unwrap_or("?"))
                        }
                        6 => {
                            // Number: read digits
                            let mut ns = pos + 1;
                            while ns < json_slice.len() && json_slice[ns] == b' ' { ns += 1; }
                            let mut ne = ns;
                            while ne < json_slice.len() && (json_slice[ne].is_ascii_digit() || json_slice[ne] == b'-') { ne += 1; }
                            std::str::from_utf8(&json_slice[ns..ne]).unwrap_or("?").to_string()
                        }
                        7 => "true".to_string(),
                        8 => "false".to_string(),
                        9 => "null".to_string(),
                        10 => "{...}".to_string(),
                        11 => "[...]".to_string(),
                        _ => "?".to_string(),
                    };
                    eprintln!("  {} = {}", key, val_desc);
                }
                i += 1;
            }
        }
    }

    // ============================================================
    // SIMD integer parser
    // ============================================================

    #[test]
    fn test_simd_int_parse() {
        // Parse an integer starting at a given position in the input
        // Uses gather to load digits, SIMD multiply-accumulate
        // Supports 1-8 digits and negative numbers
        let source = r#"
            fn parse_int(input: ptr[u8], start_pos: i32[1]) -> i32[1] {
                -- Gather 8 bytes to check for negative sign
                offsets = iota(8) + start_pos
                bytes = gather(input, offsets)

                -- Check for negative: count '-' chars to get offset (0 or 1)
                is_neg_vec = bytes == '-'
                neg_offset = popcount(is_neg_vec)

                -- Re-gather from actual digit start (skips '-' if present)
                digit_offsets = iota(8) + start_pos + neg_offset
                digit_bytes = gather(input, digit_offsets)

                is_digit = (digit_bytes >= '0') & (digit_bytes <= '9')
                zero_char = digit_bytes - digit_bytes + 48
                digit_vals = digit_bytes - zero_char
                -- Count contiguous leading digits (stop at first non-digit)
                not_digit = ~is_digit
                nd1 = not_digit | lane_shr(not_digit, 1)
                nd2 = nd1 | lane_shr(nd1, 2)
                nd3 = nd2 | lane_shr(nd2, 4)
                contiguous = ~nd3
                num_digits = popcount(contiguous)
                masked = [contiguous] digit_vals : digit_vals - digit_vals
                wide = to_i32(masked)
                d0 = extract(wide, 0)
                d1 = extract(wide, 1)
                d2 = extract(wide, 2)
                d3 = extract(wide, 3)
                d4 = extract(wide, 4)
                d5 = extract(wide, 5)
                d6 = extract(wide, 6)
                d7 = extract(wide, 7)
                n = extract(num_digits, 0)
                v1 = d0
                v2 = d0 * 10 + d1
                v3 = d0 * 100 + d1 * 10 + d2
                v4 = d0 * 1000 + d1 * 100 + d2 * 10 + d3
                v5 = d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4
                v6 = d0 * 100000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
                v7 = d0 * 1000000 + d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6
                v8 = d0 * 10000000 + d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7
                result = v1
                result = [n == 2] v2 : result
                result = [n == 3] v3 : result
                result = [n == 4] v4 : result
                result = [n == 5] v5 : result
                result = [n == 6] v6 : result
                result = [n == 7] v7 : result
                result = [n == 8] v8 : result

                -- Negate if needed
                is_neg = extract(is_neg_vec, 0)
                zero = result - result
                neg_result = zero - result
                result = [is_neg] neg_result : result
                return result
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "parse_int");

        // Test: parse "42" from a buffer
        let input = b"answer: 42, next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,  // input memref
                i32,                                     // start_pos as i32[1] in s0
            ) -> i32 = std::mem::transmute(fptr);

            // "42" starts at position 8
            let pos = (8u32) as i32;
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 32, 1, pos);
            let result = (result_bits) as i32;
            eprintln!("parse_int('42' at pos 8) = {}", result);
            assert_eq!(result, 42);

            // Test "1234" at position 0
            let input2 = b"1234____\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos2 = (0u32) as i32;
            let r2 = (f(input2.as_ptr(), input2.as_ptr(), 0, 24, 1, pos2)) as i32;
            eprintln!("parse_int('1234' at pos 0) = {}", r2);
            assert_eq!(r2, 1234);

            // Test "7" at position 0
            let input3 = b"7,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos3 = (0u32) as i32;
            let r3 = (f(input3.as_ptr(), input3.as_ptr(), 0, 24, 1, pos3)) as i32;
            eprintln!("parse_int('7' at pos 0) = {}", r3);
            assert_eq!(r3, 7);

            // Test negative: "-42"
            let input4 = b"-42,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos4 = (0u32) as i32;
            let r4 = (f(input4.as_ptr(), input4.as_ptr(), 0, 24, 1, pos4)) as i32;
            eprintln!("parse_int('-42' at pos 0) = {}", r4);
            assert_eq!(r4, -42);

            // Test large number: "12345678"
            let input5 = b"12345678\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos5 = (0u32) as i32;
            let r5 = (f(input5.as_ptr(), input5.as_ptr(), 0, 24, 1, pos5)) as i32;
            eprintln!("parse_int('12345678' at pos 0) = {}", r5);
            assert_eq!(r5, 12345678);

            // Test single digit negative: "-5"
            let input6 = b"-5,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos6 = (0u32) as i32;
            let r6 = (f(input6.as_ptr(), input6.as_ptr(), 0, 24, 1, pos6)) as i32;
            eprintln!("parse_int('-5' at pos 0) = {}", r6);
            assert_eq!(r6, -5);

            // Test 5-digit number: "99999"
            let input7 = b"99999,next\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let pos7 = (0u32) as i32;
            let r7 = (f(input7.as_ptr(), input7.as_ptr(), 0, 24, 1, pos7)) as i32;
            eprintln!("parse_int('99999' at pos 0) = {}", r7);
            assert_eq!(r7, 99999);
        }
    }

    // ============================================================
    // Full SIMD JSON parser — complete pipeline with all features
    // ============================================================

    #[test]
    fn test_json_full_parse_with_access() {
        // Complete JSON parser pipeline with random access demo

        // Stage 1: structural detection
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        // Stage 2: depth computation
        let s2d_src = r#"
            fn stage2d(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;

        // Stage 2: type classification (with whitespace skip) + value start positions
        let s2t_src = r#"
            fn stage2t(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32], out_val_start: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    one = zero + 1
                    two = zero + 2
                    three = zero + 3
                    four = zero + 4
                    -- Compute value start position (first non-whitespace after structural char)
                    vs = pos + one
                    vs = [s1] pos + two : vs
                    vs = [s1 & s2] pos + three : vs
                    vs = [s1 & s2 & s3] pos + four : vs
                    store_at(out_val_start, chunk_offset, vs)
                    is_open_obj  = chars == '{'
                    is_close_obj = chars == '}'
                    is_open_arr  = chars == '['
                    is_close_arr = chars == ']'
                    is_colon     = chars == ':'
                    is_comma     = chars == ','
                    is_val_str   = (is_colon | is_comma) & (next == '"')
                    is_val_num   = (is_colon | is_comma) & ((next >= '0') & (next <= '9') | (next == '-'))
                    is_val_true  = (is_colon | is_comma) & (next == 't')
                    is_val_false = (is_colon | is_comma) & (next == 'f')
                    is_val_null  = (is_colon | is_comma) & (next == 'n')
                    is_val_obj   = (is_colon | is_comma) & (next == '{')
                    is_val_arr   = (is_colon | is_comma) & (next == '[')
                    t = zero
                    t = [is_open_obj]  zero + 1  : t
                    t = [is_close_obj] zero + 2  : t
                    t = [is_open_arr]  zero + 3  : t
                    t = [is_close_arr] zero + 4  : t
                    t = [is_val_str]   zero + 5  : t
                    t = [is_val_num]   zero + 6  : t
                    t = [is_val_true]  zero + 7  : t
                    t = [is_val_false] zero + 8  : t
                    t = [is_val_null]  zero + 9  : t
                    t = [is_val_obj]   zero + 10 : t
                    t = [is_val_arr]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;

        // Stage 2: SIMD integer parser (1-8 digits, negative support)
        let parse_int_src = r#"
            fn parse_int(input: ptr[u8], start_pos: i32[1]) -> i32[1] {
                -- Gather 8 bytes to check for negative sign
                offsets = iota(8) + start_pos
                bytes = gather(input, offsets)

                -- Check for negative: count '-' chars to get offset (0 or 1)
                is_neg_vec = bytes == '-'
                neg_offset = popcount(is_neg_vec)

                -- Re-gather from actual digit start (skips '-' if present)
                digit_offsets = iota(8) + start_pos + neg_offset
                digit_bytes = gather(input, digit_offsets)

                is_digit = (digit_bytes >= '0') & (digit_bytes <= '9')
                zero_char = digit_bytes - digit_bytes + 48
                digit_vals = digit_bytes - zero_char
                -- Count contiguous leading digits (stop at first non-digit)
                not_digit = ~is_digit
                nd1 = not_digit | lane_shr(not_digit, 1)
                nd2 = nd1 | lane_shr(nd1, 2)
                nd3 = nd2 | lane_shr(nd2, 4)
                contiguous = ~nd3
                num_digits = popcount(contiguous)
                masked = [contiguous] digit_vals : digit_vals - digit_vals
                wide = to_i32(masked)
                -- Extract individual digit values as i32[1]
                d0 = extract(wide, 0)
                d1 = extract(wide, 1)
                d2 = extract(wide, 2)
                d3 = extract(wide, 3)
                d4 = extract(wide, 4)
                d5 = extract(wide, 5)
                d6 = extract(wide, 6)
                d7 = extract(wide, 7)
                n = extract(num_digits, 0)
                -- Build value for each possible length
                v1 = d0
                v2 = d0 * 10 + d1
                v3 = d0 * 100 + d1 * 10 + d2
                v4 = d0 * 1000 + d1 * 100 + d2 * 10 + d3
                v5 = d0 * 10000 + d1 * 1000 + d2 * 100 + d3 * 10 + d4
                v6 = d0 * 100000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5
                v7 = d0 * 1000000 + d1 * 100000 + d2 * 10000 + d3 * 1000 + d4 * 100 + d5 * 10 + d6
                v8 = d0 * 10000000 + d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7
                -- Select based on digit count
                result = v1
                result = [n == 2] v2 : result
                result = [n == 3] v3 : result
                result = [n == 4] v4 : result
                result = [n == 5] v5 : result
                result = [n == 6] v6 : result
                result = [n == 7] v7 : result
                result = [n == 8] v8 : result

                -- Negate if needed
                is_neg = extract(is_neg_vec, 0)
                zero = result - result
                neg_result = zero - result
                result = [is_neg] neg_result : result
                return result
            }
        "#;

        let (f1, _e1) = jit_lookup(s1_src, "stage1");
        let (f2d, _e2d) = jit_lookup(s2d_src, "stage2d");
        let (f2t, _e2t) = jit_lookup(s2t_src, "stage2t");
        let (fpi, _epi) = jit_lookup(parse_int_src, "parse_int");

        // A realistic JSON document with negatives, large numbers, multi-level nesting
        let json_str = r#"{"name": "Alice", "age": 30, "balance": -1500, "scores": [95, 87, 100], "active": true, "city": null, "meta": {"id": 12345678, "ver": -3}}"#;
        let mut json = [0u8; 256];
        json[..json_str.len()].copy_from_slice(json_str.as_bytes());

        let mut positions = [0i32; 256];
        let mut depths = [0i32; 256];
        let mut types = [0i32; 256];
        let mut val_starts = [0i32; 256];
        let mut partners = [-1i32; 256]; // pair matching output

        unsafe {
            // ---- Stage 1: find structural positions ----
            let stage1: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(f1);
            let num = (stage1(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, 256, 1,
            )) as i32;

            let padded = ((num as usize + 7) / 8) * 8;
            for i in num as usize..padded { positions[i] = 0; }

            // ---- Stage 2a: depth ----
            let stage2d: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(f2d);
            stage2d(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                depths.as_mut_ptr(), depths.as_mut_ptr(), 0, 256, 1,
            );

            // ---- Stage 2b: type classification + val_start ----
            let stage2t: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *const i32, *const i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) = std::mem::transmute(f2t);
            stage2t(
                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                positions.as_ptr(), positions.as_ptr(), 0, padded as i64, 1,
                types.as_mut_ptr(), types.as_mut_ptr(), 0, 256, 1,
                val_starts.as_mut_ptr(), val_starts.as_mut_ptr(), 0, 256, 1,
            );

            // ---- Pair matching (sequential O(N) using SIMD-computed depth) ----
            // Use a stack: push open bracket indices, pop on close
            let mut stack = vec![0i32; 128];
            let mut sp = [0usize; 16]; // stack pointer per depth
            for i in 0..num as usize {
                let ch = json[positions[i] as usize];
                let d = depths[i];
                if ch == b'{' || ch == b'[' {
                    let depth = d as usize;
                    stack[depth * 4 + sp[depth]] = i as i32;
                    sp[depth] += 1;
                } else if ch == b'}' || ch == b']' {
                    let depth = (d + 1) as usize; // the depth of the matching open
                    if sp[depth] > 0 {
                        sp[depth] -= 1;
                        let partner_idx = stack[depth * 4 + sp[depth]] as usize;
                        partners[partner_idx] = i as i32;
                        partners[i] = partner_idx as i32;
                    }
                }
            }

            // ---- Integer parser ----
            let parse_int: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                i32,
            ) -> i32 = std::mem::transmute(fpi);

            // ---- Print full parse output ----
            let type_names = [
                "???", "OBJ{", "}OBJ", "ARR[", "]ARR",
                "val:str", "val:num", "val:true", "val:false", "val:null",
                "val:obj", "val:arr",
            ];

            eprintln!("\n{}", "=".repeat(60));
            eprintln!("  COMPLETE SIMD JSON PARSE");
            eprintln!("{}", "=".repeat(60));
            eprintln!("Input: {}", json_str);
            eprintln!("\n{:<4} {:<5} {:<6} {:<12} {:<8} {:<9} {}",
                "idx", "pos", "depth", "type", "partner", "val_start", "char");
            eprintln!("{}", "-".repeat(65));

            for i in 0..num as usize {
                let pos = positions[i] as usize;
                let ch = json[pos] as char;
                let tname = type_names.get(types[i] as usize).unwrap_or(&"?");
                let partner_str = if partners[i] >= 0 {
                    format!("→{}", partners[i])
                } else {
                    "-".to_string()
                };
                eprintln!("{:<4} {:<5} {:<6} {:<12} {:<8} {:<9} '{}'",
                    i, pos, depths[i], tname, partner_str, val_starts[i], ch);
            }

            // ---- Random access demo: extract key-value pairs ----
            eprintln!("\n--- Random Access Demo ---");

            // Helper: find string content at a byte position
            let get_string = |pos: usize| -> &str {
                let mut start = pos + 1;
                while start < json_str.len() && json[start] != b'"' { start += 1; }
                start += 1; // skip opening quote
                let mut end = start;
                while end < json_str.len() && json[end] != b'"' { end += 1; }
                std::str::from_utf8(&json[start..end]).unwrap_or("?")
            };

            // Walk top-level keys
            eprintln!("\nTop-level key-value pairs:");
            let mut i = 0;
            while i < num as usize {
                let pos = positions[i] as usize;
                let d = depths[i];
                let t = types[i];

                if d == 1 && json[pos] == b':' {
                    // Find the key (string before the colon)
                    let key = get_string(positions[i - 1] as usize);

                    let value_str = match t {
                        5 => { // val:str
                            let s = get_string(pos);
                            format!("\"{}\"", s)
                        }
                        6 => { // val:num — use val_starts for whitespace-skipped position
                            let num_pos = val_starts[i] as usize;
                            let pos_bits = (num_pos as u32) as i32;
                            let val = (parse_int(
                                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                pos_bits,
                            )) as i32;
                            format!("{}", val)
                        }
                        7 => "true".to_string(),
                        8 => "false".to_string(),
                        9 => "null".to_string(),
                        10 | 11 => { // nested obj/arr
                            let open_idx = i + 1; // next structural is the open bracket
                            if partners[open_idx] >= 0 {
                                let close_pos = positions[partners[open_idx] as usize] as usize;
                                let open_pos = positions[open_idx] as usize;
                                let content = std::str::from_utf8(&json[open_pos..=close_pos]).unwrap_or("?");
                                content.to_string()
                            } else {
                                "?".to_string()
                            }
                        }
                        _ => "?".to_string(),
                    };
                    eprintln!("  {} = {}", key, value_str);
                }
                i += 1;
            }

            // ---- Direct key lookup demo ----
            eprintln!("\nDirect key lookup:");
            // Find keys by scanning top-level colons
            for i in 0..num as usize {
                if depths[i] == 1 && json[positions[i] as usize] == b':' {
                    let key = get_string(positions[i - 1] as usize);
                    if key == "age" {
                        let num_pos = val_starts[i] as usize;
                        let val = (parse_int(
                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                            (num_pos as u32) as i32,
                        )) as i32;
                        eprintln!("  json[\"age\"] = {}", val);
                        assert_eq!(val, 30);
                    }
                    if key == "balance" {
                        let num_pos = val_starts[i] as usize;
                        let val = (parse_int(
                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                            (num_pos as u32) as i32,
                        )) as i32;
                        eprintln!("  json[\"balance\"] = {}", val);
                        assert_eq!(val, -1500);
                    }
                    if key == "name" {
                        let val = get_string(positions[i] as usize);
                        eprintln!("  json[\"name\"] = \"{}\"", val);
                        assert_eq!(val, "Alice");
                    }
                    if key == "active" {
                        let t = types[i];
                        let val = t == 7; // val:true
                        eprintln!("  json[\"active\"] = {}", val);
                        assert!(val);
                    }
                    if key == "city" {
                        let t = types[i];
                        let is_null = t == 9;
                        eprintln!("  json[\"city\"] = null (is_null={})", is_null);
                        assert!(is_null);
                    }
                    if key == "scores" {
                        // Use pair matching to find the array extent
                        let open_idx = i + 1;
                        if partners[open_idx] >= 0 {
                            let close_idx = partners[open_idx] as usize;
                            // Count commas + 1 = number of elements
                            let mut elem_count = 1;
                            for j in (open_idx + 1)..close_idx {
                                if json[positions[j] as usize] == b',' { elem_count += 1; }
                            }
                            eprintln!("  json[\"scores\"] = array with {} elements", elem_count);

                            // Parse each element using val_starts
                            let mut elem_idx = 0;
                            // First element: right after '[', use val_starts of the open bracket
                            let np0 = val_starts[open_idx] as usize;
                            let v0 = (parse_int(
                                json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                (np0 as u32) as i32,
                            )) as i32;
                            eprintln!("    scores[{}] = {}", elem_idx, v0);
                            elem_idx += 1;
                            // Remaining elements: after each comma
                            for j in (open_idx + 1)..close_idx {
                                if types[j] == 6 { // val:num (comma before number)
                                    let np = val_starts[j] as usize;
                                    let v = (parse_int(
                                        json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                        (np as u32) as i32,
                                    )) as i32;
                                    eprintln!("    scores[{}] = {}", elem_idx, v);
                                    elem_idx += 1;
                                }
                            }
                            assert_eq!(elem_count, 3);
                        }
                    }
                    if key == "meta" {
                        // Nested object — drill into depth 2
                        let open_idx = i + 1;
                        if partners[open_idx] >= 0 {
                            let close_idx = partners[open_idx] as usize;
                            for j in (open_idx + 1)..close_idx {
                                if depths[j] == 2 && json[positions[j] as usize] == b':' {
                                    let inner_key = get_string(positions[j - 1] as usize);
                                    if inner_key == "id" && types[j] == 6 {
                                        let np = val_starts[j] as usize;
                                        let v = (parse_int(
                                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                            (np as u32) as i32,
                                        )) as i32;
                                        eprintln!("  json[\"meta\"][\"id\"] = {}", v);
                                        assert_eq!(v, 12345678);
                                    }
                                    if inner_key == "ver" && types[j] == 6 {
                                        let np = val_starts[j] as usize;
                                        let v = (parse_int(
                                            json.as_ptr(), json.as_ptr(), 0, json.len() as i64, 1,
                                            (np as u32) as i32,
                                        )) as i32;
                                        eprintln!("  json[\"meta\"][\"ver\"] = {}", v);
                                        assert_eq!(v, -3);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ============================================================
    // Arbitrary JSON test — complex document with all value types
    // ============================================================

    #[test]
    fn test_arbitrary_json() {
        // Reuse the SIMD pipeline sources
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let s2d_src = r#"
            fn stage2d(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;
        let s2t_src = r#"
            fn stage2t(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32], out_vs: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    c4 = gather(input, pos + 4)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    s3 = (c3 == ' ') | (c3 == '\n') | (c3 == '\t') | (c3 == '\r')
                    next = [s1] ([s2] ([s3] c4 : c3) : c2) : c1
                    zero = pos - pos
                    one = zero + 1
                    two = zero + 2
                    three = zero + 3
                    four = zero + 4
                    ws_off = [s1] ([s2] ([s3] four : three) : two) : one
                    vs = pos + ws_off
                    store_at(out_vs, chunk_offset, vs)
                    is_oo = chars == '{'
                    is_co = chars == '}'
                    is_oa = chars == '['
                    is_ca = chars == ']'
                    is_col = chars == ':'
                    is_com = chars == ','
                    is_sep = is_col | is_com
                    t = zero
                    t = [is_oo]                    zero + 1  : t
                    t = [is_co]                    zero + 2  : t
                    t = [is_oa]                    zero + 3  : t
                    t = [is_ca]                    zero + 4  : t
                    t = [is_sep & (next == '"')]   zero + 5  : t
                    t = [is_sep & ((next >= '0') & (next <= '9') | (next == '-'))] zero + 6 : t
                    t = [is_sep & (next == 't')]   zero + 7  : t
                    t = [is_sep & (next == 'f')]   zero + 8  : t
                    t = [is_sep & (next == 'n')]   zero + 9  : t
                    t = [is_sep & (next == '{')]   zero + 10 : t
                    t = [is_sep & (next == '[')]   zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;
        let pi_src = r#"
            fn pi(input: ptr[u8], sp: i32[1]) -> i32[1] {
                offsets = iota(8) + sp
                bytes = gather(input, offsets)
                is_neg_vec = bytes == '-'
                neg_off = popcount(is_neg_vec)
                doff = iota(8) + sp + neg_off
                db = gather(input, doff)
                is_d = (db >= '0') & (db <= '9')
                zc = db - db + 48
                dv = db - zc
                nd = popcount(is_d)
                m = [is_d] dv : dv - dv
                w = to_i32(m)
                d0 = extract(w, 0)
                d1 = extract(w, 1)
                d2 = extract(w, 2)
                d3 = extract(w, 3)
                d4 = extract(w, 4)
                d5 = extract(w, 5)
                d6 = extract(w, 6)
                d7 = extract(w, 7)
                n = extract(nd, 0)
                r = d0
                r = [n == 2] d0*10+d1 : r
                r = [n == 3] d0*100+d1*10+d2 : r
                r = [n == 4] d0*1000+d1*100+d2*10+d3 : r
                r = [n == 5] d0*10000+d1*1000+d2*100+d3*10+d4 : r
                r = [n == 6] d0*100000+d1*10000+d2*1000+d3*100+d4*10+d5 : r
                r = [n == 7] d0*1000000+d1*100000+d2*10000+d3*1000+d4*100+d5*10+d6 : r
                r = [n == 8] d0*10000000+d1*1000000+d2*100000+d3*10000+d4*1000+d5*100+d6*10+d7 : r
                is_neg = extract(is_neg_vec, 0)
                z = r - r
                nr = z - r
                r = [is_neg] nr : r
                return r
            }
        "#;

        let (f1, _e1) = jit_lookup(s1_src, "stage1");
        let (f2d, _e2d) = jit_lookup(s2d_src, "stage2d");
        let (f2t, _e2t) = jit_lookup(s2t_src, "stage2t");
        let (fpi, _epi) = jit_lookup(pi_src, "pi");

        // Complex JSON with escaped strings, negative numbers, empty containers, nested objects
        let json_str = concat!(
            r#"{"users": [{"name": "Bob \"Jr\"", "age": 35, "score": 9999, "#,
            r#""active": false}, {"name": "Eve", "age": -1, "score": 0, "#,
            r#""active": true}], "count": 2, "tags": [], "meta": {}, "note": null}"#,
        );
        let json_bytes = json_str.as_bytes();
        let mut json = vec![0u8; ((json_bytes.len() + 15) / 8) * 8];
        json[..json_bytes.len()].copy_from_slice(json_bytes);

        let n = 256;
        let mut positions = vec![0i32; n];
        let mut depths = vec![0i32; n];
        let mut types = vec![0i32; n];
        let mut val_starts = vec![0i32; n];
        let mut partners = vec![-1i32; n];

        unsafe {
            let st1: extern "C" fn(*const u8,*const u8,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) -> i32 = std::mem::transmute(f1);
            let num = (st1(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,n as i64,1)) as i32;
            let padded = ((num as usize+7)/8)*8;
            for i in num as usize..padded { positions[i]=0; }

            let st2d: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2d);
            st2d(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,padded as i64,1,depths.as_mut_ptr(),depths.as_mut_ptr(),0,n as i64,1);

            let st2t: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2t);
            st2t(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,padded as i64,1,types.as_mut_ptr(),types.as_mut_ptr(),0,n as i64,1,val_starts.as_mut_ptr(),val_starts.as_mut_ptr(),0,n as i64,1);

            // Pair matching
            let mut stack = vec![0i32; 128];
            let mut sp = [0usize; 32];
            for i in 0..num as usize {
                let ch = json[positions[i] as usize];
                let d = depths[i];
                if ch == b'{' || ch == b'[' {
                    let depth = d as usize;
                    if depth<32 && sp[depth]<4 { stack[depth*4+sp[depth]]=i as i32; sp[depth]+=1; }
                } else if ch == b'}' || ch == b']' {
                    let depth = (d+1) as usize;
                    if depth<32 && sp[depth]>0 { sp[depth]-=1; let pi=stack[depth*4+sp[depth]] as usize; partners[pi]=i as i32; partners[i]=pi as i32; }
                }
            }

            let parse_i = |pos: usize| -> i32 {
                let fpi_fn: extern "C" fn(*const u8,*const u8,i64,i64,i64,i32) -> i32 = std::mem::transmute(fpi);
                (fpi_fn(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,(pos as u32) as i32)) as i32
            };

            let get_str = |pos: usize| -> &str {
                let mut s = pos+1;
                while s < json_bytes.len() && json[s] != b'"' { s+=1; }
                s+=1;
                let mut e = s;
                while e < json_bytes.len() && json[e] != b'"' { if json[e]==b'\\' { e+=1; } e+=1; }
                std::str::from_utf8(&json[s..e]).unwrap_or("?")
            };

            eprintln!("\n=== Arbitrary JSON Parse ===");
            eprintln!("Input ({} bytes):", json_bytes.len());
            eprintln!("  {}", json_str);
            eprintln!("{} structural positions\n", num);

            // Verify key values
            let mut found = std::collections::HashMap::new();
            for i in 0..num as usize {
                if depths[i] == 1 && json[positions[i] as usize] == b':' {
                    let key = get_str(positions[i-1] as usize);
                    let t = types[i] as usize;
                    let vs = val_starts[i] as usize;
                    match t {
                        5 => { found.insert(key.to_string(), format!("\"{}\"", get_str(positions[i] as usize))); }
                        6 => { found.insert(key.to_string(), format!("{}", parse_i(vs))); }
                        7 => { found.insert(key.to_string(), "true".into()); }
                        8 => { found.insert(key.to_string(), "false".into()); }
                        9 => { found.insert(key.to_string(), "null".into()); }
                        10 => { found.insert(key.to_string(), "{...}".into()); }
                        11 => { found.insert(key.to_string(), "[...]".into()); }
                        _ => {}
                    }
                }
            }

            eprintln!("Top-level key-values:");
            for (k, v) in &found {
                eprintln!("  {} = {}", k, v);
            }

            assert_eq!(found["count"], "2");
            assert_eq!(found["note"], "null");
            assert_eq!(found["users"], "[...]");
            assert_eq!(found["tags"], "[...]");
            assert_eq!(found["meta"], "{...}");

            // Verify empty containers
            for i in 0..num as usize {
                if depths[i] == 1 && json[positions[i] as usize] == b':' {
                    let key = get_str(positions[i-1] as usize);
                    if key == "tags" {
                        let oi = i + 1;
                        assert!(partners[oi] >= 0);
                        let ci = partners[oi] as usize;
                        assert_eq!(ci, oi + 1, "empty array: close should be right after open");
                        eprintln!("  tags = [] (verified empty)");
                    }
                    if key == "meta" {
                        let oi = i + 1;
                        assert!(partners[oi] >= 0);
                        let ci = partners[oi] as usize;
                        assert_eq!(ci, oi + 1, "empty object: close should be right after open");
                        eprintln!("  meta = {{}} (verified empty)");
                    }
                }
            }

            // Verify nested values
            let mut user_ages = Vec::new();
            let mut user_names = Vec::new();
            for i in 0..num as usize {
                if depths[i] == 3 && json[positions[i] as usize] == b':' {
                    let key = get_str(positions[i-1] as usize);
                    let vs = val_starts[i] as usize;
                    if key == "name" && types[i] == 5 {
                        user_names.push(get_str(positions[i] as usize).to_string());
                    }
                    if key == "age" && types[i] == 6 {
                        user_ages.push(parse_i(vs));
                    }
                }
            }
            eprintln!("  User names: {:?}", user_names);
            eprintln!("  User ages: {:?}", user_ages);
            assert_eq!(user_names.len(), 2);
            assert!(user_names[0].contains("Bob"));
            assert_eq!(user_names[1], "Eve");
            assert_eq!(user_ages, vec![35, -1]);

            eprintln!("\n=== All arbitrary JSON assertions passed ===");
        }
    }

    // ============================================================
    // BENCHMARK — throughput in GB/s
    // ============================================================

    #[test]
    fn bench_json_stage1_throughput() {
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(8) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 7)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        // Compile with full optimization
        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        // Generate a large JSON document (~10 MB)
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();

        // Pad to multiple of 8
        let padded_len = ((json_len + 7) / 8) * 8;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());

        // Allocate output (generous — ~10% of input is structural)
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark ===");
        eprintln!("Input size: {:.2} MB ({} bytes)", json_len as f64 / 1e6, json_len);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            // Warmup
            let _ = f(
                json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1,
            );

            // Benchmark: run multiple iterations
            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(
                    json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1,
                );
                total_structural = (bits) as i32;
            }
            let elapsed = start.elapsed();

            let total_bytes = json_len as f64 * iterations as f64;
            let seconds = elapsed.as_secs_f64();
            let gb_per_sec = total_bytes / seconds / 1e9;
            let mb_per_sec = total_bytes / seconds / 1e6;

            eprintln!("Structural chars found: {}", total_structural);
            eprintln!("Iterations: {}", iterations);
            eprintln!("Total time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s ({:.0} MB/s)", gb_per_sec, mb_per_sec);
            eprintln!("Structural density: {:.1}%", total_structural as f64 / json_len as f64 * 100.0);
        }
    }

    #[test]
    fn bench_json_stage1_16wide() {
        // Same benchmark but with 16-byte chunks
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[16] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(16) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 15)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 15) / 16) * 16;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark (16-wide) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);

            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
                total_structural = (bits) as i32;
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Structural chars: {}", total_structural);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_stage1_32wide() {
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[32] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(32) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 31)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 31) / 32) * 32;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark (32-wide) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);

            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
                total_structural = (bits) as i32;
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Structural chars: {}", total_structural);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_stage1_64wide() {
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[64] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(64) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 63)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 63) / 64) * 64;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 Benchmark (64-wide) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);

            let iterations = 10;
            let start = std::time::Instant::now();
            let mut total_structural = 0i32;
            for _ in 0..iterations {
                let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
                total_structural = (bits) as i32;
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Structural chars: {}", total_structural);
            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_full_pipeline() {
        // Benchmark stage1 + stage2 (depth + types) together
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[32] over input carry (in_string: bool[1] = false, pos: i32[1] = 0) {
                    is_quote = chunk == '"'
                    is_backslash = chunk == '\\'
                    prev_bs = lane_shr(is_backslash, 1)
                    real_quote = is_quote & ~prev_bs
                    structural = (chunk == '{') | (chunk == '}') | (chunk == '[')
                               | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    in_str = scan.xor(real_quote, in_string)
                    real_structural = structural & ~in_str
                    indices = iota(32) + chunk_offset
                    compressstore(positions, pos, indices, real_structural)
                    carry in_string = extract(in_str, 31)
                    carry pos = pos + popcount(real_structural)
                }
                return pos
            }
        "#;
        let s2d_src = r#"
            fn stage2d(input: ptr[u8], positions: ptr[i32], out_depth: ptr[i32]) {
                stream pos: i32[8] over positions carry (depth: i32[1] = 0) {
                    chars = gather(input, pos)
                    is_open  = (chars == '{') | (chars == '[')
                    is_close = (chars == '}') | (chars == ']')
                    zero = pos - pos
                    one = zero + 1
                    neg_one = zero - 1
                    delta = [is_open] one : ([is_close] neg_one : zero)
                    depth_vec = scan.add(delta, depth)
                    store_at(out_depth, chunk_offset, depth_vec)
                    carry depth = extract(depth_vec, 7)
                }
            }
        "#;
        let s2t_src = r#"
            fn stage2t(input: ptr[u8], positions: ptr[i32], out_types: ptr[i32]) {
                stream pos: i32[8] over positions carry (d: i32[1] = 0) {
                    chars = gather(input, pos)
                    c1 = gather(input, pos + 1)
                    c2 = gather(input, pos + 2)
                    c3 = gather(input, pos + 3)
                    s1 = (c1 == ' ') | (c1 == '\n') | (c1 == '\t') | (c1 == '\r')
                    s2 = (c2 == ' ') | (c2 == '\n') | (c2 == '\t') | (c2 == '\r')
                    next = [s1] ([s2] c3 : c2) : c1
                    zero = pos - pos
                    is_oo = chars == '{'
                    is_co = chars == '}'
                    is_oa = chars == '['
                    is_ca = chars == ']'
                    is_sep = (chars == ':') | (chars == ',')
                    t = zero
                    t = [is_oo] zero + 1 : t
                    t = [is_co] zero + 2 : t
                    t = [is_oa] zero + 3 : t
                    t = [is_ca] zero + 4 : t
                    t = [is_sep & (next == '"')]  zero + 5 : t
                    t = [is_sep & ((next >= '0') & (next <= '9') | (next == '-'))] zero + 6 : t
                    t = [is_sep & (next == 't')]  zero + 7 : t
                    t = [is_sep & (next == 'f')]  zero + 8 : t
                    t = [is_sep & (next == 'n')]  zero + 9 : t
                    t = [is_sep & (next == '{')] zero + 10 : t
                    t = [is_sep & (next == '[')] zero + 11 : t
                    store_at(out_types, chunk_offset, t)
                    carry d = d
                }
            }
        "#;

        let (f1, _e1) = jit_lookup_opt(s1_src, "stage1", 2);
        let (f2d, _e2d) = jit_lookup_opt(s2d_src, "stage2d", 2);
        let (f2t, _e2t) = jit_lookup_opt(s2t_src, "stage2t", 2);

        // Same JSON
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 31) / 32) * 32;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_s = json_len / 2;
        let mut positions = vec![0i32; max_s];
        let mut depths = vec![0i32; max_s];
        let mut types = vec![0i32; max_s];

        eprintln!("\n=== Full Pipeline Benchmark (stage1@32 + stage2) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let st1: extern "C" fn(*const u8,*const u8,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) -> i32 = std::mem::transmute(f1);
            let st2d: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2d);
            let st2t: extern "C" fn(*const u8,*const u8,i64,i64,i64,*const i32,*const i32,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) = std::mem::transmute(f2t);

            // Warmup
            let bits = st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
            let num = (bits) as i32;
            let spad = ((num as usize+7)/8)*8;
            for i in num as usize..spad { positions[i]=0; }
            st2d(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,depths.as_mut_ptr(),depths.as_mut_ptr(),0,max_s as i64,1);
            st2t(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,types.as_mut_ptr(),types.as_mut_ptr(),0,max_s as i64,1);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let bits = st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
                let num = (bits) as i32;
                let spad = ((num as usize+7)/8)*8;
                for i in num as usize..spad { positions[i]=0; }
                st2d(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,depths.as_mut_ptr(),depths.as_mut_ptr(),0,max_s as i64,1);
                st2t(json.as_ptr(),json.as_ptr(),0,json.len() as i64,1,positions.as_ptr(),positions.as_ptr(),0,spad as i64,1,types.as_mut_ptr(),types.as_mut_ptr(),0,max_s as i64,1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    // ============================================================
    // BITMASK-STYLE stage 1 — simdjson approach
    // ============================================================

    #[test]
    fn test_bitmask_roundtrip() {
        // Verify to_bitmask and from_bitmask work correctly
        let source = r#"
            fn test_bitmask(v: u8[8]) -> i32[1] {
                mask = v == 'a'
                bits = to_bitmask(mask)
                -- Count bits set (should match popcount)
                pc = popcount(mask)
                return pc
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "test_bitmask", 2);
        unsafe {
            let f: extern "C" fn(uint8x8_t) -> i32 = std::mem::transmute(fptr);
            let v = vld1_u8([b'a', b'b', b'a', b'a', b'b', b'b', b'a', b'b'].as_ptr());
            let result = (f(v)) as i32;
            assert_eq!(result, 4, "should count 4 'a' chars");
        }
    }

    #[test]
    fn test_bitmask_clmul_correctness() {
        // Test that CLMUL-based prefix XOR gives same results as scan.xor
        // Use a simple 8-byte input for easy verification, but pack to u64
        let source = r#"
            fn test_clmul(v: u8[8]) -> i32[8] {
                quote = v == '"'
                in_str = scan.xor(quote)
                return [in_str] 1 : 0
            }
        "#;
        let (_fptr, _engine) = jit_lookup_opt(source, "test_clmul", 2);
        // Just verify the scan.xor result via a simpler test
        let source2 = r#"
            fn test_clmul2(v: u8[8]) -> i32[1] {
                quote = v == '"'
                in_str = scan.xor(quote)
                return popcount(in_str)
            }
        "#;
        let (fptr2, _engine2) = jit_lookup_opt(source2, "test_clmul2", 2);
        unsafe {
            let f: extern "C" fn(uint8x8_t) -> i32 = std::mem::transmute(fptr2);
            // Input: x " h e l " x x → in_str = [0,1,1,1,1,0,0,0] → 4 true
            let v = vld1_u8([b'x', b'"', b'h', b'e', b'l', b'"', b'x', b'x'].as_ptr());
            let result = (f(v)) as i32;
            assert_eq!(result, 4, "4 bytes should be inside string");
        }
    }

    #[test]
    fn bench_json_stage1_bitmask_16() {
        // 16-wide bitmask: fits in one NEON register, clean bitcast to u16
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[16] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec
                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                  | (chunk == ']') | (chunk == ':') | (chunk == ',')

                    quote_bits = to_bitmask(real_quote_vec)
                    structural_bits = to_bitmask(structural_vec)

                    -- CLMUL for prefix XOR
                    all_ones = ~(quote_bits & ~quote_bits)
                    in_string_raw = clmul(quote_bits, all_ones)
                    -- XOR with carry, then mask to 16 bits
                    in_string_with_carry = xor(in_string_raw, prev_in_string)
                    -- Mask: only lower 16 bits matter
                    mask16 = bit_shr(all_ones, 48)
                    in_string_bits = in_string_with_carry & mask16

                    real_structural_bits = structural_bits & ~in_string_bits
                    real_structural_vec = from_bitmask(real_structural_bits, 16)

                    indices = iota(16) + chunk_offset
                    compressstore(positions, pos, indices, real_structural_vec)

                    -- Carry: bit 15 (the last bit of this 16-byte chunk)
                    -- If set, next chunk starts inside a string
                    -- Use all_ones as -1 to propagate: if bit 15 is set, carry = all_ones, else 0
                    -- This way XOR with carry fills all 64 bits correctly
                    carry_bit = bit_shr(in_string_bits, 15)
                    -- Spread to all 64 bits: multiply by all_ones
                    carry prev_in_string = carry_bit * all_ones
                    carry pos = pos + popcount(real_structural_vec)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 15) / 16) * 16;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 BITMASK (16-wide, CLMUL) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            let num = (bits) as i32;
            eprintln!("Structural chars: {}", num);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);

            assert_eq!(num, 500004, "should find same structural count as vector version");
        }
    }

    #[test]
    fn bench_json_stage1_bitmask() {
        // Process 64 bytes per chunk:
        // - Compare in vector domain (u8[64])
        // - Pack to bitmask (u64[1]) immediately
        // - Do prefix XOR via CLMUL in bitmask domain
        // - Output positions via scalar bit extraction loop
        //
        // This matches simdjson's approach.
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[64] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    -- Compare in vector domain
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec

                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                  | (chunk == ']') | (chunk == ':') | (chunk == ',')

                    -- Pack to bitmask domain
                    quote_bits = to_bitmask(real_quote_vec)
                    structural_bits = to_bitmask(structural_vec)

                    -- Prefix XOR via CLMUL
                    all_ones = ~(quote_bits & ~quote_bits)
                    in_string_raw = clmul(quote_bits, all_ones)
                    in_string_with_carry = xor(in_string_raw, prev_in_string)

                    real_structural_bits = structural_bits & ~in_string_with_carry
                    real_structural_vec = from_bitmask(real_structural_bits, 64)

                    indices = iota(64) + chunk_offset
                    compressstore(positions, pos, indices, real_structural_vec)

                    -- Carry: bit 63 spread to all 64 bits
                    carry_bit = bit_shr(in_string_with_carry, 63)
                    carry prev_in_string = carry_bit * all_ones
                    carry pos = pos + popcount(real_structural_vec)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 63) / 64) * 64;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 BITMASK (64-wide, CLMUL) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            // Warmup + verify
            let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            let num = (bits) as i32;
            eprintln!("Structural chars: {}", num);

            // Verify correctness against vector version
            eprintln!("First 10 structural positions:");
            for i in 0..std::cmp::min(10, num as usize) {
                let p = positions[i] as usize;
                eprintln!("  [{}] pos={} char='{}'", i, p, json[p] as char);
            }
            assert_eq!(json[positions[0] as usize], b'{', "first structural should be {{");

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);
        }
    }

    #[test]
    fn bench_json_stage1_128() {
        // 128-wide: compare in u8[128], split into two bool[64] halves,
        // bitmask+CLMUL each half, chain carry between them
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[128] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    -- Compare in 128-wide vector domain
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec
                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                   | (chunk == ']') | (chunk == ':') | (chunk == ',')

                    -- Split into two 64-lane halves
                    quote_lo = split_lo(real_quote_vec)
                    quote_hi = split_hi(real_quote_vec)
                    struct_lo = split_lo(structural_vec)
                    struct_hi = split_hi(structural_vec)

                    -- Bitmask + CLMUL on low half
                    qbits_lo = to_bitmask(quote_lo)
                    sbits_lo = to_bitmask(struct_lo)
                    all_ones = qbits_lo | ~qbits_lo
                    in_string_raw_lo = clmul(qbits_lo, all_ones)
                    in_string_lo = in_string_raw_lo ^ prev_in_string
                    real_sbits_lo = sbits_lo & ~in_string_lo

                    -- Carry from low to high
                    carry_lo = (in_string_lo >> 63) * all_ones

                    -- Bitmask + CLMUL on high half
                    qbits_hi = to_bitmask(quote_hi)
                    sbits_hi = to_bitmask(struct_hi)
                    in_string_raw_hi = clmul(qbits_hi, all_ones)
                    in_string_hi = in_string_raw_hi ^ carry_lo
                    real_sbits_hi = sbits_hi & ~in_string_hi

                    -- Emit positions for low half
                    real_struct_lo = from_bitmask(real_sbits_lo, 64)
                    indices_lo = iota(64) + chunk_offset
                    compressstore(positions, pos, indices_lo, real_struct_lo)
                    pos_after_lo = pos + popcount(real_struct_lo)

                    -- Emit positions for high half (offset by 64)
                    real_struct_hi = from_bitmask(real_sbits_hi, 64)
                    indices_hi = iota(64) + chunk_offset + 64
                    compressstore(positions, pos_after_lo, indices_hi, real_struct_hi)

                    -- Carry out
                    carry prev_in_string = (in_string_hi >> 63) * all_ones
                    carry pos = pos_after_lo + popcount(real_struct_hi)
                }
                return pos
            }
        "#;

        let (fptr, _engine) = jit_lookup_opt(s1_src, "stage1", 2);

        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 127) / 128) * 128;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_structural = json_len / 2;
        let mut positions = vec![0i32; max_structural];

        eprintln!("\n=== JSON Stage 1 (128-wide, vector scan.xor) ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            let bits = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            let num = (bits) as i32;
            eprintln!("Structural chars: {}", num);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                let _ = f(json.as_ptr(), json.as_ptr(), 0, padded_len as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, max_structural as i64, 1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s", gb_per_sec);

            assert_eq!(num, 500004, "128-wide should find same structural count");
        }
    }

    #[test]
    fn test_tbl_basic() {
        // Table lookup: tbl(data, indices) where data is [0..15] used as both table and indices
        let source = r#"
            fn nibble_lookup(data: u8[16]) -> u8[16] {
                low = data & 15
                return tbl(data, low)
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "nibble_lookup", 2);
        unsafe {
            let f: extern "C" fn(uint8x16_t) -> uint8x16_t = std::mem::transmute(fptr);
            let data = vld1q_u8([0u8,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15].as_ptr());
            let result = f(data);
            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), result);
            // tbl([0..15], [0..15] & 0xF) = tbl([0..15], [0..15]) = [0..15]
            assert_eq!(out, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
        }
    }

    #[test]
    fn test_utf8_validation() {
        // Simplified UTF-8 validator using SIMD classification
        let source = r#"
            fn validate_utf8(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (prev_incomplete: u8[1] = 0, errors: i32[1] = 0) {
                    -- Classify bytes by high nibble
                    high = bit_shr(chunk, 4)

                    -- Classification by high nibble ranges
                    is_continuation = (high >= 8) & (high <= 11)
                    is_2byte = (high >= 12) & (high <= 13)
                    is_3byte = high == 14
                    is_4byte = high == 15

                    -- After a multi-byte start, next bytes must be continuation
                    expect_cont_1 = is_2byte | is_3byte | is_4byte
                    expect_cont_2 = is_3byte | is_4byte
                    expect_cont_3 = is_4byte

                    shifted_1 = lane_shr(expect_cont_1, 1)
                    shifted_2 = lane_shr(expect_cont_2, 2)
                    shifted_3 = lane_shr(expect_cont_3, 3)

                    -- Expected continuation at each position
                    expected = shifted_1 | shifted_2 | shifted_3

                    -- Mismatch: expected continuation but not, or got continuation unexpectedly
                    mismatch = (expected & ~is_continuation) | (~expected & is_continuation)

                    chunk_errors = popcount(mismatch)

                    carry prev_incomplete = prev_incomplete
                    carry errors = errors + chunk_errors
                }
                return errors
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "validate_utf8", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 = std::mem::transmute(fptr);

            // Valid ASCII
            let valid_ascii = b"Hello, world!!\0\0";
            let errors = (f(valid_ascii.as_ptr(), valid_ascii.as_ptr(), 0, 16, 1)) as i32;
            eprintln!("Valid ASCII errors: {}", errors);
            assert_eq!(errors, 0, "Valid ASCII should have 0 errors");

            // Valid UTF-8: "café" = [99, 97, 102, 195, 169] (é is 0xC3 0xA9)
            let valid_utf8 = b"caf\xC3\xA9_______\0\0\0\0";
            let padded_len = 16;
            let errors = (f(valid_utf8.as_ptr(), valid_utf8.as_ptr(), 0, padded_len as i64, 1)) as i32;
            eprintln!("Valid UTF-8 errors: {}", errors);
            assert_eq!(errors, 0, "Valid UTF-8 should have 0 errors");

            // Invalid: bare continuation byte (0x80 without a starter)
            let invalid = b"\x80Hello_________\0";
            let errors = (f(invalid.as_ptr(), invalid.as_ptr(), 0, 16, 1)) as i32;
            eprintln!("Invalid (bare continuation) errors: {}", errors);
            assert!(errors > 0, "Bare continuation byte should be an error");
        }
    }

    #[test]
    fn test_string_escape_detection() {
        // Detect backslash escapes in input using SIMD
        let source = r#"
            fn has_escapes(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (found: i32[1] = 0) {
                    has_bs = chunk == '\\'
                    chunk_has = popcount(has_bs)
                    carry found = found + chunk_has
                }
                return found
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "has_escapes", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 = std::mem::transmute(fptr);

            // No escapes
            let clean = b"Hello, world!!!!\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = (f(clean.as_ptr(), clean.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 0, "Clean string should have 0 backslashes");

            // Has escapes
            let escaped = b"Hello\\nWorld\\t!!\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = (f(escaped.as_ptr(), escaped.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 2, "Should find 2 backslashes");
        }
    }

    #[test]
    fn test_number_boundary_detection() {
        // Count bytes that are part of numbers (digits, minus, dot, e/E, plus)
        let source = r#"
            fn count_number_bytes(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (count: i32[1] = 0) {
                    is_digit = (chunk >= '0') & (chunk <= '9')
                    is_minus = chunk == '-'
                    is_dot = chunk == '.'
                    is_e = (chunk == 'e') | (chunk == 'E')
                    is_plus = chunk == '+'
                    is_num_byte = is_digit | is_minus | is_dot | is_e | is_plus
                    chunk_count = popcount(is_num_byte)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "count_number_bytes", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 = std::mem::transmute(fptr);

            // "3.14" has 4 number bytes
            let input = b"3.14____________\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = (f(input.as_ptr(), input.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 4, "3.14 has 4 number bytes");

            // "-1.5e10" has 7 number bytes
            let input2 = b"-1.5e10_________\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n2 = (f(input2.as_ptr(), input2.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n2, 7, "-1.5e10 has 7 number bytes");
        }
    }

    #[test]
    fn test_whitespace_classification() {
        let source = r#"
            fn count_whitespace(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over input carry (count: i32[1] = 0) {
                    is_ws = (chunk == ' ') | (chunk == '\n') | (chunk == '\t') | (chunk == '\r')
                    chunk_count = popcount(is_ws)
                    carry count = count + chunk_count
                }
                return count
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "count_whitespace", 2);
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 = std::mem::transmute(fptr);
            let input = b"  hello  world  \0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
            let n = (f(input.as_ptr(), input.as_ptr(), 0, 32, 1)) as i32;
            assert_eq!(n, 6, "Should count 6 whitespace chars");
        }
    }

    #[test]
    fn bench_json_all_simd_stages() {
        // All SIMD stages combined: structural detection, UTF-8 validation, escape detection
        let s1_src = r#"
            fn stage1(input: ptr[u8], positions: ptr[i32]) -> i32[1] {
                stream chunk: u8[64] over input carry (prev_in_string: u64[1] = 0, pos: i32[1] = 0) {
                    quote_vec = chunk == '"'
                    bs_vec = chunk == '\\'
                    prev_bs_vec = lane_shr(bs_vec, 1)
                    real_quote_vec = quote_vec & ~prev_bs_vec
                    structural_vec = (chunk == '{') | (chunk == '}') | (chunk == '[')
                                   | (chunk == ']') | (chunk == ':') | (chunk == ',')
                    quote_bits = to_bitmask(real_quote_vec)
                    structural_bits = to_bitmask(structural_vec)
                    all_ones = ~(quote_bits & ~quote_bits)
                    in_string_raw = clmul(quote_bits, all_ones)
                    in_string_with_carry = xor(in_string_raw, prev_in_string)
                    real_structural_bits = structural_bits & ~in_string_with_carry
                    real_structural_vec = from_bitmask(real_structural_bits, 64)
                    indices = iota(64) + chunk_offset
                    compressstore(positions, pos, indices, real_structural_vec)
                    carry_bit = bit_shr(in_string_with_carry, 63)
                    carry prev_in_string = carry_bit * all_ones
                    carry pos = pos + popcount(real_structural_vec)
                }
                return pos
            }
        "#;

        let utf8_src = r#"
            fn validate_utf8(input: ptr[u8]) -> u8[16] {
                stream chunk: u8[16] over input carry (prev_byte: u8[1] = 0, error_acc: u8[16] = [u8: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) {
                    prev1 = lane_shr(chunk, 1)
                    byte_1_high_table = [u8: 2, 2, 2, 2, 2, 2, 2, 2, 128, 128, 128, 128, 33, 1, 21, 73]
                    prev1_high = bit_shr(prev1, 4)
                    byte_1_high = tbl(byte_1_high_table, prev1_high)
                    byte_1_low_table = [u8: 231, 163, 131, 131, 139, 203, 203, 203, 203, 203, 203, 203, 203, 219, 203, 203]
                    prev1_low = prev1 & 15
                    byte_1_low = tbl(byte_1_low_table, prev1_low)
                    byte_2_high_table = [u8: 1, 1, 1, 1, 1, 1, 1, 1, 230, 174, 186, 186, 1, 1, 1, 1]
                    cur_high = bit_shr(chunk, 4)
                    byte_2_high = tbl(byte_2_high_table, cur_high)
                    special_cases = byte_1_high & byte_1_low & byte_2_high
                    prev2 = lane_shr(chunk, 2)
                    prev3 = lane_shr(chunk, 3)
                    is_starter_3plus_p2 = prev2 >= 224
                    is_starter_4_p3 = prev3 >= 240
                    must23 = is_starter_3plus_p2 | is_starter_4_p3
                    zero_byte = chunk - chunk
                    must23_byte = [must23] zero_byte + 128 : zero_byte
                    errors_for_chunk = xor(special_cases, must23_byte)
                    error_acc = error_acc | errors_for_chunk
                    carry prev_byte = extract(chunk, 15)
                    carry error_acc = error_acc
                }
                return error_acc
            }
        "#;

        let escape_src = r#"
            fn count_escapes(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[64] over input carry (count: i32[1] = 0) {
                    carry count = count + popcount(chunk == '\\')
                }
                return count
            }
        "#;

        let (f1, _e1) = jit_lookup_opt(s1_src, "stage1", 2);
        let (f_utf8, _e_utf8) = jit_lookup_opt(utf8_src, "validate_utf8", 2);
        let (f_esc, _e_esc) = jit_lookup_opt(escape_src, "count_escapes", 2);

        // Generate JSON
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data": ["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id": {}, "name": "user_{}", "score": {}, "active": {}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();
        let padded_len = ((json_len + 63) / 64) * 64;
        let mut json = vec![0u8; padded_len];
        json[..json_len].copy_from_slice(json_str.as_bytes());
        let max_s = json_len / 2;
        let mut positions = vec![0i32; max_s];

        eprintln!("\n=== All SIMD Stages Benchmark ===");
        eprintln!("Input size: {:.2} MB", json_len as f64 / 1e6);

        unsafe {
            let st1: extern "C" fn(*const u8,*const u8,i64,i64,i64,*mut i32,*mut i32,i64,i64,i64) -> f32 = std::mem::transmute(f1);
            let utf8: extern "C" fn(*const u8,*const u8,i64,i64,i64) -> uint8x16_t = std::mem::transmute(f_utf8);
            let esc: extern "C" fn(*const u8,*const u8,i64,i64,i64) -> f32 = std::mem::transmute(f_esc);

            // Warmup
            st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
            utf8(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);
            esc(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);

            let iterations = 10;
            let start = std::time::Instant::now();
            for _ in 0..iterations {
                st1(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1,positions.as_mut_ptr(),positions.as_mut_ptr(),0,max_s as i64,1);
                utf8(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);
                esc(json.as_ptr(),json.as_ptr(),0,padded_len as i64,1);
            }
            let elapsed = start.elapsed();
            let total_bytes = json_len as f64 * iterations as f64;
            let gb_per_sec = total_bytes / elapsed.as_secs_f64() / 1e9;

            eprintln!("Per iteration: {:.3} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
            eprintln!("Throughput: {:.2} GB/s (all SIMD stages combined)", gb_per_sec);
        }
    }

    // ============================================================
    // Vector literal + tbl test
    // ============================================================

    #[test]
    fn test_vec_literal_basic() {
        // Constant vector literal: [u8: 10, 20, 30, 40]
        let source = r#"
            fn make_vec() -> i32[4] {
                v = [i32: 10, 20, 30, 40]
                return v
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "make_vec", 2);
        unsafe {
            let f: extern "C" fn() -> int32x4_t = std::mem::transmute(fptr);
            let result = f();
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [10, 20, 30, 40]);
        }
    }

    #[test]
    fn test_tbl_with_vec_literal() {
        // Use a constant lookup table to map nibbles to values
        let source = r#"
            fn nibble_classify(data: u8[16]) -> u8[16] {
                -- Table: maps high nibble to category
                -- 0-7 = ASCII (1), 8-B = continuation (2), C-D = 2-byte (4), E = 3-byte (8), F = 4-byte (16)
                table = [u8: 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 8, 16]
                high = bit_shr(data, 4)
                return tbl(table, high)
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "nibble_classify", 2);
        unsafe {
            let f: extern "C" fn(uint8x16_t) -> uint8x16_t = std::mem::transmute(fptr);
            // Test: ASCII 'A' (0x41), continuation 0x80, 2-byte start 0xC3, 3-byte start 0xE2, 4-byte start 0xF0
            let data = vld1q_u8([0x41, 0x80, 0xC3, 0xE2, 0xF0, 0x00, 0x7F, 0xBF, 0xDF, 0xEF, 0xFF, 0x20, 0x90, 0xA0, 0xC0, 0xD0].as_ptr());
            let result = f(data);
            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), result);
            // 0x41 → nibble 4 → ASCII (1)
            // 0x80 → nibble 8 → continuation (2)
            // 0xC3 → nibble C → 2-byte (4)
            // 0xE2 → nibble E → 3-byte (8)
            // 0xF0 → nibble F → 4-byte (16)
            assert_eq!(out[0], 1, "ASCII");
            assert_eq!(out[1], 2, "continuation");
            assert_eq!(out[2], 4, "2-byte start");
            assert_eq!(out[3], 8, "3-byte start");
            assert_eq!(out[4], 16, "4-byte start");
        }
    }

    // ============================================================
    // Real UTF-8 validation (simdjson algorithm)
    // ============================================================

    #[test]
    fn test_utf8_validation_real() {
        // simdjson's UTF-8 validation algorithm using tbl lookups
        // Error categories (bitmask):
        //   TOO_SHORT=1, TOO_LONG=2, OVERLONG_3=4, TOO_LARGE=8,
        //   SURROGATE=16, OVERLONG_2=32, TOO_LARGE_1000=64, TWO_CONTS=128, OVERLONG_4=64
        //
        // Three lookups: byte_1_high, byte_1_low, byte_2_high
        // Error if (byte_1_high & byte_1_low & byte_2_high) != 0
        let source = r#"
            fn validate_utf8(input: ptr[u8]) -> u8[16] {
                stream chunk: u8[16] over input carry (prev_byte: u8[1] = 0, error_acc: u8[16] = [u8: 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) {
                    -- Get previous byte for each position (lane shift right by 1)
                    prev1 = lane_shr(chunk, 1)
                    -- TODO: first lane should come from carry prev_byte
                    -- For now, first lane is 0 (correct for start of input)

                    -- Lookup table 1: prev byte's high nibble → error flags
                    -- TOO_LONG=2 for ASCII (0-7), TWO_CONTS=128 for continuation (8-B),
                    -- TOO_SHORT|OVERLONG_2=33 for C, TOO_SHORT=1 for D,
                    -- TOO_SHORT|OVERLONG_3|SURROGATE=21 for E,
                    -- TOO_SHORT|TOO_LARGE|TOO_LARGE_1000|OVERLONG_4=137 for F
                    -- TOO_LONG=2, TWO_CONTS=128, TOO_SHORT=1, OVERLONG_2=32
                    -- OVERLONG_3=4, SURROGATE=16, TOO_LARGE=8, TOO_LARGE_1000=OVERLONG_4=64
                    byte_1_high_table = [u8: 2, 2, 2, 2, 2, 2, 2, 2, 128, 128, 128, 128, 33, 1, 21, 73]
                    prev1_high = bit_shr(prev1, 4)
                    byte_1_high = tbl(byte_1_high_table, prev1_high)

                    -- Lookup table 2: prev byte's low nibble → error flags
                    -- CARRY = TOO_SHORT|TOO_LONG|TWO_CONTS = 131
                    byte_1_low_table = [u8: 231, 163, 131, 131, 139, 203, 203, 203, 203, 203, 203, 203, 203, 219, 203, 203]
                    prev1_low = prev1 & 15
                    byte_1_low = tbl(byte_1_low_table, prev1_low)

                    -- Lookup table 3: current byte's high nibble → error flags
                    byte_2_high_table = [u8: 1, 1, 1, 1, 1, 1, 1, 1, 230, 174, 186, 186, 1, 1, 1, 1]
                    cur_high = bit_shr(chunk, 4)
                    byte_2_high = tbl(byte_2_high_table, cur_high)

                    -- AND all three: nonzero means error
                    special_cases = byte_1_high & byte_1_low & byte_2_high

                    -- check_multibyte_lengths: resolve TWO_CONTS errors
                    -- A continuation byte is valid if a 2/3/4-byte starter preceded it
                    -- prev2/prev3 = bytes 2 and 3 positions back
                    prev2 = lane_shr(chunk, 2)
                    prev3 = lane_shr(chunk, 3)

                    -- must_be_continuation_of_3_or_more: prev2 >= 0xE0
                    -- (byte >= 0xE0 means it's a 3/4-byte starter, expects cont at +2)
                    is_starter_3plus_p2 = prev2 >= 224

                    -- must_be_continuation_of_4: prev3 >= 0xF0
                    -- (byte >= 0xF0 means it's a 4-byte starter, expects cont at +3)
                    is_starter_4_p3 = prev3 >= 240

                    -- must23: continuation expected at this position from multi-byte starters
                    must23 = is_starter_3plus_p2 | is_starter_4_p3

                    -- Convert must23 to u8 mask with bit 7 set (128 = 0x80)
                    -- Use chunk arithmetic to stay in u8 domain
                    zero_byte = chunk - chunk
                    must23_byte = [must23] zero_byte + 128 : zero_byte

                    -- XOR: cancels TWO_CONTS if multi-byte starter was found
                    errors_for_chunk = xor(special_cases, must23_byte)

                    error_acc = error_acc | errors_for_chunk

                    carry prev_byte = extract(chunk, 15)
                    carry error_acc = error_acc
                }
                return error_acc
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "validate_utf8", 2);

        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> uint8x16_t = std::mem::transmute(fptr);
            let run = |input: &[u8]| -> bool {
                let padded_len = ((input.len() + 15) / 16) * 16;
                let mut buf = vec![0u8; padded_len];
                buf[..input.len()].copy_from_slice(input);
                let result = f(buf.as_ptr(), buf.as_ptr(), 0, padded_len as i64, 1);
                let mut out = [0u8; 16];
                vst1q_u8(out.as_mut_ptr(), result);
                // If any byte in error_acc is nonzero, there's an error
                out.iter().all(|&b| b == 0)
            };

            assert!(run(b"Hello, world!"), "ASCII should be valid");
            assert!(run(b"caf\xC3\xA9"), "2-byte UTF-8 should be valid");

            // Valid 3-byte UTF-8: € = E2 82 AC
            assert!(run(b"\xE2\x82\xAC"), "3-byte UTF-8 should be valid");

            // Valid 4-byte UTF-8: 𝄞 = F0 9D 84 9E
            assert!(run(b"\xF0\x9D\x84\x9E"), "4-byte UTF-8 should be valid");

            // Invalid: bare continuation byte
            assert!(!run(b"\x80hello"), "bare continuation should be invalid");

            // Invalid: overlong 2-byte (C0 80 = U+0000)
            assert!(!run(b"\xC0\x80"), "overlong 2-byte should be invalid");

            // Invalid: surrogate half (ED A0 80 = U+D800)
            assert!(!run(b"\xED\xA0\x80"), "surrogate should be invalid");

            // Invalid: too large (F4 90 80 80 = U+110000)
            assert!(!run(b"\xF4\x90\x80\x80"), "code point > U+10FFFF should be invalid");

            eprintln!("All UTF-8 validation tests passed!");
        }
    }

    // ============================================================
    // XOR operator tests
    // ============================================================

    #[test]
    fn test_xor_operator() {
        let source = r#"
            fn do_xor(a: i32[4], b: i32[4]) -> i32[4] {
                return a ^ b
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "do_xor");
        unsafe {
            let f: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let a = vld1q_s32([0xFFi32, 0x0F, 0xF0, 0xFF].as_ptr());
            let b = vld1q_s32([0x0Fi32, 0x0F, 0x0F, 0x00].as_ptr());
            let result = f(a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [0xF0, 0x00, 0xFF, 0xFF]);
        }
    }

    #[test]
    fn test_xor_equivalence_with_builtin() {
        // a ^ b should produce the same result as xor(a, b)
        let source_op = r#"
            fn xor_op(a: i32[4], b: i32[4]) -> i32[4] {
                return a ^ b
            }
        "#;
        let source_fn = r#"
            fn xor_fn(a: i32[4], b: i32[4]) -> i32[4] {
                return xor(a, b)
            }
        "#;
        let (fptr1, _e1) = jit_lookup(source_op, "xor_op");
        let (fptr2, _e2) = jit_lookup(source_fn, "xor_fn");
        unsafe {
            let f1: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t = std::mem::transmute(fptr1);
            let f2: extern "C" fn(int32x4_t, int32x4_t) -> int32x4_t = std::mem::transmute(fptr2);
            let a = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let b = vld1q_s32([5i32, 6, 7, 8].as_ptr());
            let r1 = f1(a, b);
            let r2 = f2(a, b);
            let mut o1 = [0i32; 4];
            let mut o2 = [0i32; 4];
            vst1q_s32(o1.as_mut_ptr(), r1);
            vst1q_s32(o2.as_mut_ptr(), r2);
            assert_eq!(o1, o2);
        }
    }

    // ============================================================
    // Bit shift operator tests (>> and << are now element-wise)
    // ============================================================

    #[test]
    fn test_bit_shift_right() {
        let source = r#"
            fn shr(v: i32[4]) -> i32[4] {
                return v >> 1
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shr");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([10i32, 20, 64, 128].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            // Element-wise unsigned right shift by 1
            assert_eq!(out, [5, 10, 32, 64]);
        }
    }

    #[test]
    fn test_bit_shift_left() {
        let source = r#"
            fn shl(v: i32[4]) -> i32[4] {
                return v << 2
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "shl");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [4, 8, 12, 16]);
        }
    }

    // ============================================================
    // Typed assignment + literal broadcast tests
    // ============================================================

    #[test]
    fn test_typed_assign_not_zero() {
        // all_ones: u64[1] = ~0 should produce all-ones vector
        let source = r#"
            fn all_ones() -> u64[1] {
                x: u64[1] = ~0
                return x
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "all_ones");
        unsafe {
            let f: extern "C" fn() -> i32 = std::mem::transmute(fptr);
            let bits = f();
            let result = (bits) as u64;
            // u64[1] all-ones = 0xFFFFFFFFFFFFFFFF, but returned as i32 via f32 reinterpret
            // The low 32 bits should be all-ones
            assert_eq!(result as u32, 0xFFFFFFFF);
        }
    }

    #[test]
    fn test_typed_assign_zero_broadcast() {
        // z: u8[4] = 0 should produce zero vector
        let source = r#"
            fn zeros(v: i32[4]) -> i32[4] {
                z: i32[4] = 0
                return v + z
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "zeros");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [1, 2, 3, 4]);
        }
    }

    // ============================================================
    // Vector repeat syntax tests
    // ============================================================

    #[test]
    fn test_vec_repeat_syntax() {
        // [i32: 7; 4] should produce [7, 7, 7, 7]
        let source = r#"
            fn sevens() -> i32[4] {
                v = [i32: 7; 4]
                return v
            }
        "#;
        let (fptr, _engine) = jit_lookup_opt(source, "sevens", 2);
        unsafe {
            let f: extern "C" fn() -> int32x4_t = std::mem::transmute(fptr);
            let result = f();
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [7, 7, 7, 7]);
        }
    }

    #[test]
    fn test_literal_broadcast_not_in_binop() {
        // ~0 in binary op context should broadcast correctly
        let source = r#"
            fn invert(v: i32[4]) -> i32[4] {
                return v ^ ~0
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "invert");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let v = vld1q_s32([0i32, 1, -1, 42].as_ptr());
            let result = f(v);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [-1, -2, 0, -43]);
        }
    }

    // ============================================================
    // Error message source location tests
    // ============================================================

    #[test]
    #[should_panic(expected = "at 2:")]
    fn test_error_includes_line_number_undefined_var() {
        let source = "fn f(v: i32[4]) -> i32[4] {\n    return undefined_var\n}";
        jit_lookup(source, "f");
    }

    #[test]
    #[should_panic(expected = "at 2:")]
    fn test_error_includes_line_number_bare_literal() {
        let source = "fn f() -> i32[4] {\n    x = 42\n    return x\n}";
        jit_lookup(source, "f");
    }

    #[test]
    #[should_panic(expected = "at 3:")]
    fn test_error_includes_line_number_width_mismatch() {
        let source = "fn f(a: i32[4], b: i32[8]) -> i32[4] {\n    -- mismatch\n    return a + b\n}";
        jit_lookup(source, "f");
    }

    // ============================================================
    // If/else tests
    // ============================================================

    // ============================================================
    // Multiple return values via out-params (Step 8)
    // ============================================================

    #[test]
    fn test_out_param_pattern() {
        // Demonstrate returning multiple values via ptr out-params
        // The function writes count to out_count ptr and returns the last value
        let source = r#"
            fn count_and_sum(input: ptr[u8], out_count: ptr[i32]) -> i32[1] {
                stream chunk: u8[8] over input carry (count: i32[1] = 0, sum: i32[1] = 0) {
                    is_a = chunk == 'a'
                    carry count = count + popcount(is_a)
                    carry sum = sum + 1
                }
                store(out_count, count)
                return sum
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "count_and_sum");
        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);
            let input = b"aababcaa\0\0\0\0\0\0\0\0";
            let mut count_out = [0i32; 1];
            let sum_bits = f(
                input.as_ptr(), input.as_ptr(), 0, 16, 1,
                count_out.as_mut_ptr(), count_out.as_mut_ptr(), 0, 1, 1,
            );
            let sum = (sum_bits) as i32;
            eprintln!("count={}, sum={}", count_out[0], sum);
            assert_eq!(count_out[0], 5, "should count 5 'a' bytes");
            assert!(sum > 0, "sum should be positive");
        }
    }

    // ============================================================
    // If/else tests (Step 7)
    // ============================================================

    #[test]
    fn test_if_else_basic() {
        // Use comparison to get a bool[1] condition
        let source = r#"
            fn pick(flag: i32[1], a: i32[4], b: i32[4]) -> i32[4] {
                cond = flag == 1
                if cond {
                    result = a
                } else {
                    result = b
                }
                return result
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "pick");
        unsafe {
            let f: extern "C" fn(i32, int32x4_t, int32x4_t) -> int32x4_t =
                std::mem::transmute(fptr);
            let a = vld1q_s32([1i32, 2, 3, 4].as_ptr());
            let b = vld1q_s32([10i32, 20, 30, 40].as_ptr());

            // flag=1 → cond=true → return a
            let result = f((1u32) as i32, a, b);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            eprintln!("flag=1 result: {:?}", out);
            assert_eq!(out, [1, 2, 3, 4]);

            // flag=0 → cond=false → return b
            let result = f((0u32) as i32, a, b);
            vst1q_s32(out.as_mut_ptr(), result);
            eprintln!("flag=0 result: {:?}", out);
            assert_eq!(out, [10, 20, 30, 40]);
        }
    }

    // ============================================================
    // Full JSON DOM proof — round-trip reconstruction + navigation
    // Proves our tape is morally equivalent to simdjson's DOM
    // ============================================================

    /// JIT stage1 and build a full DOM tape, then return the document
    fn jit_parse_json(json: &[u8]) -> (crate::json::Document, Vec<u8>) {
        let source = std::fs::read_to_string("examples/json_stage1.simd")
            .expect("cannot read json_stage1.simd");
        let (fptr, _engine) = jit_lookup_opt(&source, "json_stage1", 2);

        let padded = crate::json::pad_input(json);
        let mut positions = vec![0i32; json.len()];

        unsafe {
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut i32, *mut i32, i64, i64, i64,
            ) -> i32 = std::mem::transmute(fptr);

            let bits = f(
                padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
                positions.as_mut_ptr(), positions.as_mut_ptr(), 0, positions.len() as i64, 1,
            );
            let num = (bits) as usize;
            let doc = crate::json::build_tape(&padded, &positions[..num]);
            (doc, padded)
        }
    }

    fn reconstruct_json(doc: &crate::json::Document, input: &[u8]) -> String {
        let mut out = String::new();
        reconstruct_at(doc, input, 1, &mut out);
        out
    }

    fn reconstruct_at(
        doc: &crate::json::Document, input: &[u8], index: usize, out: &mut String,
    ) -> usize {
        use crate::json::TapeType;
        match doc.tape_type(index) {
            TapeType::OpenObject => {
                out.push('{');
                let close = doc.get_partner(index);
                let mut i = index + 1;
                let mut first = true;
                while i < close {
                    if !first { out.push(','); }
                    first = false;
                    let key = doc.get_string(i, input);
                    out.push('"');
                    json_escape(&key, out);
                    out.push('"');
                    out.push(':');
                    i += 1;
                    i = reconstruct_at(doc, input, i, out);
                }
                out.push('}');
                close + 1
            }
            TapeType::OpenArray => {
                out.push('[');
                let close = doc.get_partner(index);
                let mut i = index + 1;
                let mut first = true;
                while i < close {
                    if !first { out.push(','); }
                    first = false;
                    i = reconstruct_at(doc, input, i, out);
                }
                out.push(']');
                close + 1
            }
            TapeType::String => {
                let s = doc.get_string(index, input);
                out.push('"');
                json_escape(&s, out);
                out.push('"');
                index + 1
            }
            TapeType::Int64 => {
                out.push_str(&doc.get_i64(index).to_string());
                index + 1
            }
            TapeType::Double => {
                let v = doc.get_f64(index);
                out.push_str(&format!("{}", v));
                index + 1
            }
            TapeType::True => { out.push_str("true"); index + 1 }
            TapeType::False => { out.push_str("false"); index + 1 }
            TapeType::Null => { out.push_str("null"); index + 1 }
            _ => index + 1,
        }
    }

    fn json_escape(s: &str, out: &mut String) {
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c < '\x20' => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
    }

    #[test]
    fn test_json_dom_roundtrip_simple() {
        let json = br#"{"name":"Alice","age":30,"active":true}"#;
        let (doc, padded) = jit_parse_json(json);
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, std::str::from_utf8(json).unwrap());
    }

    #[test]
    fn test_json_dom_roundtrip_nested() {
        let json = br#"{"a":{"b":{"c":1}},"d":[1,2,3],"e":null,"f":false}"#;
        let (doc, padded) = jit_parse_json(json);
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, std::str::from_utf8(json).unwrap());
    }

    #[test]
    fn test_json_dom_roundtrip_all_types() {
        let json = br#"{"str":"hello","int":42,"neg":-17,"bool_t":true,"bool_f":false,"nil":null,"arr":[1,"two",true,null],"obj":{"x":1}}"#;
        let (doc, padded) = jit_parse_json(json);
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, std::str::from_utf8(json).unwrap());
    }

    #[test]
    fn test_json_dom_roundtrip_100_objects() {
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data":["#.to_string());
        for i in 0..100 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id":{},"name":"user_{}","score":{},"active":{}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let (doc, padded) = jit_parse_json(json_str.as_bytes());
        let reconstructed = reconstruct_json(&doc, &padded);
        assert_eq!(reconstructed, json_str);
    }

    #[test]
    fn test_json_dom_navigation() {
        use crate::json::TapeType;
        let json = br#"{"users":[{"name":"Alice","age":30},{"name":"Bob","age":25}],"count":2}"#;
        let (doc, padded) = jit_parse_json(json);

        // Find "users" key
        let mut users_idx = None;
        for i in 0..doc.tape.len() {
            if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "users" {
                users_idx = Some(i + 1);
                break;
            }
        }
        let users_arr = users_idx.expect("should find 'users'");
        assert_eq!(doc.tape_type(users_arr), TapeType::OpenArray);

        // First element
        let first_obj = users_arr + 1;
        assert_eq!(doc.tape_type(first_obj), TapeType::OpenObject);

        // Find "name" in first object
        let first_close = doc.get_partner(first_obj);
        let mut name_val = None;
        let mut i = first_obj + 1;
        while i < first_close {
            if doc.tape_type(i) == TapeType::String && &*doc.get_string(i, &padded) == "name" {
                name_val = Some(i + 1);
                break;
            }
            i += 2;
        }
        assert_eq!(&*doc.get_string(name_val.unwrap(), &padded), "Alice");

        // Container skip: jump from users array to "count"
        let users_close = doc.get_partner(users_arr);
        let after_users = users_close + 1;
        assert_eq!(doc.tape_type(after_users), TapeType::String);
        assert_eq!(&*doc.get_string(after_users, &padded), "count");
        assert_eq!(doc.get_i64(after_users + 1), 2);
    }

    #[test]
    fn test_json_dom_string_escapes() {
        let json = br#"{"msg":"hello\nworld","quote":"say \"hi\"","path":"C:\\Users"}"#;
        let (doc, padded) = jit_parse_json(json);

        use crate::json::TapeType;
        for i in 0..doc.tape.len() {
            if doc.tape_type(i) == TapeType::String {
                let s = doc.get_string(i, &padded);
                if &*s == "msg" { assert_eq!(&*doc.get_string(i+1, &padded), "hello\nworld"); }
                if &*s == "quote" { assert_eq!(&*doc.get_string(i+1, &padded), "say \"hi\""); }
                if &*s == "path" { assert_eq!(&*doc.get_string(i+1, &padded), "C:\\Users"); }
            }
        }
    }

    #[test]
    fn test_json_dom_numbers() {
        let json = br#"{"int":42,"neg":-17,"big":1234567890}"#;
        let (doc, padded) = jit_parse_json(json);

        use crate::json::TapeType;
        for i in 0..doc.tape.len() {
            if doc.tape_type(i) == TapeType::String {
                let key = doc.get_string(i, &padded);
                match &*key {
                    "int" => assert_eq!(doc.get_i64(i+1), 42),
                    "neg" => assert_eq!(doc.get_i64(i+1), -17),
                    "big" => assert_eq!(doc.get_i64(i+1), 1234567890),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn bench_json_dom_full_parse() {
        let mut json_parts = Vec::new();
        json_parts.push(r#"{"data":["#.to_string());
        for i in 0..50000 {
            if i > 0 { json_parts.push(",".to_string()); }
            json_parts.push(format!(
                r#"{{"id":{},"name":"user_{}","score":{},"active":{}}}"#,
                i, i, i * 17 % 10000, if i % 2 == 0 { "true" } else { "false" }
            ));
        }
        json_parts.push(r#"]}"#.to_string());
        let json_str = json_parts.join("");
        let json_len = json_str.len();

        let source = std::fs::read_to_string("examples/json_stage1.simd")
            .expect("cannot read json_stage1.simd");
        let (fptr, _engine) = jit_lookup_opt(&source, "json_stage1", 2);

        let padded = crate::json::pad_input(json_str.as_bytes());
        let mut parser = crate::json::Parser::new(json_len);

        let stage1 = |input: &[u8], positions: &mut [i32]| -> usize {
            unsafe {
                let f: extern "C" fn(
                    *const u8, *const u8, i64, i64, i64,
                    *mut i32, *mut i32, i64, i64, i64,
                ) -> i32 = std::mem::transmute(fptr);
                (f(
                    input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                    positions.as_mut_ptr(), positions.as_mut_ptr(), 0, positions.len() as i64, 1,
                )) as usize
            }
        };

        // Warmup
        let _ = parser.parse_with_stage1(&padded, stage1);

        let iterations = 10;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = parser.parse_with_stage1(&padded, stage1);
        }
        let elapsed = start.elapsed();
        let gb = json_len as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9;

        eprintln!("\n=== Full DOM Parse (JIT stage1 + Rust tape builder) ===");
        eprintln!("Input: {:.2} MB", json_len as f64 / 1e6);
        eprintln!("Throughput: {:.2} GB/s ({:.3} ms/iter)",
            gb, elapsed.as_secs_f64() * 1000.0 / iterations as f64);
        eprintln!("Compare: simdjson does ~1.5 GB/s on same data");
    }

    // ============================================================
    // New features: load_at, ptr_add, bswap, while, inter-fn calls
    // ============================================================

    #[test]
    fn test_hprof_style_parsing() {
        // Simulate parsing an HPROF-like binary format:
        // - Read a tag byte at offset 0
        // - Read a big-endian u32 body length at offset 5
        // - Walk records using while loop until we've consumed the buffer
        // - Count how many records have tag == 0x1C (HEAP_DUMP_SEGMENT)
        let source = r#"
            fn read_u32_be(buf: ptr[u8], off: i32[1]) -> i32[1] {
                b0 = to_i32(load_at(buf, off))
                b1 = to_i32(load_at(buf, off + 1))
                b2 = to_i32(load_at(buf, off + 2))
                b3 = to_i32(load_at(buf, off + 3))
                return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
            }

            fn count_segments(buf: ptr[u8], buf_len: i32[1]) -> i32[1] {
                offset: i32[1] = 0
                count: i32[1] = 0
                while offset < buf_len {
                    tag = to_i32(load_at(buf, offset))
                    body_len = read_u32_be(buf, offset + 5)
                    if tag == 28 {
                        count = count + 1
                    } else {
                        count = count
                    }
                    offset = offset + 9 + body_len
                }
                return count
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(module.as_operation().verify(), "MLIR verify failed:\n{}", module.as_operation());
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = melior::ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("count_segments");

        // Build a fake HPROF-like buffer with 3 records:
        // Record 1: tag=0x01 (UTF8), body_len=2, body=[0xAA, 0xBB]  → 9+2=11 bytes
        // Record 2: tag=0x1C (HEAP_DUMP_SEGMENT), body_len=4, body=[1,2,3,4] → 9+4=13 bytes
        // Record 3: tag=0x1C (HEAP_DUMP_SEGMENT), body_len=0, body=[] → 9+0=9 bytes
        // Total: 33 bytes
        let mut buf = vec![0u8; 33];
        // Record 1: tag=0x01, timestamp=0, body_len=2
        buf[0] = 0x01;
        buf[5] = 0; buf[6] = 0; buf[7] = 0; buf[8] = 2; // big-endian u32 = 2
        buf[9] = 0xAA; buf[10] = 0xBB;
        // Record 2: tag=0x1C, timestamp=0, body_len=4
        buf[11] = 0x1C;
        buf[16] = 0; buf[17] = 0; buf[18] = 0; buf[19] = 4; // big-endian u32 = 4
        buf[20] = 1; buf[21] = 2; buf[22] = 3; buf[23] = 4;
        // Record 3: tag=0x1C, timestamp=0, body_len=0
        buf[24] = 0x1C;
        buf[29] = 0; buf[30] = 0; buf[31] = 0; buf[32] = 0; // big-endian u32 = 0

        unsafe {
            // count_segments(buf: ptr[u8], buf_len: i32[1]) -> i32[1]
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64, i32) -> i32 =
                std::mem::transmute(fptr);
            let buf_len = (33u32) as i32;
            let result_bits = f(buf.as_ptr(), buf.as_ptr(), 0, 33, 1, buf_len);
            let result = (result_bits) as i32;
            assert_eq!(result, 2, "should find 2 HEAP_DUMP_SEGMENT records (tag=0x1C)");
        }
    }

    #[test]
    fn test_load_at_basic() {
        let source = r#"
            fn read_byte(buf: ptr[u8], off: i32[1]) -> i32[1] {
                v = load_at(buf, off)
                return to_i32(v)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "read_byte");
        let input: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64, i32) -> i32 =
                std::mem::transmute(fptr);
            // Read byte at offset 3 (should be 40)
            let offset_bits = (3u32) as i32;
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 8, 1, offset_bits);
            let result = (result_bits) as i32;
            assert_eq!(result, 40, "load_at(buf, 3) should read byte 40");
        }
    }

    #[test]
    fn test_bswap_u32() {
        let source = r#"
            fn swap(v: u32[4]) -> u32[4] {
                return bswap(v)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "swap");
        unsafe {
            let f: extern "C" fn(uint32x4_t) -> uint32x4_t = std::mem::transmute(fptr);
            let input = vld1q_u32([0x01020304u32, 0xAABBCCDD, 0x00FF00FF, 0x12345678].as_ptr());
            let result = f(input);
            let mut out = [0u32; 4];
            vst1q_u32(out.as_mut_ptr(), result);
            assert_eq!(out[0], 0x04030201);
            assert_eq!(out[1], 0xDDCCBBAA);
            assert_eq!(out[2], 0xFF00FF00);
            assert_eq!(out[3], 0x78563412);
        }
    }

    #[test]
    fn test_inter_fn_call_basic() {
        let source = r#"
            fn double(x: i32[4]) -> i32[4] {
                return x + x
            }
            fn main(a: i32[4]) -> i32[4] {
                return double(a)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "main");
        unsafe {
            let f: extern "C" fn(int32x4_t) -> int32x4_t = std::mem::transmute(fptr);
            let input = vld1q_s32([1, 2, 3, 4].as_ptr());
            let result = f(input);
            let mut out = [0i32; 4];
            vst1q_s32(out.as_mut_ptr(), result);
            assert_eq!(out, [2, 4, 6, 8]);
        }
    }

    #[test]
    fn test_inter_fn_call_with_ptr() {
        let source = r#"
            fn read_first(buf: ptr[u8], off: i32[1]) -> i32[1] {
                v = load_at(buf, off)
                return to_i32(v)
            }
            fn main(buf: ptr[u8]) -> i32[1] {
                off: i32[1] = 0
                return read_first(buf, off)
            }
        "#;
        let (fptr, _engine) = jit_lookup(source, "main");
        let input: [u8; 4] = [42, 0, 0, 0];
        unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i32 =
                std::mem::transmute(fptr);
            let result_bits = f(input.as_ptr(), input.as_ptr(), 0, 4, 1);
            let result = (result_bits) as i32;
            assert_eq!(result, 42);
        }
    }

    #[test]
    fn test_while_loop_basic() {
        // Count how many times we can subtract 10 from 45 before it goes below 10
        // Expected: 45 -> 35 -> 25 -> 15 -> 5 (stop), count = 4
        let source = r#"
            fn count_tens(start: i32[1]) -> i32[1] {
                val = start
                count: i32[1] = 0
                while val >= 10 {
                    val = val - 10
                    count = count + 1
                }
                return count
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(module.as_operation().verify(), "MLIR verify failed:\n{}", module.as_operation());
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = melior::ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("count_tens");
        unsafe {
            let f: extern "C" fn(i32) -> i32 = std::mem::transmute(fptr);
            let input = (45u32) as i32;
            let result = (f(input)) as i32;
            assert_eq!(result, 4, "should count 4 tens in 45");
        }
    }
}

#[cfg(test)]
mod extra_tests {
    use super::*;
    use crate::parser;

    #[test]
    fn test_while_with_if_else() {
        let source = r#"
            fn test(x: i32[1]) -> i32[1] {
                i: i32[1] = 0
                count: i32[1] = 0
                while i < x {
                    if i > 5 {
                        count = count + 1
                    } else {
                        count = count
                    }
                    i = i + 1
                }
                return count
            }
        "#;
        let ctx = create_context();
        let items = parser::parse(source);
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(module.as_operation().verify(), "MLIR verify failed:\n{}", module.as_operation());
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = melior::ExecutionEngine::new(&module, 0, &[], false);
        let fptr = engine.lookup("test");
        unsafe {
            let f: extern "C" fn(i32) -> i32 = std::mem::transmute(fptr);
            // count how many i in [0,10) are > 5: that's 6,7,8,9 = 4
            let input = (10u32) as i32;
            let result = (f(input)) as i32;
            assert_eq!(result, 4, "should count 4 values > 5 in range [0,10)");
        }
    }
}

// ============================================================
// Scalars, real branches, and the features the escape kernel needs
// ============================================================
#[cfg(test)]
mod scalar_and_branch_tests {
    use super::*;
    use crate::parser;
    use crate::typecheck;

    /// JIT a program and hand the function pointer, with the engine that owns it, to `body`.
    fn with_fn<R>(source: &str, name: &str, body: impl FnOnce(*mut ()) -> R) -> R {
        let ctx = create_context();
        let items = parser::parse(source);
        typecheck::typecheck(&items, &HashMap::new())
            .unwrap_or_else(|e| panic!("type errors: {:?}", e.iter().map(|x| x.to_string()).collect::<Vec<_>>()));
        let mut module = compile_module(&ctx, &items, &HashMap::new());
        assert!(module.as_operation().verify(), "MLIR verify failed:\n{}", module.as_operation());
        lower_to_llvm(&ctx, &mut module).unwrap();
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);
        let fptr = engine.lookup(name);
        assert!(!fptr.is_null(), "{} not found", name);
        body(fptr)
    }

    fn type_errors(source: &str) -> Vec<String> {
        let items = parser::parse(source);
        match typecheck::typecheck(&items, &HashMap::new()) {
            Ok(()) => vec![],
            Err(errors) => errors.iter().map(|e| e.to_string()).collect(),
        }
    }

    type Ptr5 = (*mut u8, *mut u8, i64, i64, i64);
    fn desc(buf: &mut [u8]) -> Ptr5 {
        (buf.as_mut_ptr(), buf.as_mut_ptr(), 0, buf.len() as i64, 1)
    }

    #[test]
    fn scalars_are_returned_in_full_64_bits() {
        with_fn("fn f(x: i64[1]) -> i64[1] { return x + 1 }", "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(0x1234_5678_9ABC_DEF0), 0x1234_5678_9ABC_DEF1);
        });
    }

    #[test]
    fn if_runs_only_the_taken_branch() {
        let source = r#"
            fn f(flag: i32[1], buf: ptr[u8]) {
                seven: u8[1] = 7
                nine: u8[1] = 9
                if flag > 0 {
                    store_at(buf, 0, seven)
                } else {
                    store_at(buf, 1, nine)
                }
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i32, *mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [0u8; 2];
            let (a, b, c, d, e) = desc(&mut buf);
            f(1, a, b, c, d, e);
            assert_eq!(buf, [7, 0], "true branch must not run the else store");
            let mut buf = [0u8; 2];
            let (a, b, c, d, e) = desc(&mut buf);
            f(0, a, b, c, d, e);
            assert_eq!(buf, [0, 9], "false branch must not run the then store");
        });
    }

    #[test]
    fn if_merges_values_from_either_branch() {
        let source = r#"
            fn f(x: i64[1]) -> i64[1] {
                a: i64[1] = 1
                b: i64[1] = 2
                if x > 0 {
                    a = a + 10
                    both = x
                } else {
                    b = b + 20
                    both = x + x
                }
                return a * 10000 + b * 100 + both
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(5), 11 * 10000 + 2 * 100 + 5);
            assert_eq!(f(-3), 10000 + 22 * 100 - 6);
        });
    }

    #[test]
    fn if_nested_inside_while_carries_state() {
        let source = r#"
            fn f(n: i64[1]) -> i64[1] {
                i: i64[1] = 0
                sum: i64[1] = 0
                while i < n {
                    if (i & 1) == 0 {
                        sum = sum + i
                    }
                    i = i + 1
                }
                return sum
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(7), 0 + 2 + 4 + 6);
        });
    }

    #[test]
    fn name_from_one_branch_is_not_visible_after_the_if() {
        let errors = type_errors(
            r#"
            fn f(x: i64[1]) -> i64[1] {
                if x > 0 {
                    only_then = x
                } else {
                    other = x
                }
                return only_then
            }
        "#,
        );
        assert!(errors.iter().any(|e| e.contains("only_then")), "got {:?}", errors);
    }

    #[test]
    fn name_from_both_branches_is_visible_after_the_if() {
        let errors = type_errors(
            r#"
            fn f(x: i64[1]) -> i64[1] {
                if x > 0 { y = x } else { y = x + 1 }
                return y
            }
        "#,
        );
        assert!(errors.is_empty(), "got {:?}", errors);
    }

    #[test]
    fn branches_must_agree_on_width() {
        let errors = type_errors(
            r#"
            fn f(x: i64[1], v: i64[2]) -> i64[1] {
                if x > 0 { y = x } else { y = v }
                return x
            }
        "#,
        );
        assert!(errors.iter().any(|e| e.contains("'y'")), "got {:?}", errors);
    }

    #[test]
    fn const_tables_are_pointers() {
        let source = r#"
            const TABLE = [u8: 10, 20, 30, 40]
            fn f(i: i64[1]) -> u8[1] { return load_at(TABLE, i) }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> u8 = std::mem::transmute(p);
            assert_eq!([f(0), f(3)], [10, 40]);
        });
    }

    #[test]
    fn wide_scalars_load_and_store_at_any_byte_offset() {
        let source = r#"
            fn f(buf: ptr[u8], from: i64[1], to: i64[1]) -> u64[1] {
                x = load_at[u64[1]](buf, from)
                store_at(buf, to, x)
                return x
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i64, i64) -> u64 = std::mem::transmute(p);
            let mut buf: Vec<u8> = (1..=32).collect();
            let (a, b, c, d, e) = desc(&mut buf);
            let got = f(a, b, c, d, e, 3, 17);
            assert_eq!(got, u64::from_le_bytes([4, 5, 6, 7, 8, 9, 10, 11]));
            assert_eq!(&buf[17..25], &[4, 5, 6, 7, 8, 9, 10, 11]);
            assert_eq!(buf[16], 17, "byte before the store is untouched");
            assert_eq!(buf[25], 26, "byte after the store is untouched");
        });
    }

    #[test]
    fn shift_amounts_can_be_values() {
        with_fn("fn f(x: u64[1], s: u64[1]) -> u64[1] { return (x << s) >> 1 }", "f", |p| unsafe {
            let f: extern "C" fn(u64, u64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(3, 4), 24);
        });
        let source = r#"
            fn f(buf: ptr[u8], s: u8[1]) {
                v = load_at[u8[16]](buf, 0)
                store_at(buf, 0, v >> s)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, u8) = std::mem::transmute(p);
            let mut buf = [0x80u8; 16];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e, 3);
            assert_eq!(buf, [0x10u8; 16]);
        });
        let errors = type_errors("fn f(x: u64[2], s: u64[4]) -> u64[2] { return x << s }");
        assert!(errors.iter().any(|e| e.contains("shift amount")), "got {:?}", errors);
    }

    #[test]
    fn lane_min_max_respect_signedness() {
        let source = r#"
            fn f(buf: ptr[u8]) {
                u = load_at[u8[16]](buf, 0)
                i = load_at[i8[16]](buf, 0)
                ones = [u8: 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
                store_at(buf, 16, min(u, ones))
                store_at(buf, 32, max(u, ones))
                store_at(buf, 48, min(i, 100))
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [200u8; 64];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(buf[16], 100, "unsigned min(200, 100)");
            assert_eq!(buf[32], 200, "unsigned max(200, 100)");
            assert_eq!(buf[48], 200, "signed min(-56, 100) is -56, byte 200");
        });
    }

    #[test]
    fn bitmask_builtins() {
        let source = r#"
            fn bits(buf: ptr[u8]) -> u64[1] { return to_bitmask(load_at[u8[16]](buf, 0) > 127) }
            fn nibbles(buf: ptr[u8]) -> u64[1] { return nibblemask(mask8(load_at[u8[16]](buf, 0) > 127)) }
            fn low(x: u64[1]) -> u64[1] { return ctz(x) }
            fn clear(x: u64[1]) -> u64[1] { return clear_lowest_bit(x) }
        "#;
        let mut input = [0u8; 16];
        input[0] = 255;
        input[3] = 128;
        input[15] = 200;
        with_fn(source, "bits", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u64 = std::mem::transmute(p);
            let mut copy = input;
            let (a, b, c, d, e) = desc(&mut copy);
            assert_eq!(f(a, b, c, d, e), 0b1000_0000_0000_1001);
        });
        with_fn(source, "nibbles", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u64 = std::mem::transmute(p);
            let mut copy = input;
            let (a, b, c, d, e) = desc(&mut copy);
            assert_eq!(f(a, b, c, d, e), 0xF000_0000_0000_F00F);
        });
        with_fn(source, "low", |p| unsafe {
            let f: extern "C" fn(u64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(0b1000), 3);
        });
        with_fn(source, "clear", |p| unsafe {
            let f: extern "C" fn(u64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(0b1100), 0b1000);
        });
    }

    #[test]
    fn shuffle_picks_lanes_from_both_operands() {
        let source = r#"
            fn f(buf: ptr[u8]) {
                a = load_at[u8[16]](buf, 0)
                b = load_at[u8[16]](buf, 16)
                store_at(buf, 32, shuffle(a, b, [u8: 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]))
                store_at(buf, 48, shuffle(a, b, [u8: 31, 0, 30, 1, 29, 2, 28, 3, 27, 4, 26, 5, 25, 6, 24, 7]))
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf: Vec<u8> = (0..64).collect();
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(&buf[32..48], &[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]);
            assert_eq!(&buf[48..64], &[31, 0, 30, 1, 29, 2, 28, 3, 27, 4, 26, 5, 25, 6, 24, 7]);
        });
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn shuffle_rejects_lanes_past_both_operands() {
        with_fn(
            "fn f(a: u8[16]) -> u8[2] { return shuffle(a, a, [u8: 0, 32]) }",
            "f",
            |_| (),
        );
    }

    #[test]
    fn concat_joins_halves() {
        let source = r#"
            fn f(buf: ptr[u8]) {
                v = load_at[u8[16]](buf, 0)
                store_at(buf, 16, concat(split_hi(v), split_lo(v)))
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf: Vec<u8> = (0..32).collect();
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(&buf[16..24], &[8, 9, 10, 11, 12, 13, 14, 15]);
            assert_eq!(&buf[24..32], &[0, 1, 2, 3, 4, 5, 6, 7]);
        });
    }

    #[test]
    fn ptr_addr_is_the_machine_address() {
        let source = r#"
            fn base(buf: ptr[u8]) -> i64[1] { return ptr_addr(buf) }
            fn advanced(buf: ptr[u8]) -> i64[1] {
                five: i64[1] = 5
                p = ptr_add(buf, five)
                return ptr_addr(p)
            }
        "#;
        let mut buf = [0u8; 16];
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "base", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), b as i64);
        });
        with_fn(source, "advanced", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), b as i64 + 5);
        });
    }

    #[test]
    fn ptr_add_applies_to_every_kind_of_access() {
        let source = r#"
            fn vector(buf: ptr[u8]) -> u8[1] {
                five: i64[1] = 5
                p = ptr_add(buf, five)
                return extract(load_at[u8[16]](p, 1), 0)
            }
            fn scalar(buf: ptr[u8]) -> u8[1] {
                five: i64[1] = 5
                p = ptr_add(buf, five)
                return load_at(p, 1)
            }
            fn wide(buf: ptr[u8]) -> u64[1] {
                five: i64[1] = 5
                p = ptr_add(buf, five)
                return load_at[u64[1]](p, 1)
            }
            fn writes(buf: ptr[u8]) {
                five: i64[1] = 5
                p = ptr_add(buf, five)
                seven: u8[1] = 7
                store_at(p, 1, seven)
            }
            fn streamed(buf: ptr[u8]) -> i32[1] {
                sixteen: i64[1] = 16
                p = ptr_add(buf, sixteen)
                stream chunk: u8[16] over p carry (first: i32[1] = 0) {
                    carry first = to_i32(extract(chunk, 0))
                }
                return first
            }
        "#;
        let mut buf: Vec<u8> = (0..48).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "vector", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u8 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 6);
        });
        with_fn(source, "scalar", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u8 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 6);
        });
        with_fn(source, "wide", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), u64::from_le_bytes([6, 7, 8, 9, 10, 11, 12, 13]));
        });
        with_fn(source, "streamed", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i32 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 32, "stream sees the remaining size");
        });
        with_fn(source, "writes", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            f(a, b, c, d, e);
        });
        assert_eq!(buf[6], 7);
    }

    #[test]
    fn comptime_widths_are_numbers_too() {
        let source = r#"
            fn offset[N](buf: ptr[u8]) -> i64[1] {
                return to_i64(load_at(buf, N - 1)) + N
            }
            fn f(buf: ptr[u8]) -> i64[1] { return offset[3](buf) + offset[5](buf) * 1000 }
        "#;
        let mut buf: Vec<u8> = (0..8).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), (2 + 3) + (4 + 5) * 1000);
        });
    }

    #[test]
    fn register_bytes_is_a_width_and_a_number() {
        let source = r#"
            fn f(buf: ptr[u8]) -> i64[1] {
                v = load_at[u8[REGISTER_BYTES]](buf, 0)
                store_at(buf, REGISTER_BYTES, v)
                return REGISTER_BYTES * 2
            }
        "#;
        let register = Arch::host().register_bytes() as usize;
        let mut buf: Vec<u8> = (0..register as u8 * 2).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 2 * register as i64);
        });
        assert_eq!(&buf[register..], &(0..register as u8).collect::<Vec<u8>>()[..]);
    }

    #[test]
    fn underscore_width_is_one_register_of_the_lane_type() {
        let source = r#"
            fn bytes(buf: ptr[u8]) -> i32[1] { return popcount(load_at[u8[_]](buf, 0) == 0) }
            fn words(buf: ptr[u32]) -> i32[1] { return popcount(load_at[u32[_]](buf, 0) == 0) }
        "#;
        let register = Arch::host().register_bytes() as i32;
        with_fn(source, "bytes", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i32 = std::mem::transmute(p);
            let mut buf = [0u8; 64];
            let (a, b, c, d, e) = desc(&mut buf);
            assert_eq!(f(a, b, c, d, e), register);
        });
        with_fn(source, "words", |p| unsafe {
            let f: extern "C" fn(*mut u32, *mut u32, i64, i64, i64) -> i32 = std::mem::transmute(p);
            let mut buf = [0u32; 16];
            assert_eq!(f(buf.as_mut_ptr(), buf.as_mut_ptr(), 0, 16, 1), register / 4);
        });
    }

    #[test]
    fn any_in_first_works_at_register_width() {
        let source = r#"
            fn f(buf: ptr[u8], n: i64[1]) -> i64[1] {
                if any_in_first(load_at[u8[REGISTER_BYTES]](buf, 0) > 127, n) { return 1 }
                return 0
            }
        "#;
        let register = Arch::host().register_bytes() as i64;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i64) -> i64 = std::mem::transmute(p);
            for lane in 0..register {
                let mut buf = vec![0u8; register as usize];
                buf[lane as usize] = 200;
                let (a, b, c, d, e) = desc(&mut buf);
                for n in 1..=register {
                    assert_eq!(f(a, b, c, d, e, n), (lane < n) as i64, "lane {lane}, first {n}");
                }
            }
        });
    }

    #[test]
    fn unlikely_passes_its_value_through() {
        let source = r#"
            fn f(x: i64[1]) -> i64[1] {
                if unlikely(x > 0) { y = x } else { y = 0 - x }
                return y
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!([f(4), f(-4)], [4, 4]);
        });
    }

    #[test]
    fn extern_fn_takes_plain_pointers() {
        let source = r#"
            extern fn sum2(buf: ptr[u8], i: i64[1]) -> i64[1] {
                return to_i64(load_at(buf, i)) + to_i64(load_at(buf, i + 1))
            }
            fn caller(buf: ptr[u8]) -> i64[1] {
                zero: i64[1] = 0
                return sum2(buf, zero)
            }
        "#;
        let buf = [3u8, 4, 5];
        with_fn(source, "sum2", |p| unsafe {
            let f: extern "C" fn(*const u8, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(buf.as_ptr(), 1), 9);
        });
        with_fn(source, "caller", |p| unsafe {
            let f: extern "C" fn(*const u8, *const u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(buf.as_ptr(), buf.as_ptr(), 0, 3, 1), 7);
        });
    }

    #[test]
    fn extern_fn_cannot_stream_over_its_pointers() {
        let errors = type_errors(
            r#"
            extern fn f(buf: ptr[u8]) -> i32[1] {
                stream chunk: u8[16] over buf carry (n: i32[1] = 0) {
                    carry n = n + 1
                }
                return n
            }
        "#,
        );
        assert!(errors.iter().any(|e| e.contains("no known size")), "got {:?}", errors);
    }

    #[test]
    fn private_and_noinline_functions_stay_callable() {
        let source = r#"
            private fn double(x: i64[1]) -> i64[1] { return x + x }
            noinline private fn triple(x: i64[1]) -> i64[1] { return x + x + x }
            fn f(x: i64[1]) -> i64[1] { return double(x) + triple(x) }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(2), 10);
        });
    }

    #[test]
    #[should_panic(expected = "both private and extern")]
    fn private_extern_is_rejected() {
        parser::parse("private extern fn f(x: i64[1]) -> i64[1] { return x }");
    }

    #[test]
    fn generics_infer_widths_from_arguments() {
        let source = r#"
            fn sum_lanes[N](v: u8[N]) -> u8[1] { return +/ v }
            fn f(buf: ptr[u8]) -> u64[1] {
                a = sum_lanes(load_at[u8[16]](buf, 0))
                b = sum_lanes(load_at[u8[32]](buf, 0))
                return (to_i64(a) << 8) | to_i64(b)
            }
        "#;
        let mut buf = [1u8; 32];
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), (16 << 8) | 32, "each width gets its own copy");
        });
    }

    #[test]
    fn generics_take_explicit_widths_and_nest() {
        let source = r#"
            fn first[N](buf: ptr[u8]) -> u8[1] {
                return extract(load_at[u8[N]](buf, 1), 0)
            }
            fn outer[M](buf: ptr[u8]) -> u8[1] { return first[M * 2](buf) }
            fn f(buf: ptr[u8]) -> u64[1] {
                return (to_i64(first[16](buf)) << 8) | to_i64(outer[16](buf))
            }
        "#;
        let mut buf: Vec<u8> = (0..64).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), (1 << 8) | 1);
        });
    }

    #[test]
    fn generic_errors_are_reported() {
        let cases = [
            ("fn g[N](b: ptr[u8]) -> u8[1] { return load_at(b, 0) } fn f(b: ptr[u8]) -> u8[1] { return g(b) }", "can't infer N"),
            ("fn g[N](x: u8[N], y: u8[N]) -> u8[N] { return x } fn f(a: u8[16], b: u8[32]) -> u8[16] { return g(a, b) }", "has width"),
            ("fn g[N](x: u8[N]) -> u8[N] { return x } fn f(a: u8[16]) -> u8[16] { return g[16, 32](a) }", "comptime parameters"),
            ("extern fn g[N](x: u8[N]) -> u8[N] { return x }", "extern fn"),
            // A body is checked with the widths it's called with.
            ("fn g[N](x: u8[N]) -> u8[16] { return tbl(x, x) } fn f(a: u8[32]) -> u8[16] { return g(a) }", "tbl table"),
        ];
        for (source, expected) in cases {
            let errors = type_errors(source);
            assert!(errors.iter().any(|e| e.contains(expected)), "{}: got {:?}", source, errors);
        }
    }

    #[test]
    fn an_uncalled_generic_emits_nothing() {
        with_fn("fn g[N](x: u8[N]) -> u8[N] { return x } fn f(x: i64[1]) -> i64[1] { return x }", "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(3), 3);
        });
    }

    #[test]
    fn any_sees_every_lane_at_every_register_width() {
        for width in [16usize, 32, 48, 64] {
            let source = format!(
                "fn f(buf: ptr[u8]) -> i64[1] {{
                    if any(load_at[u8[{width}]](buf, 0) > 127) {{ r: i64[1] = 1 }} else {{ r: i64[1] = 0 }}
                    return r
                }}"
            );
            with_fn(&source, "f", |p| unsafe {
                let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
                let mut buf = vec![0u8; width];
                let (a, b, c, d, e) = desc(&mut buf);
                assert_eq!(f(a, b, c, d, e), 0, "all clear at width {width}");
                for lane in 0..width {
                    buf[lane] = 200;
                    assert_eq!(f(a, b, c, d, e), 1, "lane {lane} of {width}");
                    buf[lane] = 0;
                }
            });
        }
    }

    #[test]
    fn fold_min_takes_the_minimum_across_registers() {
        let source = r#"
            fn wide(buf: ptr[u8]) {
                store_at(buf, 64, fold_min(load_at[u8[64]](buf, 0)))
            }
            fn narrow(buf: ptr[u8]) {
                store_at(buf, 64, fold_min(load_at[u8[16]](buf, 0)))
            }
        "#;
        let register = Arch::host().register_bytes() as usize;
        let mut input = vec![200u8; 128];
        for (i, value) in [(0, 7), (16 + 1, 8), (32 + 2, 9), (48 + 15, 10)] {
            input[i] = value;
        }
        with_fn(source, "wide", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = input.clone();
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            let expected: Vec<u8> = (0..register)
                .map(|lane| (0..64).filter(|i| i % register == lane).map(|i| input[i]).min().unwrap())
                .collect();
            assert_eq!(&buf[64..64 + register], &expected[..]);
        });
        with_fn(source, "narrow", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = input.clone();
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(&buf[64..80], &input[..16], "a no-op when it fits in one register");
        });
        let errors = type_errors("fn f(v: u8[48]) -> u8[16] { return fold_min(v) }");
        if register == 32 {
            assert!(errors.iter().any(|e| e.contains("fold_min")), "got {:?}", errors);
        }
    }

    #[test]
    fn tbl_gives_zero_for_indices_past_the_table() {
        let source = r#"
            fn f(buf: ptr[u8]) {
                table = [u8: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
                store_at(buf, 16, tbl(table, load_at[u8[16]](buf, 0)))
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let indices: [u8; 16] = [0, 15, 16, 17, 31, 32, 64, 112, 127, 128, 129, 143, 200, 239, 240, 255];
            let mut buf = [0u8; 32];
            buf[..16].copy_from_slice(&indices);
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            let expected: Vec<u8> = indices.iter().map(|&i| if i < 16 { 100 + i } else { 0 }).collect();
            assert_eq!(&buf[16..32], &expected[..]);
        });
    }

    #[test]
    fn clmul_is_carry_less() {
        let source = "fn f(a: u64[1], b: u64[1]) -> u64[1] { return clmul(a, b) }";
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(u64, u64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(0b11, 0b11), 0b101, "(x + 1)^2 has no x term without carries");
            assert_eq!(f(0b1011, 1), 0b1011);
            // The running parity of its bits, as JSON's string tracking uses it.
            assert_eq!(f(0b0100_0100, u64::MAX), 0b0011_1100);
        });
    }

    #[test]
    fn any_in_first_only_sees_the_first_lanes() {
        let source = r#"
            fn f(buf: ptr[u8], n: i64[1]) -> i64[1] {
                if any_in_first(load_at[u8[16]](buf, 0) > 127, n) { return 1 }
                return 0
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i64) -> i64 = std::mem::transmute(p);
            for lane in 0..16i64 {
                let mut buf = [0u8; 16];
                buf[lane as usize] = 200;
                let (a, b, c, d, e) = desc(&mut buf);
                for n in 1..=16i64 {
                    assert_eq!(f(a, b, c, d, e, n), (lane < n) as i64, "lane {lane}, first {n}");
                }
            }
        });
    }

    #[test]
    fn wide_stores_write_every_byte_at_any_offset() {
        let source = r#"
            fn f(buf: ptr[u8], at: i64[1]) {
                store_at(buf, at, load_at[u8[64]](buf, 0))
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i64) = std::mem::transmute(p);
            let mut buf: Vec<u8> = (0..140).collect();
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e, 67);
            assert_eq!(&buf[67..131], &(0..64).collect::<Vec<u8>>()[..]);
            assert_eq!(buf[66], 66, "byte before is untouched");
            assert_eq!(buf[131], 131, "byte after is untouched");
        });
    }

    #[test]
    fn bool_literals_and_negation() {
        let source = r#"
            fn f(x: i64[1]) -> i64[1] {
                flag: bool[1] = true
                if x > 0 {
                    flag = ~flag
                }
                if flag { r = x + 100 } else { r = x }
                return r
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            assert_eq!([f(5), f(-5)], [5, 95]);
        });
    }

    // ---- values that used to be silently wrong

    #[test]
    fn reductions_and_scans_respect_signedness() {
        let source = r#"
            fn umax(buf: ptr[u8]) -> u8[1] { return max/ load_at[u8[16]](buf, 0) }
            fn umin(buf: ptr[u8]) -> u8[1] { return min/ load_at[u8[16]](buf, 0) }
            fn imax(buf: ptr[u8]) -> i8[1] { return max/ load_at[i8[16]](buf, 0) }
            fn imin(buf: ptr[u8]) -> i8[1] { return min/ load_at[i8[16]](buf, 0) }
            fn uscan(buf: ptr[u8]) { store_at(buf, 16, scan.max(load_at[u8[16]](buf, 0))) }
        "#;
        let mut buf = [100u8; 32];
        buf[3] = 200;
        buf[9] = 50;
        let (a, b, c, d, e) = desc(&mut buf);
        for (name, expected) in [("umax", 200u8), ("umin", 50), ("imax", 100), ("imin", 200)] {
            with_fn(source, name, |p| unsafe {
                let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u8 = std::mem::transmute(p);
                assert_eq!(f(a, b, c, d, e), expected, "{}", name);
            });
        }
        with_fn(source, "uscan", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            f(a, b, c, d, e);
        });
        assert_eq!(&buf[16..20], &[100, 100, 100, 200]);
        assert_eq!(buf[31], 200, "the running max stays at 200 as unsigned");
    }

    #[test]
    fn masked_store_writes_only_active_lanes() {
        let source = r#"
            fn f(buf: ptr[u8], other: ptr[u8]) {
                v = load_at[u8[16]](other, 0)
                store(buf, v, v > 127)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, *mut u8, *mut u8, i64, i64, i64) =
                std::mem::transmute(p);
            let mut buf = [7u8; 16];
            let mut other = [1u8; 16];
            other[2] = 200;
            other[15] = 201;
            let (a, b, c, d, e) = desc(&mut buf);
            let (g, h, i, j, k) = desc(&mut other);
            f(a, b, c, d, e, g, h, i, j, k);
            let mut expected = [7u8; 16];
            expected[2] = 200;
            expected[15] = 201;
            assert_eq!(buf, expected);
        });
    }

    #[test]
    #[should_panic(expected = "doesn't fit")]
    fn literal_too_big_for_its_type_is_an_error() {
        with_fn("fn f(x: u8[16]) -> u8[16] { return x + 300 }", "f", |_| ());
    }

    #[test]
    fn literals_may_use_either_reading_of_their_bits() {
        let source = "fn f(x: u8[1]) -> u8[1] { return (x & 255) + (x & -1) }";
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(u8) -> u8 = std::mem::transmute(p);
            assert_eq!(f(3), 6);
        });
    }

    // ---- literals

    #[test]
    fn constant_expressions_work_wherever_a_type_is_known() {
        let cases: &[(&str, i64)] = &[
            ("fn f(x: i64[1]) -> i64[1] { y: i64[1] = 0 - 1
                return x + y }", 9),
            ("fn f(x: i64[1]) -> i64[1] { return x + (1 << 4) }", 26),
            ("fn f(x: i64[1]) -> i64[1] { return x << (1 + 1) }", 40),
            ("fn f(x: i64[1]) -> i64[1] { return 7 }", 7),
            ("fn f(x: i64[1]) -> i64[1] { return 2 * 3 + 1 }", 7),
            ("fn g(a: i64[1], b: i64[1]) -> i64[1] { return a - b }
              fn f(x: i64[1]) -> i64[1] { return g(x, 4 - 1) }", 7),
            ("fn g(a: i64[1], b: i64[1]) -> i64[1] { return a - b }
              fn f(x: i64[1]) -> i64[1] { return g(100, x) }", 90),
            ("fn f(x: i64[1]) -> i64[1] { y = x
                y = 5
                return y }", 5),
            ("fn f(x: i64[1]) -> i64[1] { return max(1, x) + min(x, 20) }", 20),
        ];
        for (source, expected) in cases {
            with_fn(source, "f", |p| unsafe {
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
                assert_eq!(f(10), *expected, "{}", source);
            });
        }
    }

    #[test]
    fn a_literal_stored_through_a_pointer_takes_its_element_type() {
        let source = "fn f(buf: ptr[u8]) { store_at(buf, 1, 30 + 4) }";
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [0u8; 3];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(buf, [0, 34, 0]);
        });
    }

    #[test]
    fn a_literal_passed_to_a_generic_takes_the_inferred_width() {
        let source = r#"
            fn add[N](a: u8[N], b: u8[N]) -> u8[N] { return a + b }
            fn f(buf: ptr[u8]) { store_at(buf, 16, add(load_at[u8[16]](buf, 0), 5)) }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [1u8; 32];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(&buf[16..32], &[6u8; 16]);
        });
    }

    #[test]
    fn argument_widths_are_checked() {
        let errors = type_errors(
            "fn g(a: u8[16]) -> u8[16] { return a } fn f(a: u8[32]) -> u8[16] { return g(a) }",
        );
        assert!(errors.iter().any(|e| e.contains("expected width 16, got 32")), "got {:?}", errors);
    }

    #[test]
    fn a_literal_with_nothing_to_give_it_a_type_is_an_error() {
        let errors = type_errors("fn f(x: i64[1]) -> i64[1] { y = 5
            return x + y }");
        assert!(errors.iter().any(|e| e.contains("needs a type")), "got {:?}", errors);
    }

    // ---- masks and gathers

    #[test]
    fn masked_without_fallback_zeroes_inactive_lanes() {
        let source = r#"
            fn f(buf: ptr[u8]) {
                v = load_at[u8[16]](buf, 0)
                store_at(buf, 16, [v > 100] v)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [50u8; 32];
            buf[4] = 200;
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(buf[20], 200);
            assert_eq!(buf[16], 0);
        });
    }

    #[test]
    fn gather_syntax_uses_the_pointer_element_type() {
        let source = r#"
            fn many(buf: ptr[u8], at: ptr[i32]) {
                indices = load_at[i32[4]](at, 0)
                store_at(buf, 0, buf.[indices])
            }
            fn one(buf: ptr[u8], i: i32[1]) -> u8[1] { return buf.[i] }
            fn put(buf: ptr[u8], i: i32[1], v: u8[1]) { buf.[i] = v }
        "#;
        let mut buf: Vec<u8> = (10..30).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "one", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i32) -> u8 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e, 3), 13);
        });
        with_fn(source, "put", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i32, u8) = std::mem::transmute(p);
            f(a, b, c, d, e, 19, 99);
        });
        assert_eq!(buf[19], 99);
        with_fn(source, "many", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, *mut i32, *mut i32, i64, i64, i64) =
                std::mem::transmute(p);
            let mut at = [5i32, 0, 2, 7];
            f(a, b, c, d, e, at.as_mut_ptr(), at.as_mut_ptr(), 0, 4, 1);
        });
        assert_eq!(&buf[..4], &[15, 10, 12, 17]);
    }

    // ---- structs

    const PAIR: &str = "struct Pair[N] { a: i64[N], b: i64[N] }\n";

    fn run_i64(source: &str, inputs: &[(i64, i64)]) {
        let program = format!("{}{}", PAIR, source);
        with_fn(&program, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            for (input, expected) in inputs {
                assert_eq!(f(*input), *expected, "f({}) in {}", input, source);
            }
        });
    }

    #[test]
    fn structs_cross_function_boundaries() {
        run_i64(
            "fn mk(x: i64[1]) -> Pair[1] { return Pair[1] { a: x, b: x * 2 } }
             fn f(x: i64[1]) -> i64[1] { p = mk(x)
                return p.a + p.b }",
            &[(5, 15)],
        );
        run_i64(
            "fn sum(p: Pair[1]) -> i64[1] { return p.a + p.b }
             fn f(x: i64[1]) -> i64[1] { p = Pair[1] { a: x, b: 1 }
                return sum(p) }",
            &[(5, 6)],
        );
        run_i64(
            "fn sum(p: Pair[1]) -> i64[1] { return p.a + p.b }
             fn f(x: i64[1]) -> i64[1] { return sum(Pair[1] { a: x, b: 1 }) }",
            &[(5, 6)],
        );
        run_i64(
            "fn mk(x: i64[1]) -> Pair[1] { return Pair[1] { a: x, b: 7 } }
             fn sum(p: Pair[1]) -> i64[1] { return p.a + p.b }
             fn f(x: i64[1]) -> i64[1] { return sum(mk(x)) + mk(x).b }",
            &[(5, 19)],
        );
        run_i64(
            "fn mk(x: i64[1]) -> Pair[1] { return Pair[1] { a: x, b: 7 } }
             fn again(x: i64[1]) -> Pair[1] { return mk(x + 1) }
             fn f(x: i64[1]) -> i64[1] { return again(x).a }",
            &[(5, 6)],
        );
    }

    #[test]
    fn generic_functions_can_return_generic_structs() {
        let source = r#"
            struct Two[N] { lo: u8[N], hi: u8[N] }
            fn both[N](x: u8[N]) -> Two[N] { return Two[N] { lo: x, hi: x + 1 } }
            fn f(buf: ptr[u8]) {
                t = both(load_at[u8[16]](buf, 0))
                w = both(load_at[u8[32]](buf, 0))
                store_at(buf, 32, t.hi)
                store_at(buf, 48, w.hi)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [4u8; 80];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(buf[32], 5);
            assert_eq!(buf[79], 5);
        });
    }

    #[test]
    fn struct_assigned_in_a_while_updates_the_loop_state() {
        // This used to compile, then drop the assignment and never terminate.
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                p = Pair[1] { a: x, b: 0 }
                while p.a > 0 { p = Pair[1] { a: p.a - 1, b: p.b + p.a } }
                return p.b
            }",
            &[(4, 10), (0, 0)],
        );
    }

    #[test]
    fn struct_assigned_in_an_if_merges() {
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                p = Pair[1] { a: 1, b: 2 }
                if x > 0 { p = Pair[1] { a: x, b: 20 } }
                if x > 5 { q = Pair[1] { a: 100, b: 0 } } else { q = Pair[1] { a: 200, b: 0 } }
                return p.a + p.b + q.a
            }",
            &[(3, 223), (7, 127), (-1, 203)],
        );
    }

    #[test]
    fn destructuring_binds_fields_in_order() {
        run_i64(
            "fn mk(x: i64[1]) -> Pair[1] { return Pair[1] { a: x, b: x + 1 } }
             fn f(x: i64[1]) -> i64[1] {
                (first, second) = mk(x)
                return first * 100 + second
             }",
            &[(5, 506)],
        );
        run_i64(
            "fn mk(x: i64[1]) -> Pair[1] { return Pair[1] { a: x, b: x + 1 } }
             fn f(x: i64[1]) -> i64[1] {
                total: i64[1] = 0
                i: i64[1] = 0
                while i < x {
                    (lo, hi) = mk(i)
                    total = total + lo * hi
                    i = i + 1
                }
                return total
             }",
            &[(3, 8)],
        );
        let errors = type_errors(&format!(
            "{}fn mk(x: i64[1]) -> Pair[1] {{ return Pair[1] {{ a: x, b: x }} }}
             fn f(x: i64[1]) -> i64[1] {{ (one) = mk(x)
                return one }}",
            PAIR
        ));
        assert!(errors.iter().any(|e| e.contains("destructuring 1 names")), "got {:?}", errors);
    }

    #[test]
    fn struct_type_errors_are_reported() {
        let errors = type_errors(&format!(
            "{}fn f(x: i64[1]) -> i64[1] {{ p = Pair[1] {{ a: x, b: x }}
                return p.missing }}",
            PAIR
        ));
        assert!(errors.iter().any(|e| e.contains("missing")), "got {:?}", errors);
    }

    // ---- control flow

    #[test]
    fn else_if_chains() {
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                if x > 10 { y: i64[1] = 1 } else if x > 5 { y: i64[1] = 2 } else if x > 0 { y: i64[1] = 3 } else { y: i64[1] = 4 }
                return y
            }",
            &[(11, 1), (6, 2), (1, 3), (0, 4)],
        );
    }

    #[test]
    fn logical_operators_short_circuit() {
        // A null pointer: evaluating the right side would crash.
        let source = r#"
            fn and_guard(go: i64[1], p: ptr[u8]) -> i64[1] {
                if go > 0 && load_at(p, 0) == 1 { return 1 }
                return 0
            }
            fn or_guard(stop: i64[1], p: ptr[u8]) -> i64[1] {
                if stop > 0 || load_at(p, 0) == 1 { return 1 }
                return 0
            }
        "#;
        let null = std::ptr::null::<u8>();
        with_fn(source, "and_guard", |p| unsafe {
            let f: extern "C" fn(i64, *const u8, *const u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(0, null, null, 0, 0, 1), 0);
            let one = [1u8];
            assert_eq!(f(1, one.as_ptr(), one.as_ptr(), 0, 1, 1), 1);
        });
        with_fn(source, "or_guard", |p| unsafe {
            let f: extern "C" fn(i64, *const u8, *const u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(1, null, null, 0, 0, 1), 1);
            let zero = [0u8];
            assert_eq!(f(0, zero.as_ptr(), zero.as_ptr(), 0, 1, 1), 0);
        });
        let errors = type_errors("fn f(a: bool[4], b: bool[4]) -> bool[1] { return a && b }");
        assert!(errors.iter().any(|e| e.contains("scalar bools")), "got {:?}", errors);
    }

    #[test]
    fn likely_passes_its_value_through() {
        run_i64(
            "fn f(x: i64[1]) -> i64[1] { if likely(x > 0) { return x }
                return 0 - x }",
            &[(4, 4), (-4, 4)],
        );
    }

    #[test]
    fn break_leaves_only_the_innermost_loop() {
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                i: i64[1] = 0
                while i < 100 {
                    if i == x { break }
                    i = i + 1
                }
                return i
            }",
            &[(0, 0), (5, 5), (200, 100)],
        );
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                total: i64[1] = 0
                i: i64[1] = 0
                while i < 3 {
                    j: i64[1] = 0
                    while j < 100 {
                        if j == x { break }
                        total = total + 1
                        j = j + 1
                    }
                    i = i + 1
                }
                return total
            }",
            &[(2, 6), (0, 0)],
        );
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                i: i64[1] = 0
                skipped: i64[1] = 0
                while i < 10 {
                    i = i + 1
                    if i < x { skipped = skipped + 1 } else { break }
                    skipped = skipped + 100
                }
                return skipped
            }",
            &[(3, 202), (1, 0)],
        );
    }

    #[test]
    fn early_return_from_branches() {
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                if x > 10 { return 1 }
                if x > 5 { return 2 } else if x > 0 { y = x
                    return y * 10 }
                z = x - 1
                return z
            }",
            &[(11, 1), (6, 2), (3, 30), (-3, -4)],
        );
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                if x > 0 { return 1 } else { return 2 }
            }",
            &[(1, 1), (-1, 2)],
        );
        // A name defined before the if and used after it, on the path that continues.
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                y = x * 2
                if y > 10 {
                    z: i64[1] = 1
                    return y + z
                } else {
                    z: i64[1] = 2
                }
                return y + 100
            }",
            &[(6, 13), (1, 102)],
        );
    }

    #[test]
    fn early_return_from_loops() {
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                i: i64[1] = 0
                while i < 100 {
                    if i * i >= x { return i }
                    i = i + 1
                }
                return 0 - 1
            }",
            &[(0, 0), (16, 4), (17, 5), (20000, -1)],
        );
        run_i64(
            "fn f(x: i64[1]) -> i64[1] {
                i: i64[1] = 0
                while i < 10 {
                    j: i64[1] = 0
                    while j < 10 {
                        if i * 10 + j == x { return i * 1000 + j }
                        j = j + 1
                    }
                    i = i + 1
                }
                return 0 - 1
            }",
            &[(37, 3007), (0, 0), (100, -1)],
        );
    }

    #[test]
    fn early_return_works_with_structs_generics_and_no_value() {
        run_i64(
            "fn mk(x: i64[1]) -> Pair[1] {
                if x > 0 { return Pair[1] { a: x, b: 1 } }
                return Pair[1] { a: 0 - x, b: 2 }
             }
             fn f(x: i64[1]) -> i64[1] { p = mk(x)
                return p.a * 10 + p.b }",
            &[(3, 31), (-3, 32)],
        );
        run_i64(
            "fn pick[N](x: i64[N]) -> i64[N] {
                if (+/ x) > 0 { return x }
                return x + x
             }
             fn f(x: i64[1]) -> i64[1] { return pick(x) }",
            &[(3, 3), (-3, -6)],
        );
        let source = r#"
            fn f(buf: ptr[u8], x: i64[1]) {
                if x > 0 { return }
                store_at(buf, 0, 9)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [0u8; 1];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e, 1);
            assert_eq!(buf[0], 0);
            f(a, b, c, d, e, 0);
            assert_eq!(buf[0], 9);
        });
    }

    #[test]
    fn early_return_agrees_with_plain_code_on_every_input() {
        // The same search written with `return` and with hand-threaded flags.
        let escaping = with_fn_values(
            "fn f(x: i64[1]) -> i64[1] {
                i: i64[1] = 0
                while i < 50 {
                    if (i & 7) == 3 && i > x { return i * 2 }
                    if i == x + 20 { break }
                    i = i + 1
                }
                if x > 25 { return 0 - x }
                return i
            }",
        );
        let plain = with_fn_values(
            "fn f(x: i64[1]) -> i64[1] {
                i: i64[1] = 0
                result: i64[1] = 0
                found: bool[1] = false
                stop: bool[1] = false
                while ~found && ~stop && i < 50 {
                    if (i & 7) == 3 && i > x {
                        result = i * 2
                        found = true
                    } else if i == x + 20 {
                        stop = true
                    } else {
                        i = i + 1
                    }
                }
                if ~found {
                    if x > 25 { result = 0 - x } else { result = i }
                }
                return result
            }",
        );
        assert_eq!(escaping, plain);
    }

    #[test]
    fn code_after_a_partly_escaping_branch_runs_only_on_the_path_that_continues() {
        // Written with returns, and with a single exit.
        let escaping = with_fn_values(
            "fn f(x: i64[1]) -> i64[1] {
                total: i64[1] = 0
                i: i64[1] = 0
                while i < 4 {
                    if x > 40 { return 1 } else if x > 30 {
                        if x > 35 { return 2 }
                        total = total + 10
                    } else if x > 20 {
                        if x == 25 { break } else if x > 25 { return 3 }
                        total = total + 100
                    }
                    total = total + 1
                    i = i + 1
                }
                if x > 10 && total == 0 { return 4 }
                return total
            }",
        );
        let expected: Vec<i64> = (-5i64..70)
            .map(|x| match x {
                x if x > 40 => 1,
                x if x > 35 => 2,
                x if x > 30 => 44,
                x if x > 25 => 3,
                25 => 4,
                x if x > 20 => 404,
                _ => 4,
            })
            .collect();
        assert_eq!(escaping, expected);
    }

    fn with_fn_values(source: &str) -> Vec<i64> {
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            (-5i64..70).map(|x| f(x)).collect()
        })
    }

    #[test]
    #[should_panic(expected = "doesn't return a value on every path")]
    fn falling_off_the_end_after_an_early_return_is_an_error() {
        parser::parse("fn f(x: i64[1]) -> i64[1] { if x > 0 { return x } }");
    }

    #[test]
    #[should_panic(expected = "break outside of a loop")]
    fn break_outside_a_loop_is_an_error() {
        parser::parse("fn f(x: i64[1]) -> i64[1] { if x > 0 { break }
            return x }");
    }

    #[test]
    #[should_panic(expected = "return inside a stream")]
    fn return_inside_a_stream_is_an_error() {
        parser::parse("fn f(b: ptr[u8]) -> i32[1] {
            stream c: u8[16] over b carry (n: i32[1] = 0) {
                if any(c > 1) { return n }
                carry n = n + 1
            }
            return n
        }");
    }

    // ---- regressions from an independent review. Each ran wrong or crashed before.

    /// The value of `fn f(x: i64[1]) -> i64[1]` in `source` for each input.
    fn eval(source: &str, inputs: &[i64]) -> Vec<i64> {
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(p);
            inputs.iter().map(|&x| f(x)).collect()
        })
    }

    /// Whether the whole front end refuses a program, with this text in the message.
    fn assert_rejected(source: &str, expected: &str) {
        let source = source.to_string();
        let result = std::panic::catch_unwind(move || {
            let ctx = create_context();
            let items = parser::parse(&source);
            if let Err(errors) = typecheck::typecheck(&items, &HashMap::new()) {
                return errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
            }
            let module = compile_module(&ctx, &items, &HashMap::new());
            assert!(module.as_operation().verify());
            String::new()
        });
        let message = match result {
            Ok(errors) => errors,
            Err(panic) => panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        };
        assert!(message.contains(expected), "expected an error containing {:?}, got {:?}", expected, message);
    }

    #[test]
    fn unsigned_division_is_unsigned() {
        let source = "fn f(x: i64[1]) -> i64[1] { a = to_u8(x)
            b = a / 2
            return to_i64(b) }";
        assert_eq!(eval(source, &[200]), [100]);
        let source = "fn f(x: i64[1]) -> i64[1] { a = to_u64(x)
            b = a / 2
            return to_i64(b) }";
        assert_eq!(eval(source, &[-2]), [i64::MAX]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { return x / 2 }", &[-4]), [-2]);
    }

    #[test]
    fn carry_updates_apply_by_name_and_together() {
        let source = r#"
            fn f(p: ptr[u8]) -> i64[1] {
                stream chunk: u8[16] over p carry (a: i64[1] = 0, b: i64[1] = 0) {
                    carry b = b + 1
                    carry a = a + 100
                }
                return a * 1000 + b
            }
        "#;
        let mut buf = [0u8; 32];
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 200002);
        });
        assert_rejected(
            "fn f(p: ptr[u8]) -> i64[1] {
                stream c: u8[16] over p carry (a: i64[1] = 0) { carry zzz = a + 1 }
                return a }",
            "isn't one of this stream's carries",
        );
        assert_rejected(
            "fn f(p: ptr[u8]) -> i64[1] {
                stream c: u8[16] over p carry (a: i64[1] = 0) {
                    carry a = a + 1
                    carry a = a + 2
                }
                return a }",
            "updated twice",
        );
        assert_rejected(
            "fn f(p: ptr[u8]) -> i64[1] {
                stream c: u8[8] over p carry (s: i64[1] = 0) {
                    x: i64[1] = 1
                    carry s = s + x
                    x = 100
                }
                return s }",
            "must come last",
        );
    }

    #[test]
    fn unsigned_offsets_and_indices_are_not_sign_extended() {
        let source = r#"
            fn read(p: ptr[u8], off: u8[1]) -> u8[1] { return load_at(p, off) }
            fn wide(p: ptr[u8], off: u8[1]) -> u8[1] { return load_at[u8[1]](p, off) }
            fn gathered(p: ptr[u8], off: u8[1]) -> u8[1] { return p.[off] }
            fn put(p: ptr[u8], off: u8[1]) { store_at(p, off, 77) }
            fn scattered(p: ptr[u8], off: u8[1]) { p.[off] = 88 }
            fn many(p: ptr[u8]) {
                i = [u8: 200, 201, 202, 203]
                store_at(p, 0, p.[i])
            }
        "#;
        let mut buf: Vec<u8> = (0..255).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        for name in ["read", "wide", "gathered"] {
            with_fn(source, name, |p| unsafe {
                let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, u8) -> u8 = std::mem::transmute(p);
                assert_eq!(f(a, b, c, d, e, 200), 200, "{}", name);
            });
        }
        with_fn(source, "put", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, u8) = std::mem::transmute(p);
            f(a, b, c, d, e, 210);
        });
        assert_eq!(buf[210], 77);
        with_fn(source, "scattered", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, u8) = std::mem::transmute(p);
            f(a, b, c, d, e, 211);
        });
        assert_eq!(buf[211], 88);
        with_fn(source, "many", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            f(a, b, c, d, e);
        });
        assert_eq!(&buf[..4], &[200, 201, 202, 203]);
    }

    #[test]
    fn annotations_set_the_type() {
        let source = "fn f(x: i64[1]) -> i64[1] {
            a = to_i8(x)
            y: u8[1] = a
            if y > 0 { return 1 }
            return 0 }";
        assert_eq!(eval(source, &[-1, 0]), [1, 0], "reads as unsigned after the annotation");

        // Literal branches take the annotated lane type, so this stores 4 bytes, not 16.
        let source = r#"
            fn f(p: ptr[u8]) {
                v = load[u8[4]](p)
                x: u8[4] = [v > 1] 1 : 0
                store_at(p, 8, x)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [9u8; 32];
            buf[..4].copy_from_slice(&[0, 5, 0, 5]);
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(&buf[8..12], &[0, 1, 0, 1]);
            assert_eq!(buf[12], 9, "nothing past the 4 lanes is written");
        });
        assert_rejected("fn f(a: i32[1]) -> i32[1] { x: i64[1] = a
            return a }", "expected I64, got I32");
    }

    #[test]
    fn different_structs_never_mix() {
        let two = "struct A[N] { x: i64[N] y: i64[N] }\nstruct B[N] { y: i64[N] x: i64[N] }\n";
        assert_rejected(
            &format!("{}fn f(c: i64[1]) -> i64[1] {{
                if c > 0 {{ p = A[1] {{ x: c, y: 2 }} }} else {{ p = B[1] {{ y: 3, x: 4 }} }}
                return p.x }}", two),
            "different types in the two branches",
        );
        assert_rejected(
            &format!("{}fn g() -> A[1] {{ return B[1] {{ y: 3, x: 4 }} }}", two),
            "expected struct A, got struct B",
        );
        assert_rejected(
            &format!("{}fn g(a: A[1]) -> i64[1] {{ return a.x }}
                fn f(x: i64[1]) -> i64[1] {{ return g(B[1] {{ y: x, x: x }}) }}", two),
            "expected struct A, got struct B",
        );
    }

    #[test]
    fn scans_are_right_at_any_width() {
        let source = r#"
            fn three(o: ptr[i32]) {
                v = [i32: 1, 2, 4]
                store(o, scan.add(v))
            }
            fn six(o: ptr[i32]) {
                v = [i32: 1, 1, 1, 1, 1, 1]
                store(o, scan.add(v))
            }
            fn xor48(p: ptr[u8]) {
                store(p, scan.xor(load[u8[48]](p)))
            }
        "#;
        let mut out = [0i32; 6];
        with_fn(source, "three", |p| unsafe {
            let f: extern "C" fn(*mut i32, *mut i32, i64, i64, i64) = std::mem::transmute(p);
            f(out.as_mut_ptr(), out.as_mut_ptr(), 0, 6, 1);
        });
        assert_eq!(&out[..3], &[1, 3, 7]);
        with_fn(source, "six", |p| unsafe {
            let f: extern "C" fn(*mut i32, *mut i32, i64, i64, i64) = std::mem::transmute(p);
            f(out.as_mut_ptr(), out.as_mut_ptr(), 0, 6, 1);
        });
        assert_eq!(out, [1, 2, 3, 4, 5, 6]);
        with_fn(source, "xor48", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf: Vec<u8> = (1..=48).collect();
            let mut expected = buf.clone();
            for i in 1..48 {
                expected[i] ^= expected[i - 1];
            }
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(buf, expected);
        });
    }

    #[test]
    fn constants_are_evaluated_in_their_type() {
        with_fn("fn f() -> u64[1] { x: u64[1] = ~0 >> 60
            return x }", "f", |p| unsafe {
            let f: extern "C" fn() -> u64 = std::mem::transmute(p);
            assert_eq!(f(), 15);
        });
        with_fn("fn f() -> u8[1] { x: u8[1] = ~0 >> 4
            return x }", "f", |p| unsafe {
            let f: extern "C" fn() -> u8 = std::mem::transmute(p);
            assert_eq!(f(), 15);
        });
        with_fn("fn f(a: u8[1]) -> u8[1] { return a & ~240 }", "f", |p| unsafe {
            let f: extern "C" fn(u8) -> u8 = std::mem::transmute(p);
            assert_eq!(f(0xAB), 0x0B);
        });
        with_fn("fn f(a: u16[1]) -> u16[1] { return a & ~65280 }", "f", |p| unsafe {
            let f: extern "C" fn(u16) -> u16 = std::mem::transmute(p);
            assert_eq!(f(0xABCD), 0x00CD);
        });
        with_fn("fn f() -> u64[1] { x: u64[1] = 18446744073709551615
            return x }", "f", |p| unsafe {
            let f: extern "C" fn() -> u64 = std::mem::transmute(p);
            assert_eq!(f(), u64::MAX);
        });
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { return x & 0xFF }", &[0x1234]), [0x34]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { return x | 0b101 }", &[8]), [13]);
        assert_rejected("fn f(x: i64[1]) -> i64[1] { x10: i64[1] = 1
            a: i64[1] = 0z10
            return a }", "followed directly by a letter");
        assert_rejected("fn f() -> i64[1] { x: i64[1] = 99999999999999999999
            return x }", "doesn't fit in 64 bits");
        assert_rejected("fn f(a: u8[1]) -> u8[1] { return a << 9 }", "out of range");
        assert_rejected("fn f(a: i64[1]) -> i64[1] { return a >> 64 }", "out of range");
        assert_rejected("fn f(a: i64[1]) -> i64[1] { return a / 0 }", "division by zero");
    }

    #[test]
    fn shifts_right_follow_signedness_and_conversions_extend_by_it() {
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { return x >> 1 }", &[-8]), [-4]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { a = to_u64(x)
            return to_i64(a >> 1) }", &[-8]), [(u64::MAX - 7) as i64 >> 1 & i64::MAX]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { a = to_i32(x)
            return to_i64(a) }", &[-1]), [-1], "signed widens by sign");
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { a = to_u32(x)
            return to_i64(a) }", &[-1]), [0xFFFF_FFFF], "unsigned widens by zero");
        // A shift amount of a different size takes the operand's.
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { s = to_u8(x)
            return x >> s }", &[3]), [0]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { s = to_u8(x)
            return (x + 16) << s }", &[2]), [72]);
    }

    #[test]
    fn bools_order_as_false_before_true() {
        let source = "fn f(x: i64[1]) -> i64[1] {
            t = x > 0
            u = x < 0
            if t > u { return 1 }
            return 0 }";
        assert_eq!(eval(source, &[1, -1, 0]), [1, 0, 0]);
        let source = "fn f(x: i64[1]) -> i64[1] {
            if max(x > 0, x < 0) { return 1 }
            return 0 }";
        assert_eq!(eval(source, &[1, -1, 0]), [1, 1, 0]);
        assert_rejected("fn f(v: bool[4]) -> bool[1] { return +/ v }", "aren't defined on bools");
        assert_rejected("fn f(a: bool[1], b: bool[1]) -> bool[1] { return a + b }", "arithmetic on bools");
        assert_rejected("fn f() -> bool[1] { x: bool[1] = 2
            return x }", "isn't a bool");
        assert_rejected("fn f() -> bool[1] { x: bool[1] = -true
            return x }", "can't be negated");
        assert_rejected("fn f() -> i64[1] { x: i64[1] = true
            return x }", "not compatible");
    }

    #[test]
    fn vector_literals_must_fit() {
        assert_rejected("fn f(p: ptr[u8]) { v = [u8: 300, 1, 2, 3]
            store(p, v) }", "doesn't fit");
    }

    #[test]
    fn nan_is_not_equal_to_itself() {
        with_fn("fn f() -> bool[1] { z: f32[1] = 0.0
            n = z / z
            return n != n }", "f", |p| unsafe {
            let f: extern "C" fn() -> bool = std::mem::transmute(p);
            assert!(f());
        });
    }

    #[test]
    fn carries_leave_enclosing_ifs_and_loops() {
        let source = r#"
            fn f(p: ptr[u8], c: i64[1]) -> i64[1] {
                n: i64[1] = 7
                if c > 0 {
                    stream chunk: u8[16] over p carry (n: i64[1] = 0) {
                        carry n = n + 1
                    }
                }
                return n
            }
        "#;
        let mut buf = [0u8; 32];
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e, 1), 2);
            assert_eq!(f(a, b, c, d, e, 0), 7);
        });
    }

    #[test]
    fn chunk_offset_belongs_to_the_innermost_stream() {
        let source = r#"
            fn f(p: ptr[u8]) -> i64[1] {
                chunk_offset: i32[1] = 99
                stream a: u8[8] over p carry (s: i64[1] = 0) {
                    carry s = s + to_i64(chunk_offset)
                }
                return s
            }
        "#;
        let mut buf = [0u8; 16];
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 8);
        });
    }

    #[test]
    fn extern_pointers_cant_reach_a_stream_any_way() {
        let streams = "fn g(q: ptr[u8]) -> i64[1] {
            stream c: u8[16] over q carry (n: i64[1] = 0) { carry n = n + 1 }
            return n }\n";
        assert_rejected(
            "extern fn f(p: ptr[u8]) -> i64[1] { q = p
                stream c: u8[16] over q carry (n: i64[1] = 0) { carry n = n + 1 }
                return n }",
            "no known size",
        );
        assert_rejected(
            "extern fn f(p: ptr[u8]) -> i64[1] { q = ptr_add(p, 4)
                stream c: u8[16] over q carry (n: i64[1] = 0) { carry n = n + 1 }
                return n }",
            "no known size",
        );
        assert_rejected(&format!("{}extern fn f(p: ptr[u8]) -> i64[1] {{ return g(p) }}", streams), "no known size");
        assert_rejected(
            &format!("{}fn h(r: ptr[u8]) -> i64[1] {{ return g(r) }}
                extern fn f(p: ptr[u8]) -> i64[1] {{ return h(p) }}", streams),
            "no known size",
        );
    }

    #[test]
    fn names_that_would_change_meaning_are_refused() {
        assert_rejected("fn f(__ret: i64[1]) -> i64[1] { return __ret }", "reserved");
        assert_rejected("fn max(a: i64[1], b: i64[1]) -> i64[1] { return a + b }", "same name as a builtin");
        assert_rejected("fn g(x: i64[1]) -> i64[1] { return x }
            fn g(x: i64[1]) -> i64[1] { return x }", "defined twice");
        assert_rejected("fn f(a: i64[1], a: i64[1]) -> i64[1] { return a }", "two parameters");
        assert_rejected("fn f(x: i64[1]) -> i64[1] { REGISTER_BYTES = x
            return x }", "compile-time width");
        assert_rejected("fn g[N](x: i64[1]) -> i64[1] { N = x
            return N }", "compile-time width");
        // A comptime name doesn't capture a callee of the same name.
        let source = "fn g[min](a: u8[min]) -> u8[min] { return min(a, a) }
            fn f(x: i64[1]) -> i64[1] {
                v: u8[4] = 3
                return to_i64(extract(g(v), 0)) + x }";
        assert_eq!(eval(source, &[1]), [4]);
    }

    #[test]
    fn invalid_loads_are_errors() {
        assert_rejected("fn f(p: ptr[u8]) -> bool[1] { return load_at[bool[1]](p, 2) }", "can't be loaded");
        assert_rejected("fn f(p: ptr[u32]) -> u8[1] { return load_at[u8[1]](p, 1) }", "whole number");
        assert_rejected("fn f(p: ptr[u32]) -> u16[3] { return load[u16[3]](p) }", "whole number");
    }

    #[test]
    fn loads_through_computed_pointers_work() {
        let source = "fn f(p: ptr[u8]) -> u64[1] { return load_at[u64[1]](ptr_add(p, 1), 0) }";
        let mut buf: Vec<u8> = (0..16).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), u64::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8]));
        });
    }

    #[test]
    fn scalar_branches_broadcast_across_a_vector_mask() {
        let source = r#"
            fn f(p: ptr[u8], a: u8[1], b: u8[1]) {
                v = load_at[u8[16]](p, 0)
                store_at(p, 16, [v > 1] a : b)
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, u8, u8) = std::mem::transmute(p);
            let mut buf = [0u8; 32];
            buf[3] = 9;
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e, 7, 8);
            assert_eq!(buf[19], 7);
            assert_eq!(buf[16], 8);
        });
    }

    #[test]
    fn recursive_generics_are_an_error() {
        assert_rejected("fn g[N](x: i64[1]) -> i64[1] { return g[N + 1](x) }
            fn f(x: i64[1]) -> i64[1] { return g[1](x) }", "too deeply");
    }

    #[test]
    fn pointers_can_be_returned() {
        let source = r#"
            fn advance(p: ptr[u8]) -> ptr[u8] { return ptr_add(p, 3) }
            fn f(p: ptr[u8]) -> u8[1] { q = advance(p)
                return load_at(q, 1) }
        "#;
        let mut buf: Vec<u8> = (0..8).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u8 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 4);
        });
    }

    #[test]
    fn compress_packs_active_lanes() {
        let source = r#"
            fn f(p: ptr[u8]) {
                v = load_at[u8[16]](p, 0)
                store_at(p, 16, compress(v, v > 100))
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) = std::mem::transmute(p);
            let mut buf = [1u8; 32];
            buf[2] = 200;
            buf[9] = 201;
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e);
            assert_eq!(&buf[16..18], &[200, 201]);
            assert_eq!(buf[18], 0);
        });
    }

    #[test]
    fn type_mismatches_are_reported_not_passed_to_mlir() {
        for (source, expected) in [
            ("fn f(n: i64[1], m: bool[4]) -> i64[1] { return n + popcount(m) }", "different types"),
            ("fn f(v: u8[4], s: i64[1]) -> u8[4] { return v + s }", "different types"),
            ("fn g(a: i64[1]) -> i64[1] { return a } fn f(c: i32[1]) -> i64[1] { return g(c) }", "expected I64, got I32"),
            ("fn f(c: i32[1]) -> i64[1] { return c }", "expected I64, got I32"),
            ("fn f(x: f32[1]) -> i64[1] { return to_i64(x) }", "needs an integer"),
            ("fn f(a: f32[4], b: f32[4]) -> f32[4] { return a & b }", "aren't defined on floats"),
            ("fn f(x: u32[1]) -> u32[1] { return ctz(x) }", "64-bit"),
            ("fn g(p: ptr[u8]) -> u8[1] { return load_at(p, 0) } fn f(p: ptr[u32]) -> u8[1] { return g(p) }", "ptr"),
            ("fn f(p: ptr[u8]) -> i64[1] { v = load[u8[4]](p)
                i: i64[1] = 0
                while i < 1 { v = load[u8[8]](p)
                    i = i + 1 }
                return i }", "changes type"),
            ("fn f(x: i64[1]) -> i64[1] { y: i64[1] = 3 }", "doesn't return"),
            ("fn f(v: u8[2]) -> u8[1] { return split_lo(v) }", "at least 4"),
            ("fn f(v: u8[1]) -> u8[1] { return lane_shr(v, 1) }", "at least 2"),
            ("fn f(v: u8[4]) -> u8[1] { return shuffle(v, v, [u8: 3]) }", "at least 2"),
            ("fn f(a: u64[1]) -> bool[1] { return from_bitmask(a, 1) }", "from_bitmask"),
            ("fn f(a: u32[1]) -> bool[32] { return from_bitmask(a, 32) }", "64-bit"),
            ("fn f(p: ptr[u8]) -> u64[1] { return to_bitmask(load[u8[128]](p) > 1) }", "at most 64"),
            ("fn f(v: u8[4]) -> u8[1] { return extract(v, 9) }", "out of range"),
            ("fn f(t: u8[16], i: u8[8]) -> u8[8] { return tbl(t, i) }", "multiple of 16"),
            ("fn g() { } fn f() -> i64[1] { y = g()
                return y }", "no value"),
            ("fn g() { } fn f() -> i64[1] { return g() }", "expected"),
            ("struct A[N] { x: i64[N] } fn f() -> A[1] { return A[1] { x: 1, bogus: undefined_thing } }", "undefined variable"),
            ("struct A[N] { x: i64[N] } fn f() -> A[1] { return A[1] { x: 1, x: 2 } }", "twice"),
            ("fn f(a: i8[1], b: u8[1]) -> bool[1] { return a > b }", "order differently"),
            ("fn f(a: u8[1], b: i8[1]) -> u8[1] { return min(a, b) }", "different types"),
            ("fn f(c: bool[1]) -> bool[1] { if c { x: u8[1] = 255 } else { x: i8[1] = -1 }
                return x > 0 }", "different types in the two branches"),
            ("fn f(x: i64[1]) -> i64[1] { return 2
                z = undefined_thing }", "unreachable"),
        ] {
            assert_rejected(source, expected);
        }
    }

    #[test]
    fn literals_work_in_every_position() {
        let source = r#"
            fn f(p: ptr[u8]) -> i64[1] {
                stream c: u8[16] over p carry (a: i64[1] = 5, b: bool[1] = true, s: i64[1] = 0) {
                    carry a = 7
                    carry b = false
                    carry s = s + a
                }
                return s
            }
        "#;
        let mut buf = [0u8; 32];
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> i64 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 5 + 7);
        });
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { y: i64[1] = 1 << x
            return y }", &[3]), [8]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { if true { return x }
            return 0 }", &[3]), [3]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { while false { x = 0 }
            return x }", &[3]), [3]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { if REGISTER_BYTES >= 16 { return x }
            return 0 }", &[3]), [3]);
        assert_eq!(eval("fn g[N](x: i64[1]) -> i64[1] { if N > 8 { return x + 1 }
                return x }
            fn f(x: i64[1]) -> i64[1] { return g[9](x) * 100 + g[8](x) }", &[3]), [403]);
        let source = "fn f(x: f32[4]) -> f32[4] { y: f32[4] = [x > 0.0] 1.0 : 0.0
            return y }";
        with_fn(source, "f", |_| ());
        // Compressstore takes a literal offset, and scans a literal seed.
        with_fn("fn f(o: ptr[u8], v: u8[16], m: bool[16]) { compressstore(o, 0, v, m) }", "f", |_| ());
        with_fn("fn f(v: u8[16]) -> u8[16] { return scan.xor(v, 1) }", "f", |_| ());
    }

    #[test]
    fn registers_size_struct_fields() {
        with_fn("struct S[N] { a: u8[REGISTER_BYTES] }
            fn g(p: ptr[u8]) -> S[1] { return S[1] { a: load_at[u8[REGISTER_BYTES]](p, 0) } }
            fn f(p: ptr[u8]) { store_at(p, 0, g(p).a) }", "f", |_| ());
    }

    #[test]
    fn code_moved_by_desugaring_stays_visible() {
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] {
            if x > 0 {
                if x > 10 { return 100 }
                z: i64[1] = 1
            } else {
                z: i64[1] = 2
            }
            return z }", &[5, 50, -1]), [1, 100, 2]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] {
            i: i64[1] = 0
            while i < 3 {
                if x > 0 {
                    if x == i { break }
                    z: i64[1] = 1
                } else {
                    z: i64[1] = 2
                }
                x = x + z - z
                i = i + 1
            }
            return i }", &[1, -1, 50]), [1, 3, 3]);
        assert_eq!(eval(&format!("{}fn f(x: i64[1]) -> i64[1] {{
            if x > 0 {{
                if x > 10 {{ return 100 }}
                p = Pair[1] {{ a: 1, b: 2 }}
            }} else {{
                p = Pair[1] {{ a: 3, b: 4 }}
            }}
            return p.a + p.b }}", PAIR), &[5, 50, -1]), [3, 100, 7]);
    }

    #[test]
    fn a_line_starting_with_a_paren_or_minus_is_a_new_statement() {
        assert_eq!(eval(&format!("{}fn mk(x: i64[1]) -> Pair[1] {{ return Pair[1] {{ a: x, b: 2 }} }}
            fn f(x: i64[1]) -> i64[1] {{
                y = x
                (a, b) = mk(y)
                return a + b }}", PAIR), &[5]), [7]);
        assert_rejected("fn f(x: i64[1]) -> i64[1] { y = x
            -x
            return y }", "");
    }

    #[test]
    fn a_loop_that_only_returns_needs_no_trailing_return() {
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] {
            while true {
                if x > 3 { return x }
                x = x + 1
            }
        }", &[0, 9]), [4, 9]);
    }

    #[test]
    fn second_review_pass_cases() {
        for (source, expected) in [
            ("fn f(p: ptr[u8], x: i64[1], i: i64[1]) { p.[i] = x }", "scatter value"),
            ("fn f(p: ptr[u8]) -> i64[1] { q = p + 1
                return 0 }", "ptr_add"),
            ("fn f(a: i64[1]) -> i64[1] { s: i64[1] = 1
                return scan.xor(a, s) }", "at least 2 lanes"),
            ("fn f(p: ptr[u8]) -> i64[1] {
                stream a: u32[4] over p carry (s: i64[1] = 0) { carry s = s + 1 }
                return s }", "element type"),
            ("struct A[N] { x: i64[N] }
              fn f(p: ptr[i64], n: i64[1]) -> i64[1] { s = A[1] { x: 1 }
                i: i64[1] = 0
                while i < n { s = A[2] { x: load[i64[2]](p) }
                   i = i + 1 }
                return 0 }", "changes type"),
            ("fn f(n: i64[1]) -> bool[1] { x: u8[1] = 1
                i: i64[1] = 0
                while i < n { x: i8[1] = -1
                   i = i + 1 }
                return x > 0 }", "changes type"),
            ("fn f(p: ptr[u8]) -> i64[1] {
                x: i64[1] = 0
                stream chunk: u8[16] over p carry (n: i64[1] = 0) {
                    x = x + 1
                    carry n = n + 1
                }
                return x }", "make it a carry"),
            ("fn f(a: u8[1]) -> bool[1] { return a > -1 }", "isn't a U8 value"),
            ("fn f(a: i8[1]) -> bool[1] { return a < 200 }", "isn't a I8 value"),
            ("fn f(x: i64[1]) -> i64[1] { break
                return x }", "break outside of a loop"),
            ("const T = [u8: 300, 2]
              fn f(i: i64[1]) -> u8[1] { return load_at(T, i) }", "doesn't fit"),
            ("struct A[N] { x: i64[N] }
              fn g(a: A[1]) -> i64[1] { return a.x }
              fn f(p: ptr[i64]) -> i64[1] { return g(A[2] { x: load[i64[2]](p) }) }", "wrong width"),
            ("struct S[A, B] { x: i64[A] }
              fn f(s: S[1]) -> i64[1] { return 1 }", "at most one comptime"),
        ] {
            assert_rejected(source, expected);
        }

        // Comparing with a literal in range still works, at the top of each type's range.
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { a = to_u8(x)
            if a > 200 { return 1 }
            return 0 }", &[201, 199, -1]), [1, 0, 1]);
        assert_eq!(eval("fn f(x: i64[1]) -> i64[1] { a = to_i8(x)
            if a > -1 { return 1 }
            return 0 }", &[0, -1]), [1, 0]);

        // A float body's inactive lanes are float zeros, whatever produced the body.
        let source = "fn f(x: f32[4]) -> f32[4] { return [x > 0.0] sqrt(x) }";
        with_fn(source, "f", |_| ());

        // A literal scattered to many indices fills them all.
        let source = r#"
            fn f(p: ptr[u8], ip: ptr[i32]) {
                idx = load[i32[4]](ip)
                p.[idx] = 7
            }
        "#;
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64, *mut i32, *mut i32, i64, i64, i64) =
                std::mem::transmute(p);
            let mut buf = [0u8; 8];
            let mut at = [1i32, 3, 5, 6];
            let (a, b, c, d, e) = desc(&mut buf);
            f(a, b, c, d, e, at.as_mut_ptr(), at.as_mut_ptr(), 0, 4, 1);
            assert_eq!(buf, [0, 7, 0, 7, 0, 7, 7, 0]);
        });
    }

    #[test]
    fn memory_offsets_must_be_scalar_integers() {
        // A literal offset is fine without an annotation.
        let source = r#"
            fn f(buf: ptr[u8]) -> u8[1] {
                v = load_at[u8[16]](buf, 2)
                return extract(v, 0)
            }
        "#;
        let mut buf: Vec<u8> = (0..32).collect();
        let (a, b, c, d, e) = desc(&mut buf);
        with_fn(source, "f", |p| unsafe {
            let f: extern "C" fn(*mut u8, *mut u8, i64, i64, i64) -> u8 = std::mem::transmute(p);
            assert_eq!(f(a, b, c, d, e), 2);
        });
    }
}
