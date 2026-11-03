use std::collections::HashMap;

use crate::ast;
use crate::codegen::{int_conversion_target, resolve_type, Arch, ScalarType, VecType};

// ─── Error types ───

#[derive(Debug)]
pub struct TypeError {
    pub span: ast::Span,
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error at {}: {}", self.span, self.message)
    }
}

// ─── Checked type ───

#[derive(Debug, Clone, PartialEq)]
enum CheckedType {
    Vec(VecType),
    Struct {
        name: String,
        fields: Vec<(String, VecType)>,
    },
    Ptr(ScalarType),
    /// Unresolved literal — gets resolved when combined with a concrete Vec.
    Literal(LitKind),
    Void,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LitKind {
    Int,
    Float,
    Bool,
    Char,
}

impl CheckedType {
    fn describe(&self) -> String {
        match self {
            CheckedType::Vec(vt) => format!("{:?}[{}]", vt.scalar, vt.width),
            CheckedType::Struct { name, .. } => format!("struct {}", name),
            CheckedType::Ptr(s) => format!("ptr[{:?}]", s),
            CheckedType::Literal(k) => match k {
                LitKind::Int => "int literal".into(),
                LitKind::Float => "float literal".into(),
                LitKind::Bool => "bool literal".into(),
                LitKind::Char => "char literal".into(),
            },
            CheckedType::Void => "void".into(),
        }
    }
}

// ─── Helpers ───

fn is_scalar_type_name(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool"
    )
}

fn is_ptr_type_name(name: &str) -> bool {
    name == "ptr"
}

fn is_literal(expr: &ast::Expr) -> bool {
    match expr {
        ast::Expr::FloatLit(_)
        | ast::Expr::IntLit(_)
        | ast::Expr::BoolLit(_)
        | ast::Expr::CharLit(_) => true,
        ast::Expr::UnaryOp {
            op: ast::UnaryOp::Not,
            operand,
        }
        | ast::Expr::UnaryOp {
            op: ast::UnaryOp::Neg,
            operand,
        } => is_literal(operand),
        ast::Expr::BinOp { .. } | ast::Expr::Logical { .. } => expr.const_value().is_some(),
        _ => false,
    }
}

/// Whether an expression has no type of its own and takes one from where it's used.
fn takes_expected_type(expr: &ast::Expr) -> bool {
    match expr {
        _ if is_literal(expr) => true,
        ast::Expr::Masked { body, fallback, .. } => {
            is_literal(body) && fallback.as_deref().is_none_or(is_literal)
        }
        ast::Expr::BinOp { op: ast::BinOp::BitShl | ast::BinOp::BitShr, lhs, .. } => is_literal(lhs),
        _ => false,
    }
}

/// Whether two lane types are the same size and kind, so one can be reinterpreted as the
/// other. Signedness aside, that makes them the same machine type.
fn same_machine_type(a: ScalarType, b: ScalarType) -> bool {
    a.bit_width() == b.bit_width() && a.is_float() == b.is_float()
}

/// The ptr aliases of each function that reach a `stream`, which needs to know its size.
/// A name is an alias if assigned from another alias or `ptr_add` of one.
fn params_needing_size(items: &[ast::Item]) -> HashMap<String, Vec<bool>> {
    fn root<'e>(expr: &'e ast::Expr) -> Option<&'e str> {
        match expr {
            ast::Expr::Ident(name) => Some(name),
            ast::Expr::Call { func, args } if matches!(func.as_ref(), ast::Expr::Ident(f) if f == "ptr_add") => {
                args.first().and_then(|a| root(&a.value))
            }
            _ => None,
        }
    }
    fn exprs_of(stmt: &ast::Stmt) -> Vec<&ast::Expr> {
        match &stmt.kind {
            ast::StmtKind::Assign { value, .. } => vec![value],
            ast::StmtKind::Return(e) | ast::StmtKind::Expr(e) => vec![e],
            ast::StmtKind::If { cond, .. } | ast::StmtKind::While { cond, .. } => vec![cond],
            ast::StmtKind::Stream { carry_updates, .. } => carry_updates.iter().map(|(_, e)| e).collect(),
            _ => vec![],
        }
    }
    fn nested(stmt: &ast::Stmt) -> Vec<&ast::Stmt> {
        match &stmt.kind {
            ast::StmtKind::If { then_body, else_body, .. } => then_body.iter().chain(else_body).collect(),
            ast::StmtKind::While { body, .. } | ast::StmtKind::Stream { body, .. } => body.iter().collect(),
            _ => vec![],
        }
    }
    fn calls<'e>(expr: &'e ast::Expr, out: &mut Vec<(&'e str, &'e [ast::CallArg])>) {
        match expr {
            ast::Expr::Call { func, args } => {
                match func.as_ref() {
                    ast::Expr::Ident(name) => out.push((name, args)),
                    ast::Expr::Instantiate { name, .. } => out.push((name, args)),
                    _ => {}
                }
                for arg in args {
                    calls(&arg.value, out);
                }
            }
            ast::Expr::BinOp { lhs, rhs, .. } | ast::Expr::Logical { lhs, rhs, .. } => {
                calls(lhs, out);
                calls(rhs, out);
            }
            ast::Expr::UnaryOp { operand, .. } | ast::Expr::Reduction { operand, .. } => calls(operand, out),
            ast::Expr::Masked { mask, body, fallback } => {
                calls(mask, out);
                calls(body, out);
                if let Some(f) = fallback {
                    calls(f, out);
                }
            }
            ast::Expr::Field { base, .. } => calls(base, out),
            _ => {}
        }
    }
    fn walk<'s>(stmts: &'s [ast::Stmt], visit: &mut dyn FnMut(&'s ast::Stmt)) {
        for stmt in stmts {
            visit(stmt);
            let inner = nested(stmt);
            for s in inner {
                walk(std::slice::from_ref(s), visit);
            }
        }
    }

    let fns: Vec<&ast::FnDef> = items
        .iter()
        .filter_map(|i| if let ast::Item::Fn(f) = i { Some(f) } else { None })
        .collect();
    let mut needs: HashMap<String, Vec<bool>> =
        fns.iter().map(|f| (f.name.clone(), vec![false; f.params.len()])).collect();
    loop {
        let mut changed = false;
        for f in &fns {
            for (idx, param) in f.params.iter().enumerate() {
                if needs[&f.name][idx] || !is_ptr_type_name(&param.ty.name) {
                    continue;
                }
                let mut aliases = std::collections::HashSet::from([param.name.as_str()]);
                loop {
                    let before = aliases.len();
                    walk(&f.body, &mut |stmt| {
                        if let ast::StmtKind::Assign { target: ast::AssignTarget::Ident(name), value, .. } = &stmt.kind {
                            if root(value).is_some_and(|r| aliases.contains(r)) {
                                aliases.insert(name.as_str());
                            }
                        }
                    });
                    if aliases.len() == before {
                        break;
                    }
                }
                let mut found = false;
                walk(&f.body, &mut |stmt| {
                    if let ast::StmtKind::Stream { buffer, .. } = &stmt.kind {
                        found |= aliases.contains(buffer.as_str());
                    }
                    for expr in exprs_of(stmt) {
                        let mut seen = Vec::new();
                        calls(expr, &mut seen);
                        for (callee, args) in seen {
                            let Some(flags) = needs.get(callee) else { continue };
                            for (arg, needed) in args.iter().zip(flags) {
                                found |= *needed && root(&arg.value).is_some_and(|r| aliases.contains(r));
                            }
                        }
                    }
                });
                if found {
                    needs.get_mut(&f.name).unwrap()[idx] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            return needs;
        }
    }
}

fn lit_kind(expr: &ast::Expr) -> Option<LitKind> {
    match expr {
        ast::Expr::IntLit(_) => Some(LitKind::Int),
        ast::Expr::FloatLit(_) => Some(LitKind::Float),
        ast::Expr::BoolLit(_) => Some(LitKind::Bool),
        ast::Expr::CharLit(_) => Some(LitKind::Char),
        ast::Expr::UnaryOp { operand, .. } => lit_kind(operand),
        ast::Expr::BinOp { .. } | ast::Expr::Logical { .. } => match expr.const_value()? {
            ast::ConstValue::Int(_) => Some(LitKind::Int),
            ast::ConstValue::Float(_) => Some(LitKind::Float),
            ast::ConstValue::Bool(_) => Some(LitKind::Bool),
        },
        _ => None,
    }
}

/// Check if a literal kind is compatible with a scalar type.
fn lit_compatible(kind: LitKind, scalar: ScalarType) -> bool {
    match kind {
        LitKind::Int | LitKind::Char => scalar.is_int() || scalar == ScalarType::Bool,
        LitKind::Float => scalar.is_float(),
        LitKind::Bool => scalar == ScalarType::Bool,
    }
}

// ─── Type checker ───

/// Signature of a user-defined function, for inter-function call checking.
#[derive(Clone)]
struct FnSig {
    params: Vec<(String, CheckedType)>,
    ret_ty: Option<CheckedType>,
}

struct TypeChecker {
    arch: Arch,
    /// Functions with comptime parameters the environment doesn't bind. Their bodies
    /// are checked once per set of widths they are called with.
    generics: HashMap<String, ast::FnDef>,
    checked_instances: std::collections::HashSet<String>,
    /// The environment the module was checked with, without any instance's widths.
    module_env: HashMap<String, u64>,
    instance_depth: usize,
    errors: Vec<TypeError>,
    vars: HashMap<String, CheckedType>,
    struct_defs: HashMap<String, ast::StructDef>,
    fn_sigs: HashMap<String, FnSig>,
    consts: HashMap<String, ScalarType>,
    comptime_env: HashMap<String, u64>,
    native_width: u64,
    current_span: ast::Span,
    ret_ty: Option<CheckedType>,
}

impl TypeChecker {
    fn new(comptime_env: &HashMap<String, u64>, native_width: u64, arch: Arch) -> Self {
        TypeChecker {
            arch,
            generics: HashMap::new(),
            checked_instances: std::collections::HashSet::new(),
            module_env: comptime_env.clone(),
            instance_depth: 0,
            errors: Vec::new(),
            vars: HashMap::new(),
            struct_defs: HashMap::new(),
            fn_sigs: HashMap::new(),
            consts: HashMap::new(),
            comptime_env: comptime_env.clone(),
            native_width,
            current_span: ast::Span::dummy(),
            ret_ty: None,
        }
    }

    fn err(&mut self, msg: String) {
        self.errors.push(TypeError {
            span: self.current_span,
            message: msg,
        });
    }

    // ─── Function checking ───

    fn check_fn(&mut self, f: &ast::FnDef) {
        self.vars.clear();
        for (name, scalar) in &self.consts {
            self.vars.insert(name.clone(), CheckedType::Ptr(*scalar));
        }

        // Bind params
        for p in &f.params {
            let ty = if is_ptr_type_name(&p.ty.name) {
                let elem = match &p.ty.width {
                    Some(ast::Width::Param(name)) => ScalarType::from_name(name),
                    _ => {
                        self.err(format!("ptr type '{}' must have element type (e.g. ptr[u8])", p.name));
                        ScalarType::U8
                    }
                };
                CheckedType::Ptr(elem)
            } else if is_scalar_type_name(&p.ty.name) {
                CheckedType::Vec(resolve_type(&p.ty, &self.comptime_env, self.native_width))
            } else {
                // Struct param
                let sdef = self.struct_defs.get(&p.ty.name);
                if let Some(sd) = sdef {
                    let width = p.ty.width.as_ref()
                        .expect("struct type must have width")
                        .eval(&self.comptime_env, self.native_width);
                    let mut struct_env = self.comptime_env.clone();
                    if let Some(cp) = sd.comptime_params.first() {
                        struct_env.insert(cp.name.clone(), width);
                    }
                    let fields: Vec<(String, VecType)> = sd
                        .fields
                        .iter()
                        .map(|fd| {
                            (fd.name.clone(), resolve_type(&fd.ty, &struct_env, self.native_width))
                        })
                        .collect();
                    CheckedType::Struct {
                        name: p.ty.name.clone(),
                        fields,
                    }
                } else {
                    self.err(format!("undefined struct type: {}", p.ty.name));
                    CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                }
            };
            self.vars.insert(p.name.clone(), ty);
        }

        // Store return type
        self.ret_ty = f.ret_ty.clone().map(|rt| {
            if is_ptr_type_name(&rt.name) {
                match &rt.width {
                    Some(ast::Width::Param(elem)) => CheckedType::Ptr(ScalarType::from_name(elem)),
                    _ => CheckedType::Ptr(ScalarType::U8),
                }
            } else if is_scalar_type_name(&rt.name) {
                CheckedType::Vec(resolve_type(&rt, &self.comptime_env, self.native_width))
            } else {
                let env = self.comptime_env.clone();
                self.struct_type_of(&rt, &env)
            }
        });
        if f.extern_c && matches!(self.ret_ty, Some(CheckedType::Ptr(_) | CheckedType::Struct { .. })) {
            self.err(format!("extern fn '{}' can only return a vector or scalar", f.name));
        }

        // Check body
        for stmt in &f.body {
            self.check_stmt(stmt);
        }

        if self.ret_ty.is_some() && !matches!(f.body.last().map(|s| &s.kind), Some(ast::StmtKind::Return(_))) {
            self.err(format!("function '{}' doesn't return a value", f.name));
        }
    }

    fn check_condition(&mut self, what: &str, cond: &ast::Expr) {
        match self.infer_expr(cond) {
            CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool && vt.width == 1 => {}
            CheckedType::Literal(LitKind::Bool) => {}
            CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool => {
                self.err(format!("{} condition must be width 1 (scalar bool), got width {}", what, vt.width));
            }
            other => self.err(format!("{} condition must be bool, got {}", what, other.describe())),
        }
    }

    fn check_branch_types(&mut self, name: &str, a: &CheckedType, b: &CheckedType) {
        let same = match (a, b) {
            (CheckedType::Vec(x), CheckedType::Vec(y)) => x.width == y.width && x.scalar == y.scalar,
            (CheckedType::Struct { name: x, .. }, CheckedType::Struct { name: y, .. }) => x == y,
            (CheckedType::Ptr(x), CheckedType::Ptr(y)) => x == y,
            _ => false,
        };
        if !same {
            self.err(format!(
                "variable '{}' has different types in the two branches of an if: {} and {}",
                name, a.describe(), b.describe()
            ));
        }
    }

    fn check_loop_type(&mut self, name: &str, a: &CheckedType, b: &CheckedType) {
        let same = match (a, b) {
            (CheckedType::Vec(x), CheckedType::Vec(y)) => x.width == y.width && x.scalar == y.scalar,
            (CheckedType::Struct { name: x, fields: xf }, CheckedType::Struct { name: y, fields: yf }) => {
                x == y && xf.iter().zip(yf).all(|((_, p), (_, q))| p.width == q.width && p.scalar == q.scalar)
            }
            _ => false,
        };
        if !same {
            self.err(format!(
                "variable '{}' changes type in a while loop: {} becomes {}",
                name, a.describe(), b.describe()
            ));
        }
    }

    /// Check a value used where `expected` is required.
    fn check_assignable(&mut self, what: &str, expected: &CheckedType, expr: &ast::Expr, actual: &CheckedType) {
        match (expected, actual) {
            (CheckedType::Vec(want), _) if is_literal(expr) => {
                if let Some(kind) = lit_kind(expr) {
                    if !lit_compatible(kind, want.scalar) {
                        self.err(format!(
                            "{}: {} not compatible with {:?}",
                            what, CheckedType::Literal(kind).describe(), want.scalar
                        ));
                    }
                }
            }
            (CheckedType::Vec(_), _) if takes_expected_type(expr) => {}
            (CheckedType::Vec(want), CheckedType::Vec(got)) => {
                if want.width != got.width {
                    self.err(format!("{}: expected width {}, got {}", what, want.width, got.width));
                } else if !same_machine_type(want.scalar, got.scalar) {
                    self.err(format!("{}: expected {:?}, got {:?}", what, want.scalar, got.scalar));
                }
            }
            (CheckedType::Ptr(want), CheckedType::Ptr(got)) => {
                if want != got {
                    self.err(format!("{}: expected ptr[{:?}], got ptr[{:?}]", what, want, got));
                }
            }
            (CheckedType::Struct { name: want, fields: wf }, CheckedType::Struct { name: got, fields: gf }) => {
                let same = want == got
                    && wf.iter().zip(gf).all(|((_, a), (_, b))| a.width == b.width && same_machine_type(a.scalar, b.scalar));
                if want != got {
                    self.err(format!("{}: expected struct {}, got struct {}", what, want, got));
                } else if !same {
                    self.err(format!("{}: struct {} has the wrong width", what, want));
                }
            }
            _ => self.err(format!("{}: expected {}, got {}", what, expected.describe(), actual.describe())),
        }
    }

    // ─── Statement checking ───

    fn check_stmt(&mut self, stmt: &ast::Stmt) {
        self.current_span = stmt.span;
        match &stmt.kind {
            ast::StmtKind::Assign { target, ty: ann_ty, value } => {
                match target {
                    ast::AssignTarget::Ident(name) => {
                        if let ast::Expr::StructLit {
                            name: sname,
                            width,
                            fields: field_exprs,
                        } = value
                        {
                            self.check_struct_lit(name, sname, width, field_exprs);
                        } else if let Some(ann) = ann_ty {
                            // Typed assignment
                            if is_scalar_type_name(&ann.name) {
                                let target_ty = resolve_type(ann, &self.comptime_env, self.native_width);
                                let actual = self.infer_expr(value);
                                self.check_assignable("annotation", &CheckedType::Vec(target_ty), value, &actual);
                                self.vars.insert(name.clone(), CheckedType::Vec(target_ty));
                            } else {
                                self.err(format!("type annotation must be a scalar type, got: {}", ann.name));
                                let _ = self.infer_expr(value);
                            }
                        } else if is_literal(value) {
                            match self.vars.get(name).cloned() {
                                Some(CheckedType::Vec(existing)) => {
                                    if let Some(kind) = lit_kind(value) {
                                        if !lit_compatible(kind, existing.scalar) {
                                            self.err(format!(
                                                "{} not compatible with {:?}",
                                                CheckedType::Literal(kind).describe(),
                                                existing.scalar
                                            ));
                                        }
                                    }
                                }
                                _ => self.err(format!(
                                    "a literal needs a type here: write `{}: type = ...`",
                                    name
                                )),
                            }
                        } else {
                            let vty = self.infer_expr(value);
                            if matches!(vty, CheckedType::Void) {
                                self.err(format!("'{}' can't be assigned from something with no value", name));
                            }
                            self.vars.insert(name.clone(), vty);
                        }
                    }
                    ast::AssignTarget::Scatter { base, index, mask, .. } => {
                        let bty = self.infer_expr(base);
                        if !matches!(bty, CheckedType::Ptr(_)) {
                            self.err(format!("scatter target must be a pointer, got {}", bty.describe()));
                        }
                        let ity = self.infer_expr(index);
                        if let CheckedType::Vec(ivt) = &ity {
                            if !ivt.scalar.is_int() {
                                self.err("scatter index must be an integer vector".to_string());
                            }
                        }
                        if let Some(m) = mask {
                            let mty = self.infer_expr(m);
                            if let CheckedType::Vec(mvt) = &mty {
                                if mvt.scalar != ScalarType::Bool {
                                    self.err("scatter mask must be a bool vector".to_string());
                                }
                            }
                        }
                        let vty = self.infer_expr(value);
                        if let (CheckedType::Ptr(elem), CheckedType::Vec(ivt)) = (&bty, &ity) {
                            self.check_assignable(
                                "scatter value",
                                &CheckedType::Vec(VecType { scalar: *elem, width: ivt.width }),
                                value,
                                &vty,
                            );
                        }
                    }
                    ast::AssignTarget::Destructure(names) => match self.infer_expr(value) {
                        CheckedType::Struct { name, fields } => {
                            if fields.len() != names.len() {
                                self.err(format!(
                                    "destructuring {} names from struct {}, which has {} fields",
                                    names.len(), name, fields.len()
                                ));
                            }
                            for (target, (_, vt)) in names.iter().zip(&fields) {
                                self.vars.insert(target.clone(), CheckedType::Vec(*vt));
                            }
                        }
                        other => self.err(format!(
                            "destructuring needs a struct value, got {}",
                            other.describe()
                        )),
                    },
                }
            }
            ast::StmtKind::ReturnVoid => {
                if self.ret_ty.is_some() {
                    self.err("this function must return a value".to_string());
                }
            }
            ast::StmtKind::Break => self.err("break outside of a loop".to_string()),
            ast::StmtKind::Return(expr) => {
                let rty = self.infer_expr(expr);
                match self.ret_ty.clone() {
                    Some(expected) => self.check_assignable("return value", &expected, expr, &rty),
                    None => self.err("this function has no return type, so it can't return a value".to_string()),
                }
            }
            ast::StmtKind::Expr(expr) => {
                let _ = self.infer_expr(expr);
            }
            ast::StmtKind::If {
                cond,
                then_body,
                else_body,
            } => {
                self.check_condition("if", cond);

                // Each branch sees the variables from before the `if`, and only
                // the taken branch runs. Afterwards a variable keeps its binding if
                // it existed before; a new name is visible only if both branches
                // define it, or the branch that doesn't can never continue.
                let before = self.vars.clone();
                for stmt in then_body {
                    self.check_stmt(stmt);
                }
                let then_vars = std::mem::replace(&mut self.vars, before.clone());
                for stmt in else_body {
                    self.check_stmt(stmt);
                }
                let else_vars = std::mem::replace(&mut self.vars, before.clone());

                let (then_continues, else_continues) = ast::sides_that_continue(cond, then_body, else_body);
                for (name, then_ty) in &then_vars {
                    match else_vars.get(name) {
                        Some(else_ty) => {
                            self.check_branch_types(name, then_ty, else_ty);
                            self.vars.insert(name.clone(), then_ty.clone());
                        }
                        None if !else_continues => {
                            self.vars.insert(name.clone(), then_ty.clone());
                        }
                        None => {}
                    }
                }
                if !then_continues {
                    for (name, else_ty) in &else_vars {
                        if !then_vars.contains_key(name) {
                            self.vars.insert(name.clone(), else_ty.clone());
                        }
                    }
                }
            }
            ast::StmtKind::While { cond, body } => {
                self.check_condition("while", cond);
                let before = self.vars.clone();
                for stmt in body {
                    self.check_stmt(stmt);
                }
                // A variable the loop reassigns carries from one iteration to the next.
                let after = std::mem::replace(&mut self.vars, before.clone());
                for (name, before_ty) in &before {
                    if let Some(after_ty) = after.get(name) {
                        if matches!(before_ty, CheckedType::Vec(_) | CheckedType::Struct { .. }) {
                            self.check_loop_type(name, before_ty, after_ty);
                        }
                    }
                }
            }
            ast::StmtKind::Stream {
                chunk_name,
                chunk_ty,
                buffer,
                carry,
                body,
                carry_updates,
            } => {
                // Check buffer is ptr
                if let Some(bty) = self.vars.get(buffer) {
                    if !matches!(bty, CheckedType::Ptr(_)) {
                        self.err(format!("stream buffer '{}' must be a pointer, got {}", buffer, bty.describe()));
                    }
                } else {
                    self.err(format!("undefined variable: {}", buffer));
                }

                // Save outer vars, create inner scope
                let outer_vars = self.vars.clone();
                let outer_ret = self.ret_ty.take();

                // Bind chunk
                let chunk_vty = resolve_type(chunk_ty, &self.comptime_env, self.native_width);
                if let Some(CheckedType::Ptr(elem)) = outer_vars.get(buffer) {
                    if chunk_vty.scalar != *elem {
                        self.err(format!(
                            "stream chunks must have the buffer's element type: {:?}, not {:?}",
                            elem, chunk_vty.scalar
                        ));
                    }
                }
                self.vars.insert(chunk_name.clone(), CheckedType::Vec(chunk_vty));

                // Bind chunk_offset as i32[1]
                self.vars.insert(
                    "chunk_offset".into(),
                    CheckedType::Vec(VecType {
                        scalar: ScalarType::I32,
                        width: 1,
                    }),
                );

                // Bind carries
                let mut carry_types: Vec<(String, VecType)> = Vec::new();
                for c in carry {
                    let cty = resolve_type(&c.ty, &self.comptime_env, self.native_width);
                    let init_ty = self.infer_expr(&c.init);
                    self.check_assignable(&format!("carry '{}' start", c.name), &CheckedType::Vec(cty), &c.init, &init_ty);
                    self.vars.insert(c.name.clone(), CheckedType::Vec(cty));
                    carry_types.push((c.name.clone(), cty));
                }

                // Check body
                for stmt in body {
                    self.check_stmt(stmt);
                }

                // Only carries leave the stream. Anything else assigned is a new local,
                // which would silently do nothing to an outer variable of the same name.
                let mut assigned = Vec::new();
                crate::desugar::collect_assigned(body, &mut assigned);
                for name in assigned {
                    if outer_vars.contains_key(&name) && !carry_types.iter().any(|(c, _)| *c == name) {
                        self.err(format!(
                            "'{}' is assigned in a stream, which won't change it outside; make it a carry",
                            name
                        ));
                    }
                }

                // Check carry updates
                let mut updated = std::collections::HashSet::new();
                for (cname, cexpr) in carry_updates {
                    let uty = self.infer_expr(cexpr);
                    if !updated.insert(cname.clone()) {
                        self.err(format!("carry '{}' is updated twice", cname));
                    }
                    match carry_types.iter().find(|(n, _)| n == cname) {
                        Some((_, expected)) => {
                            self.check_assignable(&format!("carry '{}' update", cname), &CheckedType::Vec(*expected), cexpr, &uty);
                        }
                        None => self.err(format!("'{}' isn't one of this stream's carries", cname)),
                    }
                }

                // Restore outer scope, but bind carry outputs into it
                self.vars = outer_vars;
                self.ret_ty = outer_ret;
                for (cname, cty) in &carry_types {
                    self.vars.insert(cname.clone(), CheckedType::Vec(*cty));
                }
            }
        }
    }

    fn check_struct_lit(
        &mut self,
        var_name: &str,
        struct_name: &str,
        width: &ast::Width,
        field_exprs: &[(String, ast::Expr)],
    ) {
        let ty = self.struct_lit_type(struct_name, width, field_exprs);
        self.vars.insert(var_name.to_string(), ty);
    }

    /// Check a struct literal's fields, and return its type.
    fn struct_lit_type(
        &mut self,
        struct_name: &str,
        width: &ast::Width,
        field_exprs: &[(String, ast::Expr)],
    ) -> CheckedType {
        let ty = ast::Type { name: struct_name.to_string(), width: Some(width.clone()) };
        let env = self.comptime_env.clone();
        let checked = self.struct_type_of(&ty, &env);
        let CheckedType::Struct { fields, .. } = &checked else {
            return checked;
        };
        let mut seen = std::collections::HashSet::new();
        for (field, expr) in field_exprs {
            if !seen.insert(field.clone()) {
                self.err(format!("struct literal '{}' sets field '{}' twice", struct_name, field));
            }
            let actual = self.infer_expr(expr);
            match fields.iter().find(|(name, _)| name == field) {
                Some((_, expected)) => {
                    self.check_assignable(&format!("field '{}'", field), &CheckedType::Vec(*expected), expr, &actual);
                }
                None => self.err(format!("struct '{}' has no field '{}'", struct_name, field)),
            }
        }
        for (name, _) in fields {
            if !seen.contains(name) {
                self.err(format!("struct literal '{}' missing field '{}'", struct_name, name));
            }
        }
        checked
    }

    // ─── Expression type inference ───

    fn infer_expr(&mut self, expr: &ast::Expr) -> CheckedType {
        match expr {
            ast::Expr::Instantiate { name, .. } => {
                self.err(format!("generic function '{}' can only be called", name));
                CheckedType::Void
            }
            ast::Expr::Logical { lhs, rhs, .. } => {
                for side in [lhs, rhs] {
                    if !matches!(side.as_ref(), ast::Expr::BoolLit(_)) {
                        let ty = self.infer_expr(side);
                        if !matches!(&ty, CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool && vt.width == 1) {
                            self.err(format!(
                                "&& and || need scalar bools, got {}; use & and | for lanes",
                                ty.describe()
                            ));
                        }
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::Bool, width: 1 })
            }
            ast::Expr::Ident(name) => {
                if let Some(ty) = self.vars.get(name) {
                    ty.clone()
                } else {
                    self.err(format!("undefined variable: {}", name));
                    // Return a dummy to keep checking
                    CheckedType::Vec(VecType {
                        scalar: ScalarType::I32,
                        width: 1,
                    })
                }
            }

            ast::Expr::IntLit(_) => CheckedType::Literal(LitKind::Int),
            ast::Expr::FloatLit(_) => CheckedType::Literal(LitKind::Float),
            ast::Expr::BoolLit(_) => CheckedType::Literal(LitKind::Bool),
            ast::Expr::CharLit(_) => CheckedType::Literal(LitKind::Char),

            ast::Expr::VecLit { elem_type, values } => {
                let scalar = ScalarType::from_name(elem_type);
                CheckedType::Vec(VecType {
                    scalar,
                    width: values.len() as u64,
                })
            }

            ast::Expr::BinOp { op, lhs, rhs } => self.infer_binop(*op, lhs, rhs),

            ast::Expr::UnaryOp { op, operand } => {
                if matches!(op, ast::UnaryOp::Neg) && lit_kind(operand) == Some(LitKind::Bool) {
                    self.err("a bool can't be negated: use ~".to_string());
                }
                if is_literal(expr) {
                    return CheckedType::Literal(lit_kind(expr).unwrap_or(LitKind::Int));
                }
                let oty = self.infer_expr(operand);
                if let CheckedType::Vec(vt) = &oty {
                    match op {
                        ast::UnaryOp::Neg if vt.scalar == ScalarType::Bool => {
                            self.err("a bool can't be negated: use ~".to_string());
                        }
                        ast::UnaryOp::Not if vt.scalar.is_float() => {
                            self.err("~ isn't defined on floats".to_string());
                        }
                        _ => {}
                    }
                }
                match oty {
                    CheckedType::Vec(vt) => {
                        match op {
                            ast::UnaryOp::Not => {
                                if vt.scalar.is_float() {
                                    self.err("bitwise NOT (~) requires integer type".to_string());
                                }
                            }
                            ast::UnaryOp::Neg => {} // works on both int and float
                        }
                        CheckedType::Vec(vt)
                    }
                    CheckedType::Literal(k) => CheckedType::Literal(k),
                    other => {
                        self.err(format!("cannot apply unary op to {}", other.describe()));
                        other
                    }
                }
            }

            ast::Expr::Masked { mask, body, fallback } => {
                let mty = self.infer_expr(mask);
                let mask_width = match &mty {
                    CheckedType::Vec(mvt) if mvt.scalar == ScalarType::Bool => mvt.width,
                    other => {
                        self.err(format!("mask must be a bool vector, got {}", other.describe()));
                        1
                    }
                };
                let fallback_ty = fallback.as_deref().map(|f| (is_literal(f), self.infer_expr(f)));
                let body_literal = is_literal(body);
                let body_ty = self.infer_expr(body);
                let branch = match (body_literal, &body_ty, &fallback_ty) {
                    (true, _, None) | (true, _, Some((true, _))) => {
                        let scalar = if lit_kind(body) == Some(LitKind::Float) { ScalarType::F32 } else { ScalarType::I32 };
                        return CheckedType::Vec(VecType { scalar, width: mask_width });
                    }
                    (true, _, Some((false, f))) => f.clone(),
                    (false, b, None) | (false, b, Some((true, _))) => b.clone(),
                    (false, b, Some((false, f))) => {
                        if let (CheckedType::Vec(x), CheckedType::Vec(y)) = (b, f) {
                            if !same_machine_type(x.scalar, y.scalar) {
                                self.err(format!("masked branches have different types: {:?} and {:?}", x.scalar, y.scalar));
                            }
                            if x.width != y.width && x.width != 1 && y.width != 1 {
                                self.err(format!("masked branches have different widths: {} and {}", x.width, y.width));
                            }
                            if x.width == 1 { f.clone() } else { b.clone() }
                        } else {
                            b.clone()
                        }
                    }
                };
                match branch {
                    CheckedType::Vec(vt) => {
                        if mask_width > 1 && vt.width != 1 && vt.width != mask_width {
                            self.err(format!("mask has {} lanes but its branch has {}", mask_width, vt.width));
                        }
                        CheckedType::Vec(VecType { scalar: vt.scalar, width: vt.width.max(mask_width) })
                    }
                    other => {
                        self.err(format!("masked branches must be vectors, got {}", other.describe()));
                        other
                    }
                }
            }

            ast::Expr::Reduction { op, operand } => {
                let oty = self.infer_expr(operand);
                match oty {
                    CheckedType::Vec(vt) => {
                        if vt.scalar == ScalarType::Bool && matches!(op, ast::ReductionOp::Add | ast::ReductionOp::Mul) {
                            self.err("+/ and */ aren't defined on bools; use popcount, |/ or &/".to_string());
                        }
                        if vt.scalar.is_float() && matches!(op, ast::ReductionOp::Or | ast::ReductionOp::And) {
                            self.err("|/ and &/ aren't defined on floats".to_string());
                        }
                        CheckedType::Vec(VecType { scalar: vt.scalar, width: 1 })
                    }
                    _ => {
                        self.err("reduction operand must be a vector".to_string());
                        oty
                    }
                }
            }

            ast::Expr::Scan { op: _, operand, seed } => {
                let oty = self.infer_expr(operand);
                if matches!(&oty, CheckedType::Vec(vt) if vt.width < 2) {
                    self.err("a scan needs a vector of at least 2 lanes".to_string());
                }
                if let Some(s) = seed {
                    let sty = self.infer_expr(s);
                    if let (CheckedType::Vec(ovt), CheckedType::Vec(svt)) = (&oty, &sty) {
                        if svt.width != 1 {
                            self.err(format!("scan seed must be width 1, got {}", svt.width));
                        }
                        if svt.scalar != ovt.scalar {
                            self.err("scan seed scalar type must match operand".to_string());
                        }
                    }
                }
                oty
            }

            ast::Expr::Field { base, field } => {
                let bty = self.infer_expr(base);
                match bty {
                    CheckedType::Struct { name, fields } => {
                        if let Some((_, vt)) = fields.iter().find(|(n, _)| n == field) {
                            CheckedType::Vec(*vt)
                        } else {
                            self.err(format!("struct '{}' has no field '{}'", name, field));
                            CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                        }
                    }
                    _ => {
                        self.err(format!("field access on non-struct: {}", bty.describe()));
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            }

            ast::Expr::Call { func, args } => self.infer_call(func, args),

            ast::Expr::Gather { base, index, mask } => {
                let bty = self.infer_expr(base);
                let ity = self.infer_expr(index);
                if let Some(m) = mask {
                    let mty = self.infer_expr(m);
                    if let CheckedType::Vec(mvt) = &mty {
                        if mvt.scalar != ScalarType::Bool {
                            self.err("gather mask must be bool vector".to_string());
                        }
                    }
                }
                match (&bty, &ity) {
                    (CheckedType::Ptr(elem_scalar), CheckedType::Vec(idx_vt)) => {
                        if !idx_vt.scalar.is_int() {
                            self.err("gather index must be integer vector".to_string());
                        }
                        CheckedType::Vec(VecType {
                            scalar: *elem_scalar,
                            width: idx_vt.width,
                        })
                    }
                    _ => {
                        if !matches!(bty, CheckedType::Ptr(_)) {
                            self.err(format!("gather base must be a pointer, got {}", bty.describe()));
                        }
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            }

            ast::Expr::StructLit { name, width, fields } => self.struct_lit_type(name, width, fields),

            ast::Expr::Load { ty, ptr, offset, .. } => {
                let pty = self.infer_expr(ptr);
                let loaded = resolve_type(ty, &self.comptime_env, self.native_width);
                if loaded.scalar == ScalarType::Bool {
                    self.err("bools can't be loaded: load bytes and compare them".to_string());
                }
                match pty {
                    CheckedType::Ptr(elem) => {
                        let bytes = loaded.scalar.bit_width() as u64 * loaded.width;
                        if bytes % elem.bit_width() as u64 != 0 {
                            self.err(format!(
                                "can't load {:?}[{}] through ptr[{:?}]: not a whole number of elements",
                                loaded.scalar, loaded.width, elem
                            ));
                        }
                    }
                    other => self.err(format!("load requires pointer, got {}", other.describe())),
                }
                if let Some(off) = offset {
                    self.check_offset(off);
                }
                CheckedType::Vec(loaded)
            }
        }
    }

    /// A memory offset: a literal, or a scalar integer.
    fn check_offset(&mut self, offset: &ast::Expr) {
        if is_literal(offset) {
            if !matches!(offset.const_value(), Some(ast::ConstValue::Int(_))) {
                self.err("a memory offset must be an integer".to_string());
            }
            return;
        }
        match self.infer_expr(offset) {
            CheckedType::Vec(vt) if vt.width == 1 && vt.scalar.is_int() => {}
            other => self.err(format!("a memory offset must be a scalar integer, got {}", other.describe())),
        }
    }

    // ─── Binary op type inference ───

    fn infer_binop(
        &mut self,
        op: ast::BinOp,
        lhs: &ast::Expr,
        rhs: &ast::Expr,
    ) -> CheckedType {
        // Bit shifts: the amount is a literal, a scalar, or matches the operand's width
        if op == ast::BinOp::BitShl || op == ast::BinOp::BitShr {
            if is_literal(lhs) {
                if !is_literal(rhs) {
                    let _ = self.infer_expr(rhs);
                }
                return CheckedType::Literal(LitKind::Int);
            }
            let lty = self.infer_expr(lhs);
            if let CheckedType::Vec(vt) = &lty {
                if !vt.scalar.is_int() {
                    self.err(format!("bit shift requires an integer, got {:?}", vt.scalar));
                }
                if let Some(ast::ConstValue::Int(n)) = rhs.const_value() {
                    if n < 0 || n >= vt.scalar.bit_width() as i64 {
                        self.err(format!("shift by {} is out of range for {:?}", n, vt.scalar));
                    }
                }
            }
            if !matches!(rhs.const_value(), Some(ast::ConstValue::Int(_))) {
                let rty = self.infer_expr(rhs);
                if let (CheckedType::Vec(lv), CheckedType::Vec(rv)) = (&lty, &rty) {
                    if !rv.scalar.is_int() {
                        self.err("bit shift amount must be an integer".to_string());
                    } else if rv.width != 1 && rv.width != lv.width {
                        self.err(format!(
                            "bit shift amount width {} does not match operand width {}",
                            rv.width, lv.width
                        ));
                    }
                }
            }
            return lty;
        }

        // Handle literal broadcasting
        let (lty, rty) = match (is_literal(lhs), is_literal(rhs)) {
            (true, true) => {
                // Both literals: a comparison is a bool, anything else keeps its kind.
                let is_comparison = matches!(
                    op,
                    ast::BinOp::Gt | ast::BinOp::Lt | ast::BinOp::GtEq | ast::BinOp::LtEq | ast::BinOp::EqEq | ast::BinOp::NotEq
                );
                let kind = if is_comparison { LitKind::Bool } else { lit_kind(lhs).unwrap_or(LitKind::Int) };
                return CheckedType::Literal(kind);
            }
            (true, false) => {
                let rt = self.infer_expr(rhs);
                let lt = if let CheckedType::Vec(rvt) = &rt {
                    if let Some(kind) = lit_kind(lhs) {
                        if !lit_compatible(kind, rvt.scalar) {
                            self.err(format!(
                                "{} not compatible with {:?}",
                                CheckedType::Literal(kind).describe(),
                                rvt.scalar
                            ));
                        }
                    }
                    CheckedType::Vec(*rvt) // literal broadcasts to match
                } else {
                    self.infer_expr(lhs)
                };
                (lt, rt)
            }
            (false, true) => {
                let lt = self.infer_expr(lhs);
                let rt = if let CheckedType::Vec(lvt) = &lt {
                    if let Some(kind) = lit_kind(rhs) {
                        if !lit_compatible(kind, lvt.scalar) {
                            self.err(format!(
                                "{} not compatible with {:?}",
                                CheckedType::Literal(kind).describe(),
                                lvt.scalar
                            ));
                        }
                    }
                    CheckedType::Vec(*lvt) // literal broadcasts to match
                } else {
                    self.infer_expr(rhs)
                };
                (lt, rt)
            }
            (false, false) => {
                let lt = self.infer_expr(lhs);
                let rt = self.infer_expr(rhs);
                (lt, rt)
            }
        };

        // Both should be Vec by now
        let (lvt, rvt) = match (&lty, &rty) {
            (CheckedType::Vec(l), CheckedType::Vec(r)) => (*l, *r),
            _ => {
                self.err(format!(
                    "this operator needs vectors, got {} and {}; ptr_add moves a pointer",
                    lty.describe(), rty.describe()
                ));
                return lty;
            }
        };

        // Auto-broadcast width-1
        let (lvt, rvt) = if lvt.width == 1 && rvt.width > 1 {
            (VecType { scalar: lvt.scalar, width: rvt.width }, rvt)
        } else if rvt.width == 1 && lvt.width > 1 {
            (lvt, VecType { scalar: rvt.scalar, width: lvt.width })
        } else {
            (lvt, rvt)
        };

        // Width check
        if lvt.width != rvt.width {
            self.err(format!(
                "width mismatch: {:?}[{}] vs {:?}[{}]",
                lvt.scalar, lvt.width, rvt.scalar, rvt.width
            ));
        }

        if !same_machine_type(lvt.scalar, rvt.scalar) {
            self.err(format!(
                "operands have different types: {:?} and {:?}; convert with to_i64 and friends",
                lvt.scalar, rvt.scalar
            ));
        }
        let ordered = matches!(
            op,
            ast::BinOp::Gt | ast::BinOp::Lt | ast::BinOp::GtEq | ast::BinOp::LtEq | ast::BinOp::Div
        );
        // Bit operations can use either reading of a literal's bits. Ordering can't: `x > -1`
        // on a u8 would compare with 255.
        for side in [lhs, rhs] {
            if let (true, Some(ast::ConstValue::Int(n))) = (ordered && lvt.scalar.is_int(), side.const_value()) {
                let bits = lvt.scalar.bit_width();
                let fits = if lvt.scalar.is_unsigned() {
                    n >= 0 && (bits == 64 || n < (1i64 << bits))
                } else {
                    bits == 64 || (n >= -(1i64 << (bits - 1)) && n < (1i64 << (bits - 1)))
                };
                if !fits {
                    self.err(format!("{} isn't a {:?} value, so comparing with it can't be right", n, lvt.scalar));
                }
            }
        }
        if ordered && lvt.scalar != rvt.scalar && !is_literal(lhs) && !is_literal(rhs) {
            self.err(format!(
                "{:?} and {:?} order differently; convert one with to_i64 and friends",
                lvt.scalar, rvt.scalar
            ));
        }
        if lvt.scalar == ScalarType::Bool && matches!(op, ast::BinOp::Add | ast::BinOp::Sub | ast::BinOp::Mul | ast::BinOp::Div) {
            self.err("arithmetic on bools isn't defined; use &, | and ^".to_string());
        }
        if lvt.scalar.is_float() && matches!(op, ast::BinOp::And | ast::BinOp::Or | ast::BinOp::Xor) {
            self.err("&, | and ^ aren't defined on floats".to_string());
        }
        if op == ast::BinOp::Div && lvt.scalar.is_int() && matches!(rhs.const_value(), Some(ast::ConstValue::Int(0))) {
            self.err("division by zero".to_string());
        }

        // Result type
        match op {
            ast::BinOp::Add
            | ast::BinOp::Sub
            | ast::BinOp::Mul
            | ast::BinOp::Div
            | ast::BinOp::And
            | ast::BinOp::Or
            | ast::BinOp::Xor => CheckedType::Vec(lvt),

            ast::BinOp::Gt
            | ast::BinOp::Lt
            | ast::BinOp::GtEq
            | ast::BinOp::LtEq
            | ast::BinOp::EqEq
            | ast::BinOp::NotEq => CheckedType::Vec(VecType {
                scalar: ScalarType::Bool,
                width: lvt.width,
            }),

            ast::BinOp::BitShl | ast::BinOp::BitShr => unreachable!(),
        }
    }

    // ─── Builtin call type inference ───

    fn infer_call(
        &mut self,
        func: &ast::Expr,
        args: &[ast::CallArg],
    ) -> CheckedType {
        let fname = match func {
            ast::Expr::Instantiate { name, widths } => {
                return self.infer_generic_call(name, Some(widths), args);
            }
            ast::Expr::Ident(n) if self.generics.contains_key(n) => {
                return self.infer_generic_call(n, None, args);
            }
            ast::Expr::Ident(n) => n.as_str(),
            ast::Expr::Field { base, field } => {
                // scan.add, scan.xor, etc.
                if let ast::Expr::Ident(base_name) = base.as_ref() {
                    if base_name == "scan" {
                        return self.infer_scan_call(field, args);
                    }
                }
                let _ = self.infer_expr(func);
                return CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 });
            }
            _ => {
                let _ = self.infer_expr(func);
                return CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 });
            }
        };

        match fname {
            "sqrt" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if let CheckedType::Vec(vt) = &aty {
                    if !vt.scalar.is_float() {
                        self.err("sqrt requires float type".to_string());
                    }
                }
                aty
            }
            "popcount" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if let CheckedType::Vec(vt) = &aty {
                    if vt.scalar != ScalarType::Bool {
                        self.err("popcount requires bool vector".to_string());
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
            }
            "compress" => {
                self.check_arg_count(fname, args, 2);
                let dty = self.infer_arg(args, 0);
                let mty = self.infer_arg(args, 1);
                if let CheckedType::Vec(mvt) = &mty {
                    if mvt.scalar != ScalarType::Bool {
                        self.err("compress mask must be bool vector".to_string());
                    }
                }
                dty
            }
            "store" | "store_at" => {
                let range = if fname == "store" { 2..=3 } else { 3..=3 };
                if !range.contains(&args.len()) {
                    self.err(format!("{} takes {:?} arguments, got {}", fname, range, args.len()));
                    return CheckedType::Void;
                }
                match self.infer_arg(args, 0) {
                    CheckedType::Ptr(_) => {}
                    other => self.err(format!("{} needs a pointer, got {}", fname, other.describe())),
                }
                if fname == "store_at" {
                    self.check_offset(&args[1].value);
                    let _ = self.infer_arg(args, 2);
                } else {
                    let value = self.infer_arg(args, 1);
                    if let Some(mask) = args.get(2) {
                        let mask = self.infer_expr(&mask.value);
                        match (&value, &mask) {
                            (CheckedType::Vec(v), CheckedType::Vec(m)) if m.scalar == ScalarType::Bool && m.width == v.width && v.width > 1 => {}
                            _ => self.err(format!(
                                "store's mask must be a bool vector with one lane per value lane, got {} for {}",
                                mask.describe(), value.describe()
                            )),
                        }
                    }
                }
                CheckedType::Void
            }
            "iota" => {
                self.check_arg_count(fname, args, 1);
                if let Some(arg) = args.first() {
                    match &arg.value {
                        ast::Expr::IntLit(n) if *n >= 1 => {
                            return CheckedType::Vec(VecType {
                                scalar: ScalarType::I32,
                                width: *n as u64,
                            });
                        }
                        _ => self.err("iota argument must be an integer literal".to_string()),
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
            }
            "extract" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                if args.len() > 1 {
                    match (&args[1].value, &aty) {
                        (ast::Expr::IntLit(n), CheckedType::Vec(vt)) if *n < 0 || *n as u64 >= vt.width => {
                            self.err(format!("extract lane {} is out of range for {} lanes", n, vt.width));
                        }
                        (ast::Expr::IntLit(_), _) => {}
                        _ => self.err("extract lane must be an integer literal".to_string()),
                    }
                }
                match aty {
                    CheckedType::Vec(vt) => CheckedType::Vec(VecType { scalar: vt.scalar, width: 1 }),
                    _ => aty,
                }
            }
            "compressstore" => {
                self.check_arg_count(fname, args, 4);
                for i in 0..args.len().min(4) {
                    let _ = self.infer_arg(args, i);
                }
                CheckedType::Void
            }
            _ if int_conversion_target(fname).is_some() => {
                self.check_arg_count(fname, args, 1);
                let scalar = int_conversion_target(fname).unwrap();
                match self.infer_arg(args, 0) {
                    CheckedType::Vec(vt) => {
                        if vt.scalar.is_float() {
                            self.err(format!("{} needs an integer or bool, got {:?}", fname, vt.scalar));
                        }
                        CheckedType::Vec(VecType { scalar, width: vt.width })
                    }
                    other => {
                        self.err(format!("{} needs a vector, got {}", fname, other.describe()));
                        other
                    }
                }
            }
            "to_bitmask" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if let CheckedType::Vec(vt) = &aty {
                    if vt.scalar != ScalarType::Bool {
                        self.err("to_bitmask requires bool vector".to_string());
                    }
                    if vt.width > 64 {
                        self.err(format!("to_bitmask can pack at most 64 lanes, got {}", vt.width));
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::U64, width: 1 })
            }
            "from_bitmask" => {
                self.check_arg_count(fname, args, 2);
                let bits = self.infer_arg(args, 0);
                if !matches!(&bits, CheckedType::Vec(vt) if vt.width == 1 && vt.scalar.bit_width() == 64 && vt.scalar.is_int()) {
                    self.err(format!("from_bitmask needs a 64-bit scalar, got {}", bits.describe()));
                }
                if let Some(arg) = args.get(1) {
                    match &arg.value {
                        ast::Expr::IntLit(n) if *n >= 2 && *n <= 64 => {
                            return CheckedType::Vec(VecType {
                                scalar: ScalarType::Bool,
                                width: *n as u64,
                            });
                        }
                        _ => self.err("from_bitmask width must be an integer literal".to_string()),
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::Bool, width: 64 })
            }
            "clmul" => {
                self.check_arg_count(fname, args, 2);
                for i in 0..2 {
                    let arg = self.infer_arg(args, i);
                    if !matches!(&arg, CheckedType::Vec(vt) if vt.width == 1 && vt.scalar.bit_width() == 64 && vt.scalar.is_int()) {
                        self.err(format!("clmul needs 64-bit scalars, got {}", arg.describe()));
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::U64, width: 1 })
            }
            "xor" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                let _ = self.infer_arg(args, 1);
                aty
            }
            "bit_shr" | "bit_shl" | "lane_shr" | "lane_shl" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                if args.len() > 1 {
                    if !matches!(args[1].value, ast::Expr::IntLit(_)) {
                        self.err(format!("{} amount must be an integer literal", fname));
                    }
                    let is_lane_shift = fname.starts_with("lane");
                    if let (true, ast::Expr::IntLit(n), CheckedType::Vec(vt)) = (is_lane_shift, &args[1].value, &aty) {
                        if vt.width < 2 {
                            self.err(format!("{} needs a vector of at least 2 lanes", fname));
                        } else if *n < 0 || *n as u64 > vt.width {
                            self.err(format!("{} by {} is out of range for {} lanes", fname, n, vt.width));
                        }
                    }
                }
                aty
            }
            "tbl" => {
                self.check_arg_count(fname, args, 2);
                let tty = self.infer_arg(args, 0);
                let ity = self.infer_arg(args, 1);
                if let CheckedType::Vec(tvt) = &tty {
                    if tvt.width != 16 || tvt.scalar != ScalarType::U8 {
                        self.err("tbl table must be u8[16]".to_string());
                    }
                }
                if !matches!(&ity, CheckedType::Vec(ivt) if ivt.scalar.bit_width() == 8 && ivt.scalar.is_int() && ivt.width % 16 == 0) {
                    self.err(format!("tbl indices must be u8 with a multiple of 16 lanes, got {}", ity.describe()));
                }
                ity // result type matches indices
            }
            // scan_dfa(table: u8[256], classes: u8[N], seed: u8[1], num_states: lit) -> u8[N]
            // Prefix scan over a DFA's transition monoid: returns the state after
            // each byte. `table[state*16 + class] = next_state` (≤16 states/classes).
            "scan_dfa" => {
                self.check_arg_count(fname, args, 4);
                let _tty = self.infer_arg(args, 0);
                let cty = self.infer_arg(args, 1);
                let _sty = self.infer_arg(args, 2);
                if !matches!(args[3].value, ast::Expr::IntLit(_)) {
                    self.err("scan_dfa num_states must be an integer literal".to_string());
                }
                cty // result shape matches the class vector
            }
            "gather" => {
                self.check_arg_count(fname, args, 2);
                let bty = self.infer_arg(args, 0);
                let ity = self.infer_arg(args, 1);
                match (&bty, &ity) {
                    (CheckedType::Ptr(elem), CheckedType::Vec(ivt)) => {
                        CheckedType::Vec(VecType { scalar: *elem, width: ivt.width })
                    }
                    _ => {
                        if !matches!(bty, CheckedType::Ptr(_)) {
                            self.err("gather first argument must be a pointer".to_string());
                        }
                        ity
                    }
                }
            }
            "likely" | "unlikely" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if !matches!(&aty, CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool && vt.width == 1) {
                    self.err(format!("{} requires a scalar bool, got {}", fname, aty.describe()));
                }
                aty
            }
            "to_u8" => {
                self.check_arg_count(fname, args, 1);
                match self.infer_arg(args, 0) {
                    CheckedType::Vec(vt) => CheckedType::Vec(VecType { scalar: ScalarType::U8, width: vt.width }),
                    other => other,
                }
            }
            "min" | "max" => {
                self.check_arg_count(fname, args, 2);
                if is_literal(&args[0].value) {
                    return self.infer_arg(args, 1);
                }
                let aty = self.infer_arg(args, 0);
                if !is_literal(&args[1].value) {
                    let bty = self.infer_arg(args, 1);
                    if let (CheckedType::Vec(a), CheckedType::Vec(b)) = (&aty, &bty) {
                        if a.width != b.width {
                            self.err(format!("{} operands have different widths: {} vs {}", fname, a.width, b.width));
                        }
                        if a.scalar != b.scalar {
                            self.err(format!("{} operands have different types: {:?} and {:?}", fname, a.scalar, b.scalar));
                        }
                    }
                }
                aty
            }
            "fold_min" => {
                self.check_arg_count(fname, args, 1);
                let register = self.arch.register_bytes();
                match self.infer_arg(args, 0) {
                    CheckedType::Vec(vt) if vt.scalar == ScalarType::U8 && (vt.width <= register || vt.width % register == 0) => {
                        CheckedType::Vec(VecType { scalar: ScalarType::U8, width: vt.width.min(register) })
                    }
                    other => {
                        self.err(format!(
                            "fold_min requires a u8 vector that fits in or is a multiple of a {}-byte register, got {}",
                            register, other.describe()
                        ));
                        other
                    }
                }
            }
            "any_in_first" => {
                self.check_arg_count(fname, args, 2);
                let mty = self.infer_arg(args, 0);
                let widths: &[u64] = match self.arch {
                    Arch::Aarch64 => &[16],
                    Arch::X86_64 => &[16, 32, 64],
                };
                if !matches!(&mty, CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool && widths.contains(&vt.width)) {
                    self.err(format!("any_in_first needs a bool vector of one register, got {}", mty.describe()));
                }
                if let (Some(ast::ConstValue::Int(n)), CheckedType::Vec(m)) = (args.get(1).and_then(|a| a.value.const_value()), &mty) {
                    if n < 1 || n as u64 > m.width {
                        self.err(format!("any_in_first's count must be from 1 to {}, got {}", m.width, n));
                    }
                } else {
                    let cty = self.infer_arg(args, 1);
                    if !matches!(&cty, CheckedType::Vec(vt) if vt.width == 1 && vt.scalar.is_int()) {
                        self.err(format!("any_in_first's count must be a scalar integer, got {}", cty.describe()));
                    }
                }
                CheckedType::Vec(VecType { scalar: ScalarType::Bool, width: 1 })
            }
            "mask8" => {
                self.check_arg_count(fname, args, 1);
                match self.infer_arg(args, 0) {
                    CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool => {
                        CheckedType::Vec(VecType { scalar: ScalarType::U8, width: vt.width })
                    }
                    other => {
                        self.err(format!("mask8 requires a bool vector, got {}", other.describe()));
                        other
                    }
                }
            }
            "nibblemask" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if !matches!(&aty, CheckedType::Vec(vt) if vt.width == 16 && vt.scalar == ScalarType::U8) {
                    self.err(format!("nibblemask requires a u8[16], got {}", aty.describe()));
                }
                CheckedType::Vec(VecType { scalar: ScalarType::U64, width: 1 })
            }
            "shuffle" => {
                self.check_arg_count(fname, args, 3);
                let aty = self.infer_arg(args, 0);
                let bty = self.infer_arg(args, 1);
                let Some(ast::Expr::VecLit { values, .. }) = args.get(2).map(|a| &a.value) else {
                    self.err("shuffle's lanes must be a vector literal".to_string());
                    return aty;
                };
                match (&aty, &bty) {
                    (CheckedType::Vec(a), CheckedType::Vec(b)) if a.scalar == b.scalar && a.width == b.width && a.width > 1 && values.len() > 1 => {
                        if let Some(bad) = values.iter().find(|&&lane| lane < 0 || lane >= 2 * a.width as i64) {
                            self.err(format!("shuffle lane {} is out of range", bad));
                        }
                        CheckedType::Vec(VecType { scalar: a.scalar, width: values.len() as u64 })
                    }
                    _ => {
                        self.err(format!("shuffle requires two vectors of the same type and at least 2 lanes, got {} and {}", aty.describe(), bty.describe()));
                        aty
                    }
                }
            }
            "concat" => {
                self.check_arg_count(fname, args, 2);
                let aty = self.infer_arg(args, 0);
                let bty = self.infer_arg(args, 1);
                match (&aty, &bty) {
                    (CheckedType::Vec(a), CheckedType::Vec(b)) if a.scalar == b.scalar && a.width == b.width && a.width > 1 => {
                        CheckedType::Vec(VecType { scalar: a.scalar, width: a.width * 2 })
                    }
                    _ => {
                        self.err(format!("concat requires two vectors of the same type, got {} and {}", aty.describe(), bty.describe()));
                        aty
                    }
                }
            }
            "ptr_addr" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if !matches!(aty, CheckedType::Ptr(_)) {
                    self.err(format!("ptr_addr requires a pointer, got {}", aty.describe()));
                }
                CheckedType::Vec(VecType { scalar: ScalarType::I64, width: 1 })
            }
            "any" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if !matches!(&aty, CheckedType::Vec(vt) if vt.scalar == ScalarType::Bool) {
                    self.err(format!("any requires a bool vector, got {}", aty.describe()));
                }
                CheckedType::Vec(VecType { scalar: ScalarType::Bool, width: 1 })
            }
            "split_lo" | "split_hi" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                match aty {
                    CheckedType::Vec(vt) => {
                        if vt.width < 4 || vt.width % 2 != 0 {
                            self.err(format!("{} needs an even number of lanes, at least 4; got {}", fname, vt.width));
                        }
                        CheckedType::Vec(VecType { scalar: vt.scalar, width: vt.width / 2 })
                    }
                    _ => aty,
                }
            }
            "ctz" | "clear_lowest_bit" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                if !matches!(&aty, CheckedType::Vec(vt) if vt.width == 1 && vt.scalar.bit_width() == 64 && vt.scalar.is_int()) {
                    self.err(format!("{} needs a 64-bit scalar, got {}", fname, aty.describe()));
                }
                aty
            }
            "load_at" => {
                self.check_arg_count(fname, args, 2);
                let pty = self.infer_arg(args, 0);
                if let Some(offset) = args.get(1) {
                    self.check_offset(&offset.value);
                }
                match pty {
                    CheckedType::Ptr(scalar) => CheckedType::Vec(VecType { scalar, width: 1 }),
                    _ => {
                        self.err("load_at first argument must be a pointer".to_string());
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            }
            "ptr_add" => {
                self.check_arg_count(fname, args, 2);
                let pty = self.infer_arg(args, 0);
                if let Some(offset) = args.get(1) {
                    self.check_offset(&offset.value);
                }
                match pty {
                    CheckedType::Ptr(scalar) => CheckedType::Ptr(scalar),
                    _ => {
                        self.err("ptr_add first argument must be a pointer".to_string());
                        CheckedType::Ptr(ScalarType::U8)
                    }
                }
            }
            "bswap" => {
                self.check_arg_count(fname, args, 1);
                let aty = self.infer_arg(args, 0);
                match &aty {
                    CheckedType::Vec(vt) => {
                        match vt.scalar {
                            ScalarType::U16 | ScalarType::I16 |
                            ScalarType::U32 | ScalarType::I32 |
                            ScalarType::U64 | ScalarType::I64 => {},
                            _ => self.err("bswap requires u16/i16/u32/i32/u64/i64 element type".to_string()),
                        }
                    }
                    _ => self.err("bswap argument must be a vector".to_string()),
                }
                aty
            }
            _ => {
                // Check if it's a user-defined function
                if let Some(sig) = self.fn_sigs.get(fname).cloned() {
                    // Validate argument count
                    if args.len() != sig.params.len() {
                        self.err(format!(
                            "function '{}' takes {} argument{}, got {}",
                            fname, sig.params.len(),
                            if sig.params.len() == 1 { "" } else { "s" },
                            args.len()
                        ));
                    }
                    for (i, a) in args.iter().enumerate() {
                        let aty = self.infer_expr(&a.value);
                        let Some((pname, pty)) = sig.params.get(i) else { continue };
                        self.check_assignable(&format!("argument '{}' of '{}'", pname, fname), pty, &a.value, &aty);
                    }
                    // Return the declared return type
                    sig.ret_ty.unwrap_or(CheckedType::Void)
                } else {
                    self.err(format!("unknown function: {}", fname));
                    for a in args {
                        let _ = self.infer_expr(&a.value);
                    }
                    CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                }
            }
        }
    }

    /// A call to a generic function: bind its comptime parameters from explicit widths
    /// and argument types, then check the body once for that binding.
    fn infer_generic_call(
        &mut self,
        name: &str,
        widths: Option<&Vec<ast::Width>>,
        args: &[ast::CallArg],
    ) -> CheckedType {
        let unknown = CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 });
        if self.instance_depth > 64 {
            self.err(format!("generic function '{}' is instantiated too deeply: is it recursive?", name));
            return unknown;
        }
        let Some(def) = self.generics.get(name).cloned() else {
            self.err(format!("'{}' is not a generic function", name));
            return unknown;
        };
        if args.len() != def.params.len() {
            self.err(format!("function '{}' takes {} arguments, got {}", name, def.params.len(), args.len()));
            return unknown;
        }

        let mut env = self.module_env.clone();
        if let Some(widths) = widths {
            if widths.len() != def.comptime_params.len() {
                self.err(format!(
                    "function '{}' takes {} comptime parameters, got {}",
                    name, def.comptime_params.len(), widths.len()
                ));
                return unknown;
            }
            for (param, width) in def.comptime_params.iter().zip(widths) {
                env.insert(param.name.clone(), width.eval(&self.comptime_env, self.native_width));
            }
        }

        let arg_types: Vec<CheckedType> = args.iter().map(|a| self.infer_expr(&a.value)).collect();
        for (param, arg) in def.params.iter().zip(&arg_types) {
            if let (Some(ast::Width::Param(p)), CheckedType::Vec(vt)) = (&param.ty.width, arg) {
                if is_scalar_type_name(&param.ty.name) && def.comptime_params.iter().any(|c| &c.name == p) {
                    match env.get(p) {
                        Some(bound) if *bound != vt.width => {
                            self.err(format!(
                                "argument '{}' of '{}' has width {}, but {} is {}",
                                param.name, name, vt.width, p, bound
                            ));
                            return unknown;
                        }
                        Some(_) => {}
                        None => {
                            env.insert(p.clone(), vt.width);
                        }
                    }
                }
            }
        }
        if let Some(missing) = def.comptime_params.iter().find(|c| !env.contains_key(&c.name)) {
            self.err(format!(
                "can't infer {} for '{}': write {}[...](...)",
                missing.name, name, name
            ));
            return unknown;
        }

        for (i, param) in def.params.iter().enumerate() {
            let expected = if is_ptr_type_name(&param.ty.name) {
                match &param.ty.width {
                    Some(ast::Width::Param(elem)) => CheckedType::Ptr(ScalarType::from_name(elem)),
                    _ => CheckedType::Ptr(ScalarType::U8),
                }
            } else if is_scalar_type_name(&param.ty.name) {
                CheckedType::Vec(resolve_type(&param.ty, &env, self.native_width))
            } else {
                self.struct_type_of(&param.ty, &env)
            };
            self.check_assignable(&format!("argument '{}' of '{}'", param.name, name), &expected, &args[i].value, &arg_types[i]);
        }

        let mut bindings: Vec<String> = def.comptime_params.iter().map(|c| format!("{}={}", c.name, env[&c.name])).collect();
        bindings.sort();
        let key = format!("{}[{}]", name, bindings.join(","));
        if self.checked_instances.insert(key) {
            let saved_vars = std::mem::take(&mut self.vars);
            let saved_ret = self.ret_ty.take();
            self.instance_depth += 1;
            let saved_span = self.current_span;
            let saved_env = std::mem::replace(&mut self.comptime_env, env.clone());
            let widths: HashMap<String, u64> = def
                .comptime_params
                .iter()
                .map(|c| (c.name.clone(), env[&c.name]))
                .collect();
            let mut instance = def.clone();
            crate::comptime::substitute_in_fn(&mut instance, &widths);
            self.check_fn(&instance);
            self.vars = saved_vars;
            self.ret_ty = saved_ret;
            self.instance_depth -= 1;
            self.current_span = saved_span;
            self.comptime_env = saved_env;
        }

        match &def.ret_ty {
            None => CheckedType::Void,
            Some(ret) if is_ptr_type_name(&ret.name) => match &ret.width {
                Some(ast::Width::Param(elem)) => CheckedType::Ptr(ScalarType::from_name(elem)),
                _ => CheckedType::Ptr(ScalarType::U8),
            },
            Some(ret) if is_scalar_type_name(&ret.name) => {
                CheckedType::Vec(resolve_type(ret, &env, self.native_width))
            }
            Some(ret) => self.struct_type_of(ret, &env),
        }
    }

    /// The checked type of a struct type written in the program.
    fn struct_type_of(&mut self, ty: &ast::Type, env: &HashMap<String, u64>) -> CheckedType {
        let Some(def) = self.struct_defs.get(&ty.name).cloned() else {
            self.err(format!("undefined struct type: {}", ty.name));
            return CheckedType::Void;
        };
        let Some(width) = &ty.width else {
            self.err(format!("struct type {} needs a width", ty.name));
            return CheckedType::Void;
        };
        let mut struct_env = env.clone();
        if let Some(cp) = def.comptime_params.first() {
            struct_env.insert(cp.name.clone(), width.eval(env, self.native_width));
        }
        let fields = def
            .fields
            .iter()
            .map(|f| (f.name.clone(), resolve_type(&f.ty, &struct_env, self.native_width)))
            .collect();
        CheckedType::Struct { name: ty.name.clone(), fields }
    }

    fn infer_scan_call(&mut self, op_name: &str, args: &[ast::CallArg]) -> CheckedType {
        match op_name {
            "add" | "xor" | "max" | "preceding_any" => {}
            _ => self.err(format!("unknown scan operation: scan.{}", op_name)),
        }
        if args.is_empty() || args.len() > 2 {
            self.err(format!("scan.{} takes 1 or 2 arguments, got {}", op_name, args.len()));
        }
        let oty = if !args.is_empty() {
            self.infer_expr(&args[0].value)
        } else {
            CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
        };
        if args.len() == 2 {
            let sty = self.infer_expr(&args[1].value);
            if let (CheckedType::Vec(ovt), CheckedType::Vec(svt)) = (&oty, &sty) {
                if svt.width != 1 {
                    self.err(format!("scan seed must be width 1, got {}", svt.width));
                }
                if svt.scalar != ovt.scalar {
                    self.err("scan seed scalar type must match operand".to_string());
                }
            }
        }
        oty
    }

    fn check_arg_count(&mut self, name: &str, args: &[ast::CallArg], expected: usize) {
        if args.len() != expected {
            self.err(format!(
                "{} takes {} argument{}, got {}",
                name,
                expected,
                if expected == 1 { "" } else { "s" },
                args.len()
            ));
        }
    }

    fn infer_arg(&mut self, args: &[ast::CallArg], idx: usize) -> CheckedType {
        if let Some(arg) = args.get(idx) {
            self.infer_expr(&arg.value)
        } else {
            CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
        }
    }
}

// ─── Public API ───

#[cfg(test)]
pub fn typecheck(
    items: &[ast::Item],
    comptime_env: &HashMap<String, u64>,
) -> Result<(), Vec<TypeError>> {
    typecheck_for(items, comptime_env, Arch::host())
}

/// Type check a program for a given architecture, which sets register sizes.
pub fn typecheck_for(
    items: &[ast::Item],
    comptime_env: &HashMap<String, u64>,
    arch: Arch,
) -> Result<(), Vec<TypeError>> {
    let native_width = arch.register_bytes();
    let items = &crate::comptime::with_register_bytes(items, native_width)[..];
    let mut checker = TypeChecker::new(comptime_env, native_width, arch);

    // Collect struct defs first
    for item in items {
        if let ast::Item::Struct(sd) = item {
            if sd.comptime_params.len() > 1 {
                checker.err(format!("struct '{}' can have at most one comptime parameter", sd.name));
            }
            checker.struct_defs.insert(sd.name.clone(), sd.clone());
        }
    }

    if !checker.errors.is_empty() {
        return Err(checker.errors);
    }

    // Const tables are visible in every function as pointers
    for item in items {
        if let ast::Item::Const(c) = item {
            if !is_scalar_type_name(&c.elem_type) {
                checker.err(format!("const '{}' has unknown element type: {}", c.name, c.elem_type));
                continue;
            }
            checker.consts.insert(c.name.clone(), ScalarType::from_name(&c.elem_type));
        }
    }

    // Collect function signatures for inter-function call checking
    for item in items {
        if let ast::Item::Fn(f) = item {
            if f.comptime_params.iter().any(|p| !comptime_env.contains_key(&p.name)) {
                if f.extern_c {
                    checker.err(format!("extern fn '{}' can't have comptime parameters", f.name));
                }
                checker.generics.insert(f.name.clone(), f.clone());
                continue;
            }
            let params: Vec<(String, CheckedType)> = f.params.iter().map(|p| {
                let ty = if is_ptr_type_name(&p.ty.name) {
                    let elem = match &p.ty.width {
                        Some(ast::Width::Param(name)) => ScalarType::from_name(name),
                        _ => ScalarType::U8,
                    };
                    CheckedType::Ptr(elem)
                } else if is_scalar_type_name(&p.ty.name) {
                    CheckedType::Vec(resolve_type(&p.ty, &comptime_env, native_width))
                } else {
                    // Struct param — resolve to struct type
                    if let Some(sd) = checker.struct_defs.get(&p.ty.name) {
                        let width = p.ty.width.as_ref()
                            .expect("struct type must have width")
                            .eval(&comptime_env, native_width);
                        let mut struct_env = comptime_env.clone();
                        if let Some(cp) = sd.comptime_params.first() {
                            struct_env.insert(cp.name.clone(), width);
                        }
                        let fields = sd.fields.iter().map(|fld| {
                            (fld.name.clone(), resolve_type(&fld.ty, &struct_env, native_width))
                        }).collect();
                        CheckedType::Struct { name: p.ty.name.clone(), fields }
                    } else {
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                };
                (p.name.clone(), ty)
            }).collect();

            let ret_ty = f.ret_ty.as_ref().map(|rt| {
                if is_ptr_type_name(&rt.name) {
                    let elem = match &rt.width {
                        Some(ast::Width::Param(name)) => ScalarType::from_name(name),
                        _ => ScalarType::U8,
                    };
                    CheckedType::Ptr(elem)
                } else if is_scalar_type_name(&rt.name) {
                    CheckedType::Vec(resolve_type(rt, &comptime_env, native_width))
                } else {
                    // Struct return
                    if let Some(sd) = checker.struct_defs.get(&rt.name) {
                        let width = rt.width.as_ref()
                            .expect("struct return must have width")
                            .eval(&comptime_env, native_width);
                        let mut struct_env = comptime_env.clone();
                        if let Some(cp) = sd.comptime_params.first() {
                            struct_env.insert(cp.name.clone(), width);
                        }
                        let fields = sd.fields.iter().map(|fld| {
                            (fld.name.clone(), resolve_type(&fld.ty, &struct_env, native_width))
                        }).collect();
                        CheckedType::Struct { name: rt.name.clone(), fields }
                    } else {
                        CheckedType::Vec(VecType { scalar: ScalarType::I32, width: 1 })
                    }
                }
            });

            checker.fn_sigs.insert(f.name.clone(), FnSig { params, ret_ty });
        }
    }

    let needs_size = params_needing_size(items);
    for item in items {
        if let ast::Item::Fn(f) = item {
            if f.extern_c {
                for (param, needed) in f.params.iter().zip(&needs_size[&f.name]) {
                    if *needed {
                        checker.err(format!(
                            "cannot stream over '{}': an extern fn's pointers have no known size",
                            param.name
                        ));
                    }
                }
            }
            if !checker.generics.contains_key(&f.name) {
                checker.check_fn(f);
            }
        }
    }

    if checker.errors.is_empty() {
        Ok(())
    } else {
        Err(checker.errors)
    }
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    fn check(source: &str) -> Vec<String> {
        let items = parser::parse(source);
        match typecheck(&items, &HashMap::new()) {
            Ok(()) => vec![],
            Err(errs) => errs.into_iter().map(|e| e.message).collect(),
        }
    }

    fn check_ok(source: &str) {
        let errs = check(source);
        assert!(errs.is_empty(), "expected no errors, got: {:?}", errs);
    }

    fn check_err(source: &str, expected_substr: &str) {
        let errs = check(source);
        assert!(
            errs.iter().any(|e| e.contains(expected_substr)),
            "expected error containing '{}', got: {:?}",
            expected_substr,
            errs
        );
    }

    // --- Basic passing ---

    #[test]
    fn test_simple_add() {
        check_ok("fn f(a: i32[4], b: i32[4]) -> i32[4] { return a + b }");
    }

    #[test]
    fn test_literal_broadcast() {
        check_ok("fn f(a: i32[4]) -> i32[4] { return a + 1 }");
    }

    #[test]
    fn test_char_literal_broadcast() {
        check_ok("fn f(chunk: u8[64]) -> bool[64] { return chunk == '\"' }");
    }

    #[test]
    fn test_comparison_produces_bool() {
        check_ok("fn f(a: i32[4], b: i32[4]) -> bool[4] { return a > b }");
    }

    #[test]
    fn test_width_1_broadcast() {
        check_ok("fn f(a: i32[4], b: i32[1]) -> i32[4] { return a + b }");
    }

    #[test]
    fn test_typed_assign() {
        check_ok("fn f() -> u64[1] { x: u64[1] = ~0\n return x }");
    }

    #[test]
    fn test_vec_literal() {
        check_ok("fn f() -> u8[4] { v = [u8: 1, 2, 3, 4]\n return v }");
    }

    #[test]
    fn test_vec_repeat() {
        check_ok("fn f() -> u8[4] { v = [u8: 0; 4]\n return v }");
    }

    #[test]
    fn test_masked_expression() {
        check_ok("fn f(a: i32[4], m: bool[4]) -> i32[4] { return [m] a + 1 : a }");
    }

    #[test]
    fn test_reduction() {
        check_ok("fn f(a: f32[8]) -> f32[1] { return +/ a }");
    }

    #[test]
    fn test_xor_operator() {
        check_ok("fn f(a: u64[1], b: u64[1]) -> u64[1] { return a ^ b }");
    }

    #[test]
    fn test_bit_shift() {
        check_ok("fn f(a: u8[64]) -> u8[64] { return a >> 4 }");
    }

    #[test]
    fn test_lane_shift_builtin() {
        check_ok("fn f(a: u8[64]) -> u8[64] { return lane_shr(a, 1) }");
    }

    // --- Error cases ---

    #[test]
    fn test_undefined_variable() {
        check_err(
            "fn f() -> i32[4] { return undefined_var }",
            "undefined variable: undefined_var",
        );
    }

    #[test]
    fn test_width_mismatch() {
        check_err(
            "fn f(a: i32[4], b: i32[8]) -> i32[4] { return a + b }",
            "width mismatch",
        );
    }

    #[test]
    fn test_sqrt_requires_float() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return sqrt(a) }",
            "sqrt requires float",
        );
    }

    #[test]
    fn test_wrong_arg_count() {
        check_err(
            "fn f(a: f32[4], b: f32[4]) -> f32[4] { return sqrt(a, b) }",
            "sqrt takes 1 argument",
        );
    }

    #[test]
    fn test_unknown_function() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return nonexistent(a) }",
            "unknown function: nonexistent",
        );
    }

    #[test]
    fn test_popcount_requires_bool() {
        check_err(
            "fn f(a: i32[4]) -> i32[1] { return popcount(a) }",
            "popcount requires bool",
        );
    }

    #[test]
    fn test_to_bitmask_requires_bool() {
        check_err(
            "fn f(a: i32[64]) -> u64[1] { return to_bitmask(a) }",
            "to_bitmask requires bool",
        );
    }

    #[test]
    fn test_float_literal_in_int_context() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return a + 1.0 }",
            "not compatible",
        );
    }

    #[test]
    fn test_bitwise_not_on_float() {
        check_err(
            "fn f(a: f32[4]) -> f32[4] { return ~a }",
            "bitwise NOT (~) requires integer",
        );
    }

    #[test]
    fn test_field_on_non_struct() {
        check_err(
            "fn f(a: i32[4]) -> i32[4] { return a.x }",
            "field access on non-struct",
        );
    }

    // --- Stream validation ---

    #[test]
    fn test_stream_valid() {
        check_ok(r#"
            fn f(input: ptr[u8]) -> i32[1] {
                stream chunk: u8[8] over input carry (count: i32[1] = 0) {
                    is_a = chunk == 'a'
                    carry count = count + popcount(is_a)
                }
                return count
            }
        "#);
    }

    #[test]
    fn test_stream_buffer_must_be_ptr() {
        check_err(
            "fn f(input: i32[4]) -> i32[1] { stream chunk: u8[8] over input { x = chunk } return 0 }",
            "must be a pointer",
        );
    }

    // --- json_stage1 as integration test ---

    #[test]
    fn test_json_stage1_typechecks() {
        let source = std::fs::read_to_string("examples/json_stage1.simd")
            .expect("cannot read json_stage1.simd");
        let items = parser::parse(&source);
        let result = typecheck(&items, &HashMap::new());
        match result {
            Ok(()) => {}
            Err(errs) => {
                for e in &errs {
                    eprintln!("  {}", e);
                }
                panic!("json_stage1.simd had {} type errors", errs.len());
            }
        }
    }

    // --- load_at ---

    #[test]
    fn test_load_at_ok() {
        check_ok("fn f(buf: ptr[u8], off: i32[1]) -> u8[1] { return load_at(buf, off) }");
    }

    #[test]
    fn test_load_at_non_ptr() {
        check_err(
            "fn f(a: i32[4], off: i32[1]) -> i32[1] { return load_at(a, off) }",
            "load_at first argument must be a pointer",
        );
    }

    // --- ptr_add ---

    #[test]
    fn test_ptr_add_ok() {
        check_ok("fn f(buf: ptr[u8], off: i32[1]) -> u8[1] { p2 = ptr_add(buf, off)\n return load_at(p2, 0) }");
    }

    #[test]
    fn test_ptr_add_non_ptr() {
        check_err(
            "fn f(a: i32[4], off: i32[1]) { x = ptr_add(a, off) }",
            "ptr_add first argument must be a pointer",
        );
    }

    // --- bswap ---

    #[test]
    fn test_bswap_u32() {
        check_ok("fn f(a: u32[4]) -> u32[4] { return bswap(a) }");
    }

    #[test]
    fn test_bswap_u64_scalar() {
        check_ok("fn f(a: u64[1]) -> u64[1] { return bswap(a) }");
    }

    #[test]
    fn test_bswap_bool_err() {
        check_err(
            "fn f(a: bool[4]) -> bool[4] { return bswap(a) }",
            "bswap requires u16/i16/u32/i32/u64/i64",
        );
    }

    #[test]
    fn test_bswap_u8_err() {
        check_err(
            "fn f(a: u8[4]) -> u8[4] { return bswap(a) }",
            "bswap requires u16/i16/u32/i32/u64/i64",
        );
    }

    // --- inter-function calls ---

    #[test]
    fn test_inter_fn_call_ok() {
        check_ok(r#"
            fn add_one(x: i32[4]) -> i32[4] { return x + 1 }
            fn f(a: i32[4]) -> i32[4] { return add_one(a) }
        "#);
    }

    #[test]
    fn test_inter_fn_call_wrong_arg_count() {
        check_err(
            r#"
            fn add(x: i32[4], y: i32[4]) -> i32[4] { return x + y }
            fn f(a: i32[4]) -> i32[4] { return add(a) }
            "#,
            "takes 2 arguments, got 1",
        );
    }

    #[test]
    fn test_inter_fn_call_with_ptr() {
        check_ok(r#"
            fn read_byte(buf: ptr[u8], off: i32[1]) -> u8[1] { return load_at(buf, off) }
            fn f(data: ptr[u8]) -> u8[1] { return read_byte(data, 0) }
        "#);
    }
}
