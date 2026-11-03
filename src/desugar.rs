//! Rewrites early `return` and `break` into structured control flow.
//!
//! The IR only has structured `if` and `while`, so an escape becomes flags. Code after
//! an `if` that may escape moves into a guard, or into the other branch when the `if`
//! always escapes on one side:
//!
//! ```text
//! while c {                     __returned: bool[1] = false
//!     if d { return x }         __ret: T = 0
//!     a()                       while ~__returned && c {
//! }                                 if d { __ret = x; __returned = true } else { a() }
//! return y                      }
//!                               if ~__returned { __ret = y; __returned = true }
//!                               return __ret
//! ```
//!
//! LLVM's jump threading removes the flags, leaving the direct branches. Functions
//! whose only `return` is their last statement aren't touched.

use crate::ast::*;
use std::collections::HashMap;

const RETURNED: &str = "__returned";
const RET_VALUE: &str = "__ret";

/// Reject names that would otherwise change a program's meaning without saying so.
fn check_names(items: &[Item]) {
    let mut seen = std::collections::HashSet::new();
    for item in items {
        let (kind, name) = match item {
            Item::Fn(def) => ("function", &def.name),
            Item::Struct(def) => ("struct", &def.name),
            Item::Const(def) => ("const", &def.name),
        };
        if !seen.insert(name.clone()) {
            panic!("{} '{}' is defined twice", kind, name);
        }
        if name == "REGISTER_BYTES" {
            panic!("REGISTER_BYTES is reserved");
        }
        match item {
            Item::Fn(def) => {
                if BUILTINS.contains(&name.as_str()) {
                    panic!("function '{}' has the same name as a builtin", name);
                }
                let mut params = std::collections::HashSet::new();
                for param in &def.params {
                    if !params.insert(param.name.clone()) {
                        panic!("function '{}' has two parameters named '{}'", name, param.name);
                    }
                }
                let mut assigned = Vec::new();
                collect_assigned(&def.body, &mut assigned);
                assigned.extend(def.params.iter().map(|p| p.name.clone()));
                for local in &assigned {
                    if local == "REGISTER_BYTES" || def.comptime_params.iter().any(|c| &c.name == local) {
                        panic!("in '{}': '{}' is a compile-time width and can't also be a variable", name, local);
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn collect_assigned(stmts: &[Stmt], out: &mut Vec<String>) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Assign { target: AssignTarget::Ident(name), .. } => out.push(name.clone()),
            StmtKind::Assign { target: AssignTarget::Destructure(names), .. } => out.extend(names.iter().cloned()),
            StmtKind::If { then_body, else_body, .. } => {
                collect_assigned(then_body, out);
                collect_assigned(else_body, out);
            }
            StmtKind::While { body, .. } => collect_assigned(body, out),
            StmtKind::Stream { chunk_name, carry, body, .. } => {
                out.push(chunk_name.clone());
                out.extend(carry.iter().map(|c| c.name.clone()));
                collect_assigned(body, out);
            }
            _ => {}
        }
    }
}

pub fn desugar_items(items: Vec<Item>) -> Vec<Item> {
    check_names(&items);
    let structs: HashMap<String, StructDef> = items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(def) => Some((def.name.clone(), def.clone())),
            _ => None,
        })
        .collect();
    items
        .into_iter()
        .map(|item| match item {
            Item::Fn(mut def) => {
                desugar_fn(&mut def, &structs);
                Item::Fn(def)
            }
            other => other,
        })
        .collect()
}

fn desugar_fn(def: &mut FnDef, structs: &HashMap<String, StructDef>) {
    let last_is_plain_return = matches!(
        def.body.last().map(|s| &s.kind),
        Some(StmtKind::Return(_) | StmtKind::ReturnVoid)
    );
    let checked = if last_is_plain_return { def.body.len() - 1 } else { def.body.len() };
    if !def.body[..checked].iter().any(has_escape) {
        return;
    }
    let span = def.body[0].span;

    let mut lowering = Lowering {
        fn_name: def.name.clone(),
        ret_ty: def.ret_ty.clone(),
        next_flag: 0,
    };
    let lowered = lowering.lower(&def.body, None);

    let mut body = vec![assign_flag(RETURNED, false, span)];
    if let Some(ret) = &def.ret_ty {
        if !always_escapes(&def.body) {
            panic!("function '{}' doesn't return a value on every path", def.name);
        }
        body.push(zero_ret_value(ret, &def.name, structs, span));
    }
    body.extend(lowered);
    if def.ret_ty.is_some() {
        body.push(Stmt { kind: StmtKind::Return(Expr::Ident(RET_VALUE.to_string())), span });
    }
    def.body = body;
}

/// `__ret` has to exist before any branch assigns it. This value is never observed.
fn zero_ret_value(ret: &Type, fn_name: &str, structs: &HashMap<String, StructDef>, span: Span) -> Stmt {
    let zero_of = |type_name: &str| match type_name {
        "f32" | "f64" => Expr::FloatLit(0.0),
        "bool" => Expr::BoolLit(false),
        _ => Expr::IntLit(0),
    };
    let (ty, value) = if ret.name == "ptr" {
        panic!("function '{}' returns a pointer, which early return doesn't support", fn_name);
    } else if let Some(def) = structs.get(&ret.name) {
        let width = ret.width.clone().unwrap_or(Width::Fixed(1));
        let fields = def.fields.iter().map(|f| (f.name.clone(), zero_of(&f.ty.name))).collect();
        (None, Expr::StructLit { name: ret.name.clone(), width, fields })
    } else {
        (Some(ret.clone()), zero_of(&ret.name))
    };
    Stmt {
        kind: StmtKind::Assign { target: AssignTarget::Ident(RET_VALUE.to_string()), ty, value },
        span,
    }
}

fn bool_type() -> Type {
    Type { name: "bool".to_string(), width: Some(Width::Fixed(1)) }
}

fn assign_flag(name: &str, value: bool, span: Span) -> Stmt {
    Stmt {
        kind: StmtKind::Assign {
            target: AssignTarget::Ident(name.to_string()),
            ty: Some(bool_type()),
            value: Expr::BoolLit(value),
        },
        span,
    }
}

fn not(name: &str) -> Expr {
    Expr::UnaryOp { op: UnaryOp::Not, operand: Box::new(Expr::Ident(name.to_string())) }
}

/// `~flag1 && ~flag2 && last`, evaluating `last` only if no flag is set.
fn guarded(flags: &[&str], last: Option<Expr>) -> Expr {
    let mut terms: Vec<Expr> = flags.iter().map(|f| not(f)).collect();
    terms.extend(last);
    let mut iter = terms.into_iter().rev();
    let mut expr = iter.next().expect("a guard needs at least one term");
    for term in iter {
        expr = Expr::Logical { op: LogicalOp::And, lhs: Box::new(term), rhs: Box::new(expr) };
    }
    expr
}

/// Whether the statement contains any `return` or `break`, however nested.
fn has_escape(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Return(_) | StmtKind::ReturnVoid | StmtKind::Break => true,
        StmtKind::If { then_body, else_body, .. } => {
            then_body.iter().any(has_escape) || else_body.iter().any(has_escape)
        }
        StmtKind::While { body, .. } | StmtKind::Stream { body, .. } => body.iter().any(has_escape),
        _ => false,
    }
}

/// Whether the statement can leave the function.
fn may_return(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Return(_) | StmtKind::ReturnVoid => true,
        StmtKind::If { then_body, else_body, .. } => {
            then_body.iter().any(may_return) || else_body.iter().any(may_return)
        }
        StmtKind::While { body, .. } | StmtKind::Stream { body, .. } => body.iter().any(may_return),
        _ => false,
    }
}

/// Whether the statement can leave the loop it's directly in.
fn may_break(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Break => true,
        StmtKind::If { then_body, else_body, .. } => {
            then_body.iter().any(may_break) || else_body.iter().any(may_break)
        }
        _ => false,
    }
}

/// Whether the statements never fall through to whatever follows them.
fn always_escapes(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_always_escapes)
}

fn stmt_always_escapes(stmt: &Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Return(_) | StmtKind::ReturnVoid | StmtKind::Break => true,
        StmtKind::If { then_body, else_body, .. } => {
            always_escapes(then_body) && always_escapes(else_body)
        }
        // A loop that can't finish and can't be broken out of only leaves by returning.
        StmtKind::While { cond, body } => {
            cond.const_value() == Some(ConstValue::Bool(true)) && !body.iter().any(may_break)
        }
        _ => false,
    }
}

struct Lowering {
    fn_name: String,
    ret_ty: Option<Type>,
    next_flag: usize,
}

impl Lowering {
    /// Rewrite statements, with `break_flag` set by a `break` directly in them.
    fn lower(&mut self, stmts: &[Stmt], break_flag: Option<&str>) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(stmts.len());
        for (i, stmt) in stmts.iter().enumerate() {
            if !has_escape(stmt) {
                out.push(stmt.clone());
                continue;
            }
            let rest = &stmts[i + 1..];
            let span = stmt.span;
            if matches!(stmt.kind, StmtKind::Break) && break_flag.is_none() {
                panic!("at {}: break outside of a loop", span);
            }
            if stmt_always_escapes(stmt) && !rest.is_empty() {
                panic!("at {}: unreachable code", rest[0].span);
            }
            match &stmt.kind {
                StmtKind::Return(value) => {
                    let Some(ret) = self.ret_ty.clone() else {
                        panic!("at {}: function '{}' has no return type, so it can't return a value", span, self.fn_name);
                    };
                    let ty = if is_struct_type(&ret) { None } else { Some(ret) };
                    out.push(Stmt {
                        kind: StmtKind::Assign {
                            target: AssignTarget::Ident(RET_VALUE.to_string()),
                            ty,
                            value: value.clone(),
                        },
                        span,
                    });
                    out.push(assign_flag(RETURNED, true, span));
                    return out;
                }
                StmtKind::ReturnVoid => {
                    if self.ret_ty.is_some() {
                        panic!("at {}: function '{}' must return a value", span, self.fn_name);
                    }
                    out.push(assign_flag(RETURNED, true, span));
                    return out;
                }
                StmtKind::Break => {
                    let Some(flag) = break_flag else {
                        panic!("at {}: break outside of a loop", span);
                    };
                    out.push(assign_flag(flag, true, span));
                    return out;
                }
                StmtKind::If { cond, then_body, else_body } => {
                    // If one side always escapes, what follows the `if` only runs after the
                    // other side, so it joins that side's statements. Lowering them together
                    // matters: that side may itself escape on only some paths.
                    let (then_escapes, else_escapes) = (always_escapes(then_body), always_escapes(else_body));
                    let then_stmts = if else_escapes && !then_escapes {
                        [then_body.as_slice(), rest].concat()
                    } else {
                        then_body.clone()
                    };
                    let else_stmts = if then_escapes && !else_escapes {
                        [else_body.as_slice(), rest].concat()
                    } else {
                        else_body.clone()
                    };
                    let then_lowered = self.lower(&then_stmts, break_flag);
                    let else_lowered = self.lower(&else_stmts, break_flag);
                    out.push(Stmt {
                        kind: StmtKind::If {
                            cond: cond.clone(),
                            then_body: then_lowered,
                            else_body: else_lowered,
                        },
                        span,
                    });
                    if then_escapes || else_escapes {
                        return out;
                    }
                    let mut flags: Vec<&str> = Vec::new();
                    if may_return(stmt) {
                        flags.push(RETURNED);
                    }
                    if may_break(stmt) {
                        flags.push(break_flag.expect("a break was lowered, so there is a loop"));
                    }
                    if !flags.is_empty() {
                        if !rest.is_empty() {
                            let then_body = self.lower(rest, break_flag);
                            out.push(Stmt {
                                kind: StmtKind::If { cond: guarded(&flags, None), then_body, else_body: vec![] },
                                span,
                            });
                        }
                        return out;
                    }
                }
                StmtKind::While { cond, body } => {
                    let breaks = body.iter().any(may_break);
                    let returns = body.iter().any(may_return);
                    let flag = breaks.then(|| {
                        self.next_flag += 1;
                        format!("__break_{}", self.next_flag)
                    });
                    if let Some(flag) = &flag {
                        out.push(assign_flag(flag, false, span));
                    }
                    let lowered = self.lower(body, flag.as_deref());
                    let mut flags: Vec<&str> = Vec::new();
                    if returns {
                        flags.push(RETURNED);
                    }
                    if let Some(flag) = &flag {
                        flags.push(flag);
                    }
                    out.push(Stmt {
                        kind: StmtKind::While { cond: guarded(&flags, Some(cond.clone())), body: lowered },
                        span,
                    });
                    if returns {
                        if !rest.is_empty() {
                            let then_body = self.lower(rest, break_flag);
                            out.push(Stmt {
                                kind: StmtKind::If { cond: guarded(&[RETURNED], None), then_body, else_body: vec![] },
                                span,
                            });
                        }
                        return out;
                    }
                }
                StmtKind::Stream { chunk_name, chunk_ty, buffer, carry, body, carry_updates } => {
                    if body.iter().any(may_return) {
                        panic!("at {}: return inside a stream isn't supported", span);
                    }
                    if body.iter().any(may_break) {
                        panic!("at {}: break out of a stream isn't supported", span);
                    }
                    out.push(Stmt {
                        kind: StmtKind::Stream {
                            chunk_name: chunk_name.clone(),
                            chunk_ty: chunk_ty.clone(),
                            buffer: buffer.clone(),
                            carry: carry.clone(),
                            body: self.lower(body, None),
                            carry_updates: carry_updates.clone(),
                        },
                        span,
                    });
                }
                other => unreachable!("{:?} has no nested statements", other),
            }
        }
        out
    }
}

fn is_struct_type(ty: &Type) -> bool {
    !matches!(
        ty.name.as_str(),
        "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool" | "ptr"
    )
}
