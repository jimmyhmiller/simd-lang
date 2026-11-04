//! Substitutes compile-time constants into a function body, so a width can also be used
//! as a number: `load_at[u8[N]](p, N - 16)`, or `n < REGISTER_BYTES`.

use crate::ast::*;
use std::collections::HashMap;

/// Replace each name in `values` with its value, in expressions and widths. Parsing has
/// already rejected variables that share a name with a compile-time width.
pub fn substitute_in_fn(def: &mut FnDef, values: &HashMap<String, u64>) {
    in_stmts(&mut def.body, values);
    crate::streams::expand_when_ready(def);
}

fn in_stmts(stmts: &mut [Stmt], values: &HashMap<String, u64>) {
    visit_stmts(
        stmts,
        &mut |expr| in_expr(expr, values),
        &mut |ty| in_type(ty, values),
        &mut |stream| {
            if let Some(bytes) = values.get("REGISTER_BYTES") {
                stream.register_bytes = Some(*bytes);
            }
        },
    );
}

/// Calls `f` on each statement's expressions, however nested.
pub fn for_each_expr(stmts: &mut [Stmt], f: &mut dyn FnMut(&mut Expr)) {
    visit_stmts(stmts, f, &mut |_| {}, &mut |_| {});
}

fn visit_stmts(
    stmts: &mut [Stmt],
    on_expr: &mut dyn FnMut(&mut Expr),
    on_type: &mut dyn FnMut(&mut Type),
    on_range_stream: &mut dyn FnMut(&mut RangeStream),
) {
    for stmt in stmts {
        match &mut stmt.kind {
            StmtKind::Assign { target, ty, value } => {
                if let AssignTarget::Scatter { base, index, mask } = target {
                    on_expr(base);
                    on_expr(index);
                    if let Some(mask) = mask {
                        on_expr(mask);
                    }
                }
                if let Some(ty) = ty {
                    on_type(ty);
                }
                on_expr(value);
            }
            StmtKind::Return(value) | StmtKind::Expr(value) => on_expr(value),
            StmtKind::ReturnVoid | StmtKind::Break => {}
            StmtKind::If { cond, then_body, else_body } => {
                on_expr(cond);
                visit_stmts(then_body, on_expr, on_type, on_range_stream);
                visit_stmts(else_body, on_expr, on_type, on_range_stream);
            }
            StmtKind::While { cond, body } => {
                on_expr(cond);
                visit_stmts(body, on_expr, on_type, on_range_stream);
            }
            StmtKind::Stream { chunk_ty, carry, body, carry_updates, .. } => {
                on_type(chunk_ty);
                for def in carry {
                    on_type(&mut def.ty);
                    on_expr(&mut def.init);
                }
                visit_stmts(body, on_expr, on_type, on_range_stream);
                for (_, update) in carry_updates {
                    on_expr(update);
                }
            }
            StmtKind::RangeStream(stream) => {
                on_type(&mut stream.chunk_ty);
                on_expr(&mut stream.start);
                on_expr(&mut stream.end);
                if let Some(until) = &mut stream.until {
                    on_expr(until);
                }
                for def in &mut stream.carry {
                    on_type(&mut def.ty);
                    on_expr(&mut def.init);
                }
                visit_stmts(&mut stream.body, on_expr, on_type, on_range_stream);
                for (_, update) in &mut stream.carry_updates {
                    on_expr(update);
                }
                if let Some(stopped) = &mut stream.stopped {
                    visit_stmts(stopped, on_expr, on_type, on_range_stream);
                }
                on_range_stream(stream);
            }
        }
    }
}

fn in_type(ty: &mut Type, values: &HashMap<String, u64>) {
    if let Some(width) = &mut ty.width {
        *width = width.substitute(values);
    }
}

fn in_expr(expr: &mut Expr, values: &HashMap<String, u64>) {
    walk_expr(expr, &mut |e| match e {
        Expr::Ident(name) => {
            if let Some(value) = values.get(name) {
                *e = Expr::IntLit(*value as i64);
            }
        }
        Expr::Instantiate { widths, .. } => {
            for width in widths {
                *width = width.substitute(values);
            }
        }
        Expr::StructLit { width, .. } => *width = width.substitute(values),
        Expr::Load { ty, .. } => in_type(ty, values),
        _ => {}
    });
}

/// Calls `f` on an expression and then on each expression inside it. A plain callee is a
/// function's name, not a value, so it isn't visited.
pub fn walk_expr(expr: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    f(expr);
    match expr {
        Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::Ident(_)
        | Expr::VecLit { .. }
        | Expr::Instantiate { .. } => {}
        Expr::Logical { lhs, rhs, .. } | Expr::BinOp { lhs, rhs, .. } => {
            walk_expr(lhs, f);
            walk_expr(rhs, f);
        }
        Expr::UnaryOp { operand, .. } | Expr::Reduction { operand, .. } => walk_expr(operand, f),
        // A comprehension only runs at compile time, so it has no run-time expressions.
        Expr::VecComp(_) => {}
        Expr::Index { index, .. } => walk_expr(index, f),
        Expr::SubSlice { start, end, .. } => {
            if let Some(start) = start {
                walk_expr(start, f);
            }
            if let Some(end) = end {
                walk_expr(end, f);
            }
        }
        Expr::Masked { mask, body, fallback } => {
            walk_expr(mask, f);
            walk_expr(body, f);
            if let Some(fallback) = fallback {
                walk_expr(fallback, f);
            }
        }
        Expr::Field { base, .. } => walk_expr(base, f),
        Expr::Call { func, args } => {
            if !matches!(func.as_ref(), Expr::Ident(_)) {
                walk_expr(func, f);
            }
            for arg in args {
                walk_expr(&mut arg.value, f);
            }
        }
        Expr::Scan { operand, seed, .. } => {
            walk_expr(operand, f);
            if let Some(seed) = seed {
                walk_expr(seed, f);
            }
        }
        Expr::Gather { base, index, mask } => {
            walk_expr(base, f);
            walk_expr(index, f);
            if let Some(mask) = mask {
                walk_expr(mask, f);
            }
        }
        Expr::StructLit { fields, .. } => {
            for (_, value) in fields {
                walk_expr(value, f);
            }
        }
        Expr::Load { ptr, offset, .. } => {
            walk_expr(ptr, f);
            if let Some(offset) = offset {
                walk_expr(offset, f);
            }
        }
    }
}

/// Copies `items` with `REGISTER_BYTES`, the width of one vector register, filled in.
pub fn with_register_bytes(items: &[Item], bytes: u64) -> Vec<Item> {
    let values = HashMap::from([("REGISTER_BYTES".to_string(), bytes)]);
    items
        .iter()
        .map(|item| {
            let mut item = item.clone();
            match &mut item {
                Item::Fn(def) => {
                    substitute_in_fn(def, &values);
                    for param in &mut def.params {
                        in_type(&mut param.ty, &values);
                    }
                    if let Some(ret) = &mut def.ret_ty {
                        in_type(ret, &values);
                    }
                }
                Item::Struct(def) => {
                    for field in &mut def.fields {
                        in_type(&mut field.ty, &values);
                    }
                }
                Item::Const(_) | Item::ComptimeFn(_) => {}
            }
            item
        })
        .collect()
}
