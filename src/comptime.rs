//! Substitutes compile-time constants into a function body, so a width can also be used
//! as a number: `load_at[u8[N]](p, N - 16)`, or `n < REGISTER_BYTES`.

use crate::ast::*;
use std::collections::HashMap;

/// Replace each name in `values` with its value, in expressions and widths. Parsing has
/// already rejected variables that share a name with a compile-time width.
pub fn substitute_in_fn(def: &mut FnDef, values: &HashMap<String, u64>) {
    for stmt in &mut def.body {
        in_stmt(stmt, values);
    }
}

fn in_stmts(stmts: &mut [Stmt], values: &HashMap<String, u64>) {
    for stmt in stmts {
        in_stmt(stmt, values);
    }
}

fn in_stmt(stmt: &mut Stmt, values: &HashMap<String, u64>) {
    match &mut stmt.kind {
        StmtKind::Assign { target, ty, value } => {
            if let AssignTarget::Scatter { base, index, mask } = target {
                in_expr(base, values);
                in_expr(index, values);
                if let Some(mask) = mask {
                    in_expr(mask, values);
                }
            }
            if let Some(ty) = ty {
                in_type(ty, values);
            }
            in_expr(value, values);
        }
        StmtKind::Return(value) | StmtKind::Expr(value) => in_expr(value, values),
        StmtKind::ReturnVoid | StmtKind::Break => {}
        StmtKind::If { cond, then_body, else_body } => {
            in_expr(cond, values);
            in_stmts(then_body, values);
            in_stmts(else_body, values);
        }
        StmtKind::While { cond, body } => {
            in_expr(cond, values);
            in_stmts(body, values);
        }
        StmtKind::Stream { chunk_ty, carry, body, carry_updates, .. } => {
            in_type(chunk_ty, values);
            for def in carry {
                in_type(&mut def.ty, values);
                in_expr(&mut def.init, values);
            }
            in_stmts(body, values);
            for (_, update) in carry_updates {
                in_expr(update, values);
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
    match expr {
        Expr::Ident(name) => {
            if let Some(value) = values.get(name) {
                *expr = Expr::IntLit(*value as i64);
            }
        }
        Expr::IntLit(_) | Expr::FloatLit(_) | Expr::BoolLit(_) | Expr::CharLit(_) | Expr::VecLit { .. } => {}
        Expr::Instantiate { widths, .. } => {
            for width in widths {
                *width = width.substitute(values);
            }
        }
        Expr::Logical { lhs, rhs, .. } | Expr::BinOp { lhs, rhs, .. } => {
            in_expr(lhs, values);
            in_expr(rhs, values);
        }
        Expr::UnaryOp { operand, .. } | Expr::Reduction { operand, .. } => in_expr(operand, values),
        Expr::Masked { mask, body, fallback } => {
            in_expr(mask, values);
            in_expr(body, values);
            if let Some(fallback) = fallback {
                in_expr(fallback, values);
            }
        }
        Expr::Field { base, .. } => in_expr(base, values),
        Expr::Call { func, args } => {
            // A plain callee is a function's name, not a value.
            if !matches!(func.as_ref(), Expr::Ident(_)) {
                in_expr(func, values);
            }
            for arg in args {
                in_expr(&mut arg.value, values);
            }
        }
        Expr::Scan { operand, seed, .. } => {
            in_expr(operand, values);
            if let Some(seed) = seed {
                in_expr(seed, values);
            }
        }
        Expr::Gather { base, index, mask } => {
            in_expr(base, values);
            in_expr(index, values);
            if let Some(mask) = mask {
                in_expr(mask, values);
            }
        }
        Expr::StructLit { width, fields, .. } => {
            *width = width.substitute(values);
            for (_, value) in fields {
                in_expr(value, values);
            }
        }
        Expr::Load { ty, ptr, offset, .. } => {
            in_type(ty, values);
            in_expr(ptr, values);
            if let Some(offset) = offset {
                in_expr(offset, values);
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
                Item::Const(_) => {}
            }
            item
        })
        .collect()
}
