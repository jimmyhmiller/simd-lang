//! Compile-time tables. `comptime fn`s run here on plain integers, filling in table
//! comprehensions like `[u8: count_bits(i) for i in 0..256]`, and `T[i]` becomes a load of one
//! element of `T`. Nothing in this file reaches codegen: afterwards every table is a literal,
//! and the `comptime fn`s are gone.

use crate::ast::*;
use crate::comptime::{for_each_expr, walk_expr};
use crate::desugar::collect_assigned;
use std::collections::{HashMap, HashSet};

/// Enough for any table, and a clear error rather than a hang for a loop that never ends.
const MAX_STEPS: u64 = 100_000_000;
const MAX_CALL_DEPTH: u32 = 256;

pub fn resolve(items: Vec<Item>) -> Vec<Item> {
    let comptime_fns: HashMap<String, FnDef> = items
        .iter()
        .filter_map(|item| match item {
            Item::ComptimeFn(def) => Some((def.name.clone(), def.clone())),
            _ => None,
        })
        .collect();
    for def in comptime_fns.values() {
        check_signature(def);
    }

    let mut out = Vec::with_capacity(items.len());
    let mut tables: HashMap<String, (String, u64)> = HashMap::new();
    for item in &items {
        if let Item::Const(def) = item {
            tables.insert(def.name.clone(), (def.elem_type.clone(), def.row));
        }
    }
    for item in items {
        match item {
            Item::ComptimeFn(_) => {}
            Item::Const(mut def) => {
                if let Some(comprehension) = def.comprehension.take() {
                    def.values = evaluate(&comprehension, &comptime_fns, &format!("const '{}'", def.name));
                }
                check_values_fit(&def);
                if def.row > 1 && def.values.len() as u64 % def.row != 0 {
                    panic!(
                        "const '{}': {} values don't make whole rows of {}",
                        def.name,
                        def.values.len(),
                        def.row
                    );
                }
                out.push(Item::Const(def));
            }
            Item::Fn(mut def) => {
                resolve_in_fn(&mut def, &tables, &comptime_fns);
                out.push(Item::Fn(def));
            }
            other => out.push(other),
        }
    }
    out
}

fn resolve_in_fn(def: &mut FnDef, tables: &HashMap<String, (String, u64)>, comptime_fns: &HashMap<String, FnDef>) {
    // A parameter or local with a table's name hides that table.
    let mut hidden: HashSet<String> = def.params.iter().map(|p| p.name.clone()).collect();
    let mut assigned = Vec::new();
    collect_assigned(&def.body, &mut assigned);
    hidden.extend(assigned);

    let fn_name = def.name.clone();
    for_each_expr(&mut def.body, &mut |top| {
        walk_expr(top, &mut |expr| match expr {
            Expr::VecComp(comprehension) => {
                let values = evaluate(comprehension, comptime_fns, &format!("a vector literal in '{}'", fn_name));
                *expr = Expr::VecLit { elem_type: comprehension.elem_type.clone(), values };
            }
            Expr::Index { name, index } => {
                let table = tables.get(name.as_str()).filter(|_| !hidden.contains(name.as_str()));
                let Some((elem_type, row)) = table else {
                    panic!("in '{}': can't index '{}', since only const tables and slices can be indexed", fn_name, name);
                };
                // `load_at` counts in elements of the pointer's type, so a row is `row` of them.
                let index = std::mem::replace(index.as_mut(), Expr::IntLit(0));
                let offset = if *row == 1 {
                    index
                } else {
                    Expr::BinOp { op: BinOp::Mul, lhs: Box::new(index), rhs: Box::new(Expr::IntLit(*row as i64)) }
                };
                *expr = Expr::Load {
                    aligned: false,
                    ty: Type { name: elem_type.clone(), width: Some(Width::Fixed(*row)) },
                    ptr: Box::new(Expr::Ident(name.clone())),
                    offset: Some(Box::new(offset)),
                };
            }
            Expr::Call { func, .. } => {
                if let Expr::Ident(callee) | Expr::Instantiate { name: callee, .. } = func.as_ref() {
                    if comptime_fns.contains_key(callee) {
                        panic!(
                            "in '{}': '{}' is a comptime fn, so it can only be used to build a table",
                            fn_name, callee
                        );
                    }
                }
            }
            _ => {}
        });
    });
}

fn check_values_fit(def: &ConstDef) {
    let bits = match def.elem_type.as_str() {
        "i8" | "u8" => 8,
        "i16" | "u16" => 16,
        "i32" | "u32" => 32,
        _ => return,
    };
    if let Some(bad) = def.values.iter().find(|&&v| v < -(1i64 << (bits - 1)) || v >= (1i64 << bits)) {
        panic!("const '{}': literal {} doesn't fit in {}", def.name, bad, def.elem_type);
    }
}

fn check_signature(def: &FnDef) {
    let scalar = |ty: &Type| {
        matches!(ty.name.as_str(), "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool")
            && ty.width == Some(Width::Fixed(1))
    };
    if !def.comptime_params.is_empty() {
        panic!("comptime fn '{}' can't have compile-time parameters", def.name);
    }
    if let Some(param) = def.params.iter().find(|p| !scalar(&p.ty)) {
        panic!("comptime fn '{}': parameter '{}' must be a scalar like i64[1]", def.name, param.name);
    }
    if def.ret_ty.as_ref().is_some_and(|ret| !scalar(ret)) {
        panic!("comptime fn '{}' must return a scalar like i64[1]", def.name);
    }
}

fn evaluate(comprehension: &Comprehension, fns: &HashMap<String, FnDef>, what: &str) -> Vec<i64> {
    let bound = |expr: &Expr, side: &str| match expr.const_value() {
        Some(ConstValue::Int(n)) => n,
        _ => panic!("{}: the range's {} must be a number known while parsing", what, side),
    };
    let (start, end) = (bound(&comprehension.start, "start"), bound(&comprehension.end, "end"));
    if end <= start {
        panic!("{}: the range {}..{} is empty", what, start, end);
    }
    let mut eval = Eval { fns, what, steps: 0, depth: 0, emitted: None };
    let mut values = Vec::with_capacity((end - start) as usize);
    for i in start..end {
        let mut vars = HashMap::from([(comprehension.var.clone(), Val::Int(i))]);
        match comprehension.row {
            None => match eval.expr(&comprehension.value, &mut vars) {
                Val::Int(n) => values.push(n),
                other => panic!("{}: an element must be a number, got {:?} at {}", what, other, i),
            },
            Some(row) => {
                eval.emitted = Some(Vec::new());
                eval.expr(&comprehension.value, &mut vars);
                let mut emitted = eval.emitted.take().expect("set just above");
                if emitted.len() as u64 > row {
                    panic!("{}: row {} emitted {} values, more than its {}", what, i, emitted.len(), row);
                }
                if (emitted.len() as u64) < row {
                    let Some(fill) = comprehension.fill else {
                        panic!(
                            "{}: row {} emitted {} of its {} values; say `fill` for what goes in the rest",
                            what,
                            i,
                            emitted.len(),
                            row
                        );
                    };
                    emitted.resize(row as usize, fill);
                }
                values.extend(emitted);
            }
        }
    }
    values
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Val {
    Int(i64),
    Bool(bool),
    Unit,
}

enum Flow {
    Next,
    Return(Val),
    Break,
}

struct Eval<'a> {
    fns: &'a HashMap<String, FnDef>,
    what: &'a str,
    steps: u64,
    depth: u32,
    emitted: Option<Vec<i64>>,
}

impl Eval<'_> {
    fn fail(&self, message: String) -> ! {
        panic!("{}: {}", self.what, message)
    }

    fn int(&self, value: Val, context: &str) -> i64 {
        match value {
            Val::Int(n) => n,
            other => self.fail(format!("{} needs a number, got {:?}", context, other)),
        }
    }

    fn truth(&self, value: Val, context: &str) -> bool {
        match value {
            Val::Bool(b) => b,
            other => self.fail(format!("{} needs a bool, got {:?}", context, other)),
        }
    }

    fn stmts(&mut self, stmts: &[Stmt], vars: &mut HashMap<String, Val>) -> Flow {
        for stmt in stmts {
            self.steps += 1;
            if self.steps > MAX_STEPS {
                self.fail(format!("evaluation didn't finish in {} steps", MAX_STEPS));
            }
            match &stmt.kind {
                StmtKind::Assign { target: AssignTarget::Ident(name), value, .. } => {
                    let value = self.expr(value, vars);
                    vars.insert(name.clone(), value);
                }
                StmtKind::Return(value) => {
                    let value = self.expr(value, vars);
                    return Flow::Return(value);
                }
                StmtKind::ReturnVoid => return Flow::Return(Val::Unit),
                StmtKind::Break => return Flow::Break,
                StmtKind::Expr(value) => {
                    self.expr(value, vars);
                }
                StmtKind::If { cond, then_body, else_body } => {
                    let taken = self.expr(cond, vars);
                    let body = if self.truth(taken, "`if`") { then_body } else { else_body };
                    match self.stmts(body, vars) {
                        Flow::Next => {}
                        other => return other,
                    }
                }
                StmtKind::While { cond, body } => loop {
                    self.steps += 1;
                    if self.steps > MAX_STEPS {
                        self.fail(format!("evaluation didn't finish in {} steps", MAX_STEPS));
                    }
                    let go = self.expr(cond, vars);
                    if !self.truth(go, "`while`") {
                        break;
                    }
                    match self.stmts(body, vars) {
                        Flow::Next => {}
                        Flow::Break => break,
                        Flow::Return(value) => return Flow::Return(value),
                    }
                },
                _ => self.fail(format!(
                    "at {}: a comptime fn can only use scalar assignments, `if`, `while`, `break` and `return`",
                    stmt.span
                )),
            }
        }
        Flow::Next
    }

    fn expr(&mut self, expr: &Expr, vars: &mut HashMap<String, Val>) -> Val {
        match expr {
            Expr::IntLit(n) => Val::Int(*n),
            Expr::CharLit(c) => Val::Int(*c as i64),
            Expr::BoolLit(b) => Val::Bool(*b),
            Expr::Ident(name) => match vars.get(name) {
                Some(value) => *value,
                None => self.fail(format!("undefined variable: {}", name)),
            },
            Expr::Logical { op, lhs, rhs } => {
                let left = self.expr(lhs, vars);
                let left = self.truth(left, "`&&` or `||`");
                match (op, left) {
                    (LogicalOp::And, false) => Val::Bool(false),
                    (LogicalOp::Or, true) => Val::Bool(true),
                    _ => {
                        let right = self.expr(rhs, vars);
                        Val::Bool(self.truth(right, "`&&` or `||`"))
                    }
                }
            }
            Expr::UnaryOp { op, operand } => {
                let value = self.expr(operand, vars);
                match (op, value) {
                    (UnaryOp::Neg, Val::Int(n)) => Val::Int(
                        n.checked_neg().unwrap_or_else(|| self.fail(format!("-{} overflows", n))),
                    ),
                    (UnaryOp::Not, Val::Int(n)) => Val::Int(!n),
                    (UnaryOp::Not, Val::Bool(b)) => Val::Bool(!b),
                    (op, other) => self.fail(format!("{:?} doesn't apply to {:?}", op, other)),
                }
            }
            Expr::BinOp { op, lhs, rhs } => {
                let left = self.expr(lhs, vars);
                let right = self.expr(rhs, vars);
                self.binary(*op, left, right)
            }
            Expr::Call { func, args } => {
                let Expr::Ident(name) = func.as_ref() else {
                    self.fail("a comptime fn can only call comptime fns by name".to_string());
                };
                let values: Vec<Val> = args.iter().map(|arg| self.expr(&arg.value, vars)).collect();
                self.call(name, values)
            }
            other => self.fail(format!("a comptime fn can only compute with scalars, not {:?}", other)),
        }
    }

    fn binary(&self, op: BinOp, left: Val, right: Val) -> Val {
        if let (Val::Bool(a), Val::Bool(b)) = (left, right) {
            return match op {
                BinOp::And => Val::Bool(a & b),
                BinOp::Or => Val::Bool(a | b),
                BinOp::Xor | BinOp::NotEq => Val::Bool(a ^ b),
                BinOp::EqEq => Val::Bool(a == b),
                other => self.fail(format!("{:?} doesn't apply to bools", other)),
            };
        }
        let (a, b) = (self.int(left, "arithmetic"), self.int(right, "arithmetic"));
        let checked = |result: Option<i64>| {
            Val::Int(result.unwrap_or_else(|| self.fail(format!("{:?} of {} and {} overflows", op, a, b))))
        };
        match op {
            BinOp::Add => checked(a.checked_add(b)),
            BinOp::Sub => checked(a.checked_sub(b)),
            BinOp::Mul => checked(a.checked_mul(b)),
            BinOp::Div => checked(a.checked_div(b)),
            BinOp::And => Val::Int(a & b),
            BinOp::Or => Val::Int(a | b),
            BinOp::Xor => Val::Int(a ^ b),
            BinOp::BitShl => checked(u32::try_from(b).ok().and_then(|s| a.checked_shl(s)).filter(|r| r >> b == a)),
            BinOp::BitShr => checked(u32::try_from(b).ok().and_then(|s| a.checked_shr(s))),
            BinOp::Gt => Val::Bool(a > b),
            BinOp::Lt => Val::Bool(a < b),
            BinOp::GtEq => Val::Bool(a >= b),
            BinOp::LtEq => Val::Bool(a <= b),
            BinOp::EqEq => Val::Bool(a == b),
            BinOp::NotEq => Val::Bool(a != b),
        }
    }

    fn call(&mut self, name: &str, args: Vec<Val>) -> Val {
        if name == "emit" {
            let [value] = args.as_slice() else { self.fail("emit takes one value".to_string()) };
            let value = self.int(*value, "emit");
            match &mut self.emitted {
                Some(row) => row.push(value),
                None => self.fail("emit only works in a table with rows, like [u8[16]: ...]".to_string()),
            }
            return Val::Unit;
        }
        if let "ctz" | "clear_lowest_bit" | "to_i64" = name {
            let [value] = args.as_slice() else { self.fail(format!("{} takes one value", name)) };
            let n = self.int(*value, name);
            return Val::Int(match name {
                "ctz" => n.trailing_zeros() as i64,
                "clear_lowest_bit" => n & n.wrapping_sub(1),
                _ => n,
            });
        }
        let fns = self.fns;
        let Some(def) = fns.get(name) else {
            self.fail(format!("'{}' isn't a comptime fn", name));
        };
        if def.params.len() != args.len() {
            self.fail(format!("'{}' takes {} arguments, got {}", name, def.params.len(), args.len()));
        }
        self.depth += 1;
        if self.depth > MAX_CALL_DEPTH {
            self.fail(format!("calls nested more than {} deep", MAX_CALL_DEPTH));
        }
        let mut vars: HashMap<String, Val> =
            def.params.iter().map(|p| p.name.clone()).zip(args).collect();
        let result = match self.stmts(&def.body, &mut vars) {
            Flow::Return(value) => value,
            Flow::Next => Val::Unit,
            Flow::Break => self.fail(format!("`break` outside of a loop in '{}'", name)),
        };
        self.depth -= 1;
        if def.ret_ty.is_some() && result == Val::Unit {
            self.fail(format!("'{}' doesn't return a value on every path", name));
        }
        result
    }
}
