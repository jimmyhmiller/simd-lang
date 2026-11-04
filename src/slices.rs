//! Lowers slices to what you'd write by hand: `s: slice[u8]` is the pointer `s` plus the length
//! `s__len`, in that order, so an `extern fn` taking a slice has the C signature
//! `(const u8 *s, int64_t s_len)`, and using one costs nothing.
//!
//! Slices don't add bounds checks. `s[i]` reads wherever `i` says, and a stream over a slice
//! stays in range because it's written in terms of the length.

use crate::ast::*;
use crate::comptime::{for_each_expr, walk_expr};
use std::collections::HashMap;

fn len_name(slice: &str) -> String {
    format!("{}__len", slice)
}

fn elem_of(ty: &Type) -> String {
    match &ty.width {
        Some(Width::Param(elem)) => elem.clone(),
        other => panic!("a slice's element must be a type like slice[u8], got {:?}", other),
    }
}

pub fn lower(items: Vec<Item>) -> Vec<Item> {
    let mut signatures: HashMap<String, Vec<bool>> = HashMap::new();
    for item in &items {
        match item {
            Item::Fn(def) => {
                let flags = def.params.iter().map(|p| p.ty.name == "slice").collect();
                signatures.insert(def.name.clone(), flags);
                if def.ret_ty.as_ref().is_some_and(|ret| ret.name == "slice") {
                    panic!("function '{}' can't return a slice yet", def.name);
                }
            }
            Item::Struct(def) => {
                if let Some(field) = def.fields.iter().find(|f| f.ty.name == "slice") {
                    panic!("struct '{}' can't have a slice field yet: '{}'", def.name, field.name);
                }
            }
            _ => {}
        }
    }
    items
        .into_iter()
        .map(|item| match item {
            Item::Fn(mut def) => {
                lower_fn(&mut def, &signatures);
                Item::Fn(def)
            }
            other => other,
        })
        .collect()
}

struct Lowering<'a> {
    fn_name: String,
    /// Each slice in this function, with its element type.
    slices: HashMap<String, String>,
    signatures: &'a HashMap<String, Vec<bool>>,
    next_temp: usize,
}

fn lower_fn(def: &mut FnDef, signatures: &HashMap<String, Vec<bool>>) {
    let mut slices: HashMap<String, String> =
        def.params.iter().filter(|p| p.ty.name == "slice").map(|p| (p.name.clone(), elem_of(&p.ty))).collect();
    // A local is a slice if it's ever assigned one, possibly through another local.
    loop {
        let before = slices.len();
        find_slice_locals(&def.body, &mut slices);
        if slices.len() == before {
            break;
        }
    }

    let mut lowering = Lowering { fn_name: def.name.clone(), slices, signatures, next_temp: 0 };
    def.body = lowering.stmts(std::mem::take(&mut def.body));

    let mut params = Vec::with_capacity(def.params.len());
    for param in std::mem::take(&mut def.params) {
        if param.ty.name == "slice" {
            let len = Param {
                name: len_name(&param.name),
                ty: Type { name: "i64".to_string(), width: Some(Width::Fixed(1)) },
            };
            params.push(Param { name: param.name, ty: Type { name: "ptr".to_string(), width: param.ty.width } });
            params.push(len);
        } else {
            params.push(param);
        }
    }
    def.params = params;
}

fn find_slice_locals(stmts: &[Stmt], slices: &mut HashMap<String, String>) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::Assign { target: AssignTarget::Ident(name), value, .. } => {
                let source = match value {
                    Expr::SubSlice { name: source, .. } | Expr::Ident(source) => source,
                    _ => continue,
                };
                if let Some(elem) = slices.get(source).cloned() {
                    slices.insert(name.clone(), elem);
                }
            }
            StmtKind::If { then_body, else_body, .. } => {
                find_slice_locals(then_body, slices);
                find_slice_locals(else_body, slices);
            }
            StmtKind::While { body, .. } | StmtKind::Stream { body, .. } => find_slice_locals(body, slices),
            StmtKind::RangeStream(stream) => {
                find_slice_locals(&stream.body, slices);
                if let Some(stopped) = &stream.stopped {
                    find_slice_locals(stopped, slices);
                }
            }
            _ => {}
        }
    }
}

fn assign(name: &str, ty: Option<Type>, value: Expr, span: Span) -> Stmt {
    Stmt { kind: StmtKind::Assign { target: AssignTarget::Ident(name.to_string()), ty, value }, span }
}

fn scalar() -> Type {
    Type { name: "i64".to_string(), width: Some(Width::Fixed(1)) }
}

fn ident(name: &str) -> Expr {
    Expr::Ident(name.to_string())
}

fn sub(a: Expr, b: Expr) -> Expr {
    Expr::BinOp { op: BinOp::Sub, lhs: Box::new(a), rhs: Box::new(b) }
}

impl Lowering<'_> {
    fn fail(&self, message: String) -> ! {
        panic!("in '{}': {}", self.fn_name, message)
    }

    fn stmts(&mut self, stmts: Vec<Stmt>) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(stmts.len());
        for stmt in stmts {
            self.stmt(stmt, &mut out);
        }
        out
    }

    fn stmt(&mut self, stmt: Stmt, out: &mut Vec<Stmt>) {
        let span = stmt.span;
        match stmt.kind {
            StmtKind::Assign { target: AssignTarget::Ident(name), value: Expr::SubSlice { name: source, start, end }, .. } => {
                // Bound first, since assigning a slice to itself changes what these would read.
                self.next_temp += 1;
                let start_name = format!("__slice_start_{}", self.next_temp);
                let end_name = format!("__slice_end_{}", self.next_temp);
                let (mut start, mut end) = self.bounds(&source, start, end);
                self.expr(&mut start);
                self.expr(&mut end);
                out.push(assign(&start_name, Some(scalar()), start, span));
                out.push(assign(&end_name, Some(scalar()), end, span));
                let advanced = Expr::Call {
                    func: Box::new(ident("ptr_add")),
                    args: vec![
                        CallArg { name: None, value: ident(&source) },
                        CallArg { name: None, value: ident(&start_name) },
                    ],
                };
                out.push(assign(&name, None, advanced, span));
                out.push(assign(&len_name(&name), Some(scalar()), sub(ident(&end_name), ident(&start_name)), span));
            }
            StmtKind::Assign { target: AssignTarget::Ident(name), value: Expr::Ident(source), .. }
                if self.slices.contains_key(&source) =>
            {
                out.push(assign(&name, None, ident(&source), span));
                out.push(assign(&len_name(&name), Some(scalar()), ident(&len_name(&source)), span));
            }
            StmtKind::Assign { target: AssignTarget::Ident(name), .. } if self.slices.contains_key(&name) => {
                self.fail(format!("at {}: '{}' is a slice, so it can only be assigned a slice", span, name));
            }
            StmtKind::If { cond, then_body, else_body } => {
                let mut cond = cond;
                self.expr(&mut cond);
                let kind = StmtKind::If { cond, then_body: self.stmts(then_body), else_body: self.stmts(else_body) };
                out.push(Stmt { kind, span });
            }
            StmtKind::While { cond, body } => {
                let mut cond = cond;
                self.expr(&mut cond);
                let kind = StmtKind::While { cond, body: self.stmts(body) };
                out.push(Stmt { kind, span });
            }
            StmtKind::Stream { chunk_name, chunk_ty, buffer, mut carry, body, mut carry_updates } => {
                for def in &mut carry {
                    self.expr(&mut def.init);
                }
                for (_, update) in &mut carry_updates {
                    self.expr(update);
                }
                let body = self.stmts(body);
                let kind = StmtKind::Stream { chunk_name, chunk_ty, buffer, carry, body, carry_updates };
                out.push(Stmt { kind, span });
            }
            StmtKind::RangeStream(mut stream) => {
                let is_own_length = matches!(&stream.end, Expr::Field { base, field }
                    if field == "len" && matches!(base.as_ref(), Expr::Ident(b) if *b == stream.buffer));
                if is_own_length && !self.slices.contains_key(&stream.buffer) {
                    self.fail(format!(
                        "at {}: '{}' isn't a slice, so it has no length; write its range out, like over {}[0..n]",
                        span, stream.buffer, stream.buffer
                    ));
                }
                self.expr(&mut stream.start);
                self.expr(&mut stream.end);
                if let Some(until) = &mut stream.until {
                    self.expr(until);
                }
                for def in &mut stream.carry {
                    self.expr(&mut def.init);
                }
                stream.body = self.stmts(std::mem::take(&mut stream.body));
                for (_, update) in &mut stream.carry_updates {
                    self.expr(update);
                }
                stream.stopped = stream.stopped.take().map(|stopped| self.stmts(stopped));
                out.push(Stmt { kind: StmtKind::RangeStream(stream), span });
            }
            mut kind => {
                let mut one = [Stmt { kind: std::mem::replace(&mut kind, StmtKind::ReturnVoid), span }];
                for_each_expr(&mut one, &mut |expr| self.expr(expr));
                let [stmt] = one;
                out.push(stmt);
            }
        }
    }

    /// The range of `source[start..end]`, where a missing end means its length.
    fn bounds(&self, source: &str, start: Option<Box<Expr>>, end: Option<Box<Expr>>) -> (Expr, Expr) {
        if !self.slices.contains_key(source) {
            self.fail(format!("only a slice can be sliced, and '{}' isn't one", source));
        }
        let start = start.map_or(Expr::IntLit(0), |e| *e);
        let end = end.map_or_else(|| ident(&len_name(source)), |e| *e);
        (start, end)
    }

    fn expr(&self, top: &mut Expr) {
        // Each expression is rewritten before the walk goes into it, so a call's new arguments
        // get the same treatment as any other expression.
        walk_expr(top, &mut |expr| self.lower_one(expr));
    }

    fn lower_one(&self, expr: &mut Expr) {
        match expr {
            Expr::Field { base, field } => {
                let Expr::Ident(name) = base.as_ref() else { return };
                if !self.slices.contains_key(name) {
                    return;
                }
                let replacement = match field.as_str() {
                    "len" => ident(&len_name(name)),
                    "ptr" => ident(name),
                    other => self.fail(format!("a slice has `.len` and `.ptr`, not `.{}`", other)),
                };
                *expr = replacement;
            }
            Expr::Index { name, index } => {
                if let Some(elem) = self.slices.get(name.as_str()) {
                    let offset = std::mem::replace(index.as_mut(), Expr::IntLit(0));
                    *expr = Expr::Load {
                        aligned: false,
                        ty: Type { name: elem.clone(), width: Some(Width::Fixed(1)) },
                        ptr: Box::new(ident(name)),
                        offset: Some(Box::new(offset)),
                    };
                }
            }
            Expr::SubSlice { name, .. } => {
                self.fail(format!("'{}[..]' can only be assigned to a name or passed to a slice parameter", name));
            }
            Expr::Call { func, args } => {
                let (Expr::Ident(callee) | Expr::Instantiate { name: callee, .. }) = func.as_ref() else { return };
                let Some(flags) = self.signatures.get(callee) else { return };
                if flags.iter().any(|takes_slice| *takes_slice) {
                    let callee = callee.clone();
                    *args = self.pass_slices(&callee, flags, std::mem::take(args));
                }
            }
            _ => {}
        }
    }

    /// A call's arguments, with each slice as its pointer and then its length.
    fn pass_slices(&self, callee: &str, flags: &[bool], args: Vec<CallArg>) -> Vec<CallArg> {
        let mut lowered = Vec::with_capacity(args.len() + 1);
        for (i, arg) in args.into_iter().enumerate() {
            if !flags.get(i).copied().unwrap_or(false) {
                lowered.push(arg);
                continue;
            }
            if arg.name.is_some() {
                self.fail(format!("a slice can't be passed by name, in a call to '{}'", callee));
            }
            match arg.value {
                Expr::Ident(name) if self.slices.contains_key(&name) => {
                    lowered.push(CallArg { name: None, value: ident(&name) });
                    lowered.push(CallArg { name: None, value: ident(&len_name(&name)) });
                }
                Expr::SubSlice { name, start, end } => {
                    let (start, end) = self.bounds(&name, start, end);
                    let advanced = Expr::Call {
                        func: Box::new(ident("ptr_add")),
                        args: vec![
                            CallArg { name: None, value: ident(&name) },
                            CallArg { name: None, value: start.clone() },
                        ],
                    };
                    lowered.push(CallArg { name: None, value: advanced });
                    lowered.push(CallArg { name: None, value: sub(end, start) });
                }
                other => self.fail(format!(
                    "argument {} of '{}' must be a slice, like s or s[a..b], not {:?}",
                    i + 1,
                    callee,
                    other
                )),
            }
        }
        lowered
    }
}
