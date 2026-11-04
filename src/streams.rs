//! Expands range streams into plain `while` and `if`, so they cost exactly what the
//! hand-written loop would. This runs once the register width and the chunk width are known,
//! because they decide which paths exist.
//!
//! `tail overlap` becomes, for a chunk of `W` lanes in registers of `R`:
//!
//! ```text
//! if len < R {                     if len isn't 0 and the read stays on its page: one
//!                                  register, testing only the first `len` lanes
//! } else if len <= 2 * R {         two overlapping registers, tested with one branch
//! } else {
//!     while offset + W < end {     two halves of W, tested with one branch
//!     }
//!     two halves ending at `end`   clamped to the start, so there's no length branch
//! }
//! ```
//!
//! The test always comes before the stores, which is what lets one branch cover two chunks.
//! `tail masked` and `tail leave` are a `while` over whole chunks, then the partial chunk if
//! there is one.

use crate::ast::*;
use crate::comptime::walk_expr;
use crate::desugar::{always_escapes, collect_assigned};
use std::collections::{HashMap, HashSet};

const OFFSET: &str = "chunk_offset";
const LEN: &str = "chunk_len";

pub fn has_range_stream(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match &stmt.kind {
        StmtKind::RangeStream(_) => true,
        StmtKind::If { then_body, else_body, .. } => has_range_stream(then_body) || has_range_stream(else_body),
        StmtKind::While { body, .. } | StmtKind::Stream { body, .. } => has_range_stream(body),
        _ => false,
    })
}

/// Expand a function's range streams and lower its escapes, unless a chunk's width still
/// depends on a compile-time parameter that a later substitution will fill in.
pub fn expand_when_ready(def: &mut FnDef) {
    if !has_range_stream(&def.body) || !ready(&def.body) {
        return;
    }
    let mut defined: HashSet<String> = def.params.iter().map(|p| p.name.clone()).collect();
    let mut expander = Expander { next: 0 };
    def.body = expander.expand_stmts(std::mem::take(&mut def.body), &mut defined);
    crate::desugar::desugar_expanded_fn(def);
}

fn ready(stmts: &[Stmt]) -> bool {
    stmts.iter().all(|stmt| match &stmt.kind {
        StmtKind::RangeStream(stream) => {
            stream.register_bytes.is_some()
                && matches!(stream.chunk_ty.width, Some(Width::Fixed(_)) | Some(Width::Native))
        }
        StmtKind::If { then_body, else_body, .. } => ready(then_body) && ready(else_body),
        StmtKind::While { body, .. } | StmtKind::Stream { body, .. } => ready(body),
        _ => true,
    })
}

/// `until` may be evaluated together for two chunks, before either chunk's stores, so it
/// can't have side effects of its own.
pub fn check_until_is_pure(stmts: &[Stmt], fns: &HashMap<String, FnDef>) {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::RangeStream(stream) => {
                if let Some(until) = &stream.until {
                    let mut seen = HashSet::new();
                    if let Some(what) = expr_writes(until, fns, &mut seen) {
                        panic!("at {}: a stream's `until` can't write memory, but it reaches {}", stmt.span, what);
                    }
                }
            }
            StmtKind::If { then_body, else_body, .. } => {
                check_until_is_pure(then_body, fns);
                check_until_is_pure(else_body, fns);
            }
            StmtKind::While { body, .. } | StmtKind::Stream { body, .. } => check_until_is_pure(body, fns),
            _ => {}
        }
    }
}

fn expr_writes(expr: &Expr, fns: &HashMap<String, FnDef>, seen: &mut HashSet<String>) -> Option<String> {
    let mut callees = Vec::new();
    walk_expr(&mut expr.clone(), &mut |e| {
        if let Expr::Call { func, .. } = e {
            if let Expr::Ident(name) | Expr::Instantiate { name, .. } = func.as_ref() {
                callees.push(name.clone());
            }
        }
    });
    for name in callees {
        if matches!(name.as_str(), "store" | "store_at" | "compressstore") {
            return Some(format!("`{}`", name));
        }
        if let Some(def) = fns.get(&name) {
            if seen.insert(name.clone()) {
                if let Some(what) = stmts_write(&def.body, fns, seen) {
                    return Some(what);
                }
            }
        }
    }
    None
}

fn stmts_write(stmts: &[Stmt], fns: &HashMap<String, FnDef>, seen: &mut HashSet<String>) -> Option<String> {
    stmts.iter().find_map(|stmt| match &stmt.kind {
        StmtKind::Assign { target: AssignTarget::Scatter { .. }, .. } => Some("a scatter".to_string()),
        StmtKind::Assign { value, .. } | StmtKind::Return(value) | StmtKind::Expr(value) => {
            expr_writes(value, fns, seen)
        }
        StmtKind::If { cond, then_body, else_body } => expr_writes(cond, fns, seen)
            .or_else(|| stmts_write(then_body, fns, seen))
            .or_else(|| stmts_write(else_body, fns, seen)),
        StmtKind::While { cond, body } => expr_writes(cond, fns, seen).or_else(|| stmts_write(body, fns, seen)),
        StmtKind::Stream { body, .. } => stmts_write(body, fns, seen),
        StmtKind::RangeStream(stream) => stmts_write(&stream.body, fns, seen),
        StmtKind::ReturnVoid | StmtKind::Break => None,
    })
}

// --- Building the AST ---

fn id(name: &str) -> Expr {
    Expr::Ident(name.to_string())
}

fn int(n: i64) -> Expr {
    Expr::IntLit(n)
}

fn bin(op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
    Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        func: Box::new(id(name)),
        args: args.into_iter().map(|value| CallArg { name: None, value }).collect(),
    }
}

fn unlikely(cond: Expr) -> Expr {
    call("unlikely", vec![cond])
}

fn i64_type() -> Type {
    Type { name: "i64".to_string(), width: Some(Width::Fixed(1)) }
}

fn u8_type(lanes: u64) -> Type {
    Type { name: "u8".to_string(), width: Some(Width::Fixed(lanes)) }
}

fn rename_in_expr(expr: &mut Expr, names: &HashMap<String, String>) {
    walk_expr(expr, &mut |e| {
        if let Expr::Ident(name) = e {
            if let Some(new) = names.get(name) {
                *name = new.clone();
            }
        }
    });
}

fn rename_in_stmts(stmts: &mut [Stmt], names: &HashMap<String, String>) {
    for stmt in stmts {
        match &mut stmt.kind {
            StmtKind::Assign { target, value, .. } => {
                match target {
                    AssignTarget::Ident(name) => {
                        if let Some(new) = names.get(name) {
                            *name = new.clone();
                        }
                    }
                    AssignTarget::Destructure(targets) => {
                        for name in targets {
                            if let Some(new) = names.get(name) {
                                *name = new.clone();
                            }
                        }
                    }
                    AssignTarget::Scatter { base, index, mask } => {
                        rename_in_expr(base, names);
                        rename_in_expr(index, names);
                        if let Some(mask) = mask {
                            rename_in_expr(mask, names);
                        }
                    }
                }
                rename_in_expr(value, names);
            }
            StmtKind::Return(value) | StmtKind::Expr(value) => rename_in_expr(value, names),
            StmtKind::ReturnVoid | StmtKind::Break => {}
            StmtKind::If { cond, then_body, else_body } => {
                rename_in_expr(cond, names);
                rename_in_stmts(then_body, names);
                rename_in_stmts(else_body, names);
            }
            StmtKind::While { cond, body } => {
                rename_in_expr(cond, names);
                rename_in_stmts(body, names);
            }
            StmtKind::Stream { .. } | StmtKind::RangeStream(_) => {
                unreachable!("streams inside a range stream are rejected before renaming")
            }
        }
    }
}

// --- Expansion ---

struct Expander {
    next: usize,
}

/// One range stream, while it's being expanded. Each copy of the body gets its own names,
/// since the same local can have a different width on another path.
struct Expansion<'s> {
    stream: &'s RangeStream,
    span: Span,
    /// Names the stream's `stopped` assigns that exist outside it, which keep their names.
    outer: HashSet<String>,
    end: String,
    start: String,
    /// Set by `stopped` when it falls through, to skip the rest of the stream.
    stopped_flag: Option<String>,
    /// A return to put at the end of each path through the stream.
    after: Option<Stmt>,
    register: u64,
    lanes: u64,
}

impl Expander {
    fn fresh(&mut self, name: &str) -> String {
        self.next += 1;
        format!("__{}_{}", name, self.next)
    }

    fn expand_stmts(&mut self, stmts: Vec<Stmt>, defined: &mut HashSet<String>) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(stmts.len());
        let mut stmts = stmts.into_iter().peekable();
        while let Some(stmt) = stmts.next() {
            let span = stmt.span;
            match stmt.kind {
                StmtKind::RangeStream(stream) => {
                    // A return straight after an overlap stream goes on each of its paths, so
                    // each can reuse what it already computed, as a hand-written one would.
                    let copies_return = stream.tail == Tail::Overlap
                        && stream.stopped.as_deref().is_some_and(always_escapes)
                        && stmts.len() == 1
                        && matches!(stmts.peek().map(|s| &s.kind), Some(StmtKind::Return(_) | StmtKind::ReturnVoid));
                    let after = if copies_return { stmts.next() } else { None };
                    out.extend(self.expand(&stream, span, defined, after));
                    defined.insert(OFFSET.to_string());
                    defined.extend(stream.carry.iter().map(|c| c.name.clone()));
                }
                StmtKind::If { cond, then_body, else_body } => {
                    let kind = StmtKind::If {
                        cond,
                        then_body: self.expand_stmts(then_body, defined),
                        else_body: self.expand_stmts(else_body, defined),
                    };
                    out.push(Stmt { kind, span });
                }
                StmtKind::While { cond, body } => {
                    let kind = StmtKind::While { cond, body: self.expand_stmts(body, defined) };
                    out.push(Stmt { kind, span });
                }
                StmtKind::Stream { body, .. } if has_range_stream(&body) => {
                    panic!("at {}: a stream can't contain a stream over a range", span);
                }
                kind => {
                    let stmt = Stmt { kind, span };
                    let mut assigned = Vec::new();
                    collect_assigned(std::slice::from_ref(&stmt), &mut assigned);
                    defined.extend(assigned);
                    out.push(stmt);
                }
            }
        }
        out
    }

    fn expand(&mut self, stream: &RangeStream, span: Span, defined: &HashSet<String>, after: Option<Stmt>) -> Vec<Stmt> {
        let register = stream.register_bytes.expect("expansion waits for the register width");
        let lanes = match &stream.chunk_ty.width {
            Some(Width::Fixed(n)) => *n,
            Some(Width::Native) => register,
            other => panic!("at {}: a stream's chunk needs a width, got {:?}", span, other),
        };
        check_stream(stream, span, defined, lanes);

        let stmt = |kind| Stmt { kind, span };
        let assign = |name: &str, ty: Option<Type>, value: Expr| {
            stmt(StmtKind::Assign { target: AssignTarget::Ident(name.to_string()), ty, value })
        };

        let start = self.fresh("start");
        let end = self.fresh("end");
        let mut out = Vec::new();
        out.push(assign(&start, Some(i64_type()), stream.start.clone()));
        out.push(assign(&end, Some(i64_type()), stream.end.clone()));
        out.push(assign(OFFSET, Some(i64_type()), id(&start)));
        for carry in &stream.carry {
            out.push(assign(&carry.name, Some(carry.ty.clone()), carry.init.clone()));
        }

        let falls_through = stream.until.is_some() && !stream.stopped.as_deref().is_some_and(always_escapes);
        let stopped_flag = falls_through.then(|| self.fresh("stopped"));
        if let Some(flag) = &stopped_flag {
            out.push(stmt(StmtKind::Assign {
                target: AssignTarget::Ident(flag.clone()),
                ty: Some(Type { name: "bool".to_string(), width: Some(Width::Fixed(1)) }),
                value: Expr::BoolLit(false),
            }));
        }
        let mut outer: HashSet<String> = defined.clone();
        outer.insert(OFFSET.to_string());
        outer.extend(stream.carry.iter().map(|c| c.name.clone()));
        let expansion = Expansion { stream, span, outer, end, start, stopped_flag, after, register, lanes };
        out.extend(match stream.tail {
            Tail::Overlap => self.overlap(&expansion),
            Tail::Masked => self.masked(&expansion),
            Tail::Leave => self.whole_chunks(&expansion),
        });
        out
    }

    /// A copy of statements for one load of the chunk: the chunk and the locals renamed.
    fn instance(&mut self, x: &Expansion, stmts: &[Stmt], chunk: &str) -> Vec<Stmt> {
        let names = self.names_for(x, stmts, chunk);
        let mut out = stmts.to_vec();
        rename_in_stmts(&mut out, &names);
        out
    }

    fn until_on(&self, x: &Expansion, chunk: &str) -> Expr {
        let mut until = x.stream.until.clone().expect("overlap requires until");
        let names = HashMap::from([(x.stream.chunk_name.clone(), chunk.to_string())]);
        rename_in_expr(&mut until, &names);
        until
    }

    /// `stopped`, then the way out of the stream if it doesn't return.
    fn stop(&mut self, x: &Expansion, in_loop: bool) -> Vec<Stmt> {
        let stmt = |kind| Stmt { kind, span: x.span };
        // The chunk isn't in scope here: on some paths nothing was loaded.
        let stopped = x.stream.stopped.as_deref().unwrap_or(&[]);
        let mut out = self.instance(x, stopped, &x.stream.chunk_name);
        if let Some(flag) = &x.stopped_flag {
            out.push(stmt(StmtKind::Assign {
                target: AssignTarget::Ident(flag.clone()),
                ty: None,
                value: Expr::BoolLit(true),
            }));
            if in_loop {
                out.push(stmt(StmtKind::Break));
            }
        }
        out
    }

    /// What ends a path that handled the whole range: the offset, and any copied return.
    fn finish(&self, x: &Expansion) -> Vec<Stmt> {
        let mut out = vec![self.set_offset(x, id(&x.end))];
        out.extend(x.after.clone());
        out
    }

    fn load(&self, x: &Expansion, name: &str, lanes: u64, offset: Expr) -> Stmt {
        Stmt {
            kind: StmtKind::Assign {
                target: AssignTarget::Ident(name.to_string()),
                ty: None,
                value: Expr::Load {
                    aligned: false,
                    ty: u8_type(lanes),
                    ptr: Box::new(id(&x.stream.buffer)),
                    offset: Some(Box::new(offset)),
                },
            },
            span: x.span,
        }
    }

    fn set_offset(&self, x: &Expansion, value: Expr) -> Stmt {
        Stmt {
            kind: StmtKind::Assign { target: AssignTarget::Ident(OFFSET.to_string()), ty: None, value },
            span: x.span,
        }
    }

    fn if_else(&self, x: &Expansion, cond: Expr, then_body: Vec<Stmt>, else_body: Vec<Stmt>) -> Stmt {
        Stmt { kind: StmtKind::If { cond, then_body, else_body }, span: x.span }
    }

    fn overlap(&mut self, x: &Expansion) -> Vec<Stmt> {
        let (r, w) = (x.register, x.lanes);
        if w < r || w % r != 0 || !(w / r).is_power_of_two() {
            panic!(
                "at {}: `tail overlap` needs a chunk of 1, 2, 4... registers ({} bytes each), got {} lanes",
                x.span, r, w
            );
        }
        let len = bin(BinOp::Sub, id(&x.end), id(&x.start));

        // Whole chunks at a time, with the last one ending exactly at the end.
        let mut ladder = {
            let advance = bin(BinOp::Add, id(OFFSET), int(w as i64));
            let cond = bin(BinOp::Lt, advance, id(&x.end));
            let (loop_body, last) = if w == r {
                let body = self.run_single(x, w, id(OFFSET), true);
                let last = self.run_single(x, w, bin(BinOp::Sub, id(&x.end), int(w as i64)), false);
                (body, last)
            } else {
                let half = w / 2;
                let second = bin(BinOp::Add, id(OFFSET), int(half as i64));
                let body = self.run_pair(x, half, id(OFFSET), second, true);
                let clamped = call("max", vec![bin(BinOp::Sub, id(&x.end), int(w as i64)), id(&x.start)]);
                let tail = bin(BinOp::Sub, id(&x.end), int(half as i64));
                (body, self.run_pair(x, half, clamped, tail, false))
            };
            let mut out = vec![Stmt { kind: StmtKind::While { cond, body: loop_body }, span: x.span }];
            match &x.stopped_flag {
                Some(flag) => {
                    let not_stopped = Expr::UnaryOp { op: UnaryOp::Not, operand: Box::new(id(flag)) };
                    out.push(self.if_else(x, not_stopped, last, vec![]));
                }
                None => out.extend(last),
            }
            out
        };

        // Narrower pairs for what's too short for a whole chunk, widest first.
        let mut lanes = w / 4;
        while lanes >= r {
            let second = bin(BinOp::Sub, id(&x.end), int(lanes as i64));
            let pair = self.run_pair(x, lanes, id(OFFSET), second, false);
            let fits = bin(BinOp::LtEq, len.clone(), int(2 * lanes as i64));
            ladder = vec![self.if_else(x, fits, pair, ladder)];
            lanes /= 2;
        }

        let shortest = self.shorter_than_a_register(x, len.clone());
        let is_short = bin(BinOp::Lt, len, int(r as i64));
        vec![self.if_else(x, is_short, shortest, ladder)]
    }

    /// Under a register: read a whole one when that stays on the page, and test only the
    /// lanes in range. Otherwise it's `stopped`'s to handle, like any remaining input.
    fn shorter_than_a_register(&mut self, x: &Expansion, len: Expr) -> Vec<Stmt> {
        let r = x.register;
        let chunk = self.fresh("chunk");
        let address = bin(BinOp::Add, call("ptr_addr", vec![id(&x.stream.buffer)]), id(OFFSET));
        let in_page = bin(BinOp::And, address, int(4095));
        let crosses = unlikely(bin(BinOp::Gt, in_page, int(4096 - r as i64)));
        let dirty = unlikely(call("any_in_first", vec![self.until_on(x, &chunk), len.clone()]));

        let mut commit = self.instance(x, &x.stream.body, &chunk);
        commit.extend(self.finish(x));
        let (stop_dirty, stop_crossing) = (self.stop(x, false), self.stop(x, false));
        let tested = vec![self.load(x, &chunk, r, id(OFFSET)), self.if_else(x, dirty, stop_dirty, commit)];
        let guarded = vec![self.if_else(x, crosses, stop_crossing, tested)];
        vec![self.if_else(x, unlikely(bin(BinOp::EqEq, len, int(0))), self.finish(x), guarded)]
    }

    /// Two chunks, one test: they're either both committed, or neither is.
    fn run_pair(&mut self, x: &Expansion, lanes: u64, first: Expr, second: Expr, in_loop: bool) -> Vec<Stmt> {
        let (a, b) = (self.fresh("chunk"), self.fresh("chunk"));
        let (a_at, b_at) = (self.fresh("at"), self.fresh("at"));
        let stmt = |kind| Stmt { kind, span: x.span };
        let bind = |name: &str, value: Expr| {
            stmt(StmtKind::Assign { target: AssignTarget::Ident(name.to_string()), ty: Some(i64_type()), value })
        };
        let dirty = call("any", vec![bin(BinOp::Or, self.until_on(x, &a), self.until_on(x, &b))]);

        let mut commit = vec![self.set_offset(x, id(&a_at))];
        commit.extend(self.instance(x, &x.stream.body, &a));
        commit.push(self.set_offset(x, id(&b_at)));
        commit.extend(self.instance(x, &x.stream.body, &b));
        if in_loop {
            commit.push(self.set_offset(x, bin(BinOp::Add, id(&b_at), int(lanes as i64))));
        } else {
            commit.extend(self.finish(x));
        }

        let stop = self.stop(x, in_loop);
        vec![
            bind(&a_at, first),
            bind(&b_at, second),
            self.load(x, &a, lanes, id(&a_at)),
            self.load(x, &b, lanes, id(&b_at)),
            self.if_else(x, unlikely(dirty), stop, commit),
        ]
    }

    fn run_single(&mut self, x: &Expansion, lanes: u64, at: Expr, in_loop: bool) -> Vec<Stmt> {
        let chunk = self.fresh("chunk");
        let at_name = self.fresh("at");
        let dirty = call("any", vec![self.until_on(x, &chunk)]);
        let mut commit = vec![self.set_offset(x, id(&at_name))];
        commit.extend(self.instance(x, &x.stream.body, &chunk));
        if in_loop {
            commit.push(self.set_offset(x, bin(BinOp::Add, id(&at_name), int(lanes as i64))));
        } else {
            commit.extend(self.finish(x));
        }
        let stop = self.stop(x, in_loop);
        vec![
            Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident(at_name.clone()),
                    ty: Some(i64_type()),
                    value: at,
                },
                span: x.span,
            },
            self.load(x, &chunk, lanes, id(&at_name)),
            self.if_else(x, unlikely(dirty), stop, commit),
        ]
    }

    /// The names one copy of the given statements uses: the chunk, and its own locals.
    fn names_for(&mut self, x: &Expansion, stmts: &[Stmt], chunk: &str) -> HashMap<String, String> {
        let mut assigned = Vec::new();
        collect_assigned(stmts, &mut assigned);
        let mut names = HashMap::from([(x.stream.chunk_name.clone(), chunk.to_string())]);
        for name in assigned {
            if !x.outer.contains(&name) && !names.contains_key(&name) {
                let renamed = self.fresh(&name);
                names.insert(name, renamed);
            }
        }
        names
    }

    /// One copy of the body and its carry updates.
    fn commit_with_carries(&mut self, x: &Expansion, chunk: &str) -> Vec<Stmt> {
        let names = self.names_for(x, &x.stream.body, chunk);
        let mut out = x.stream.body.clone();
        rename_in_stmts(&mut out, &names);

        // Updates apply together, each computed from the values before any of them.
        let mut updates: Vec<(String, Expr)> = x.stream.carry_updates.clone();
        for (_, value) in &mut updates {
            rename_in_expr(value, &names);
        }
        let assign = |name: &str, value: Expr| Stmt {
            kind: StmtKind::Assign { target: AssignTarget::Ident(name.to_string()), ty: None, value },
            span: x.span,
        };
        if updates.len() == 1 {
            let (name, value) = updates.remove(0);
            out.push(assign(&name, value));
            return out;
        }
        let temps: Vec<String> = updates.iter().map(|(name, _)| self.fresh(name)).collect();
        for ((_, value), temp) in updates.iter().zip(&temps) {
            out.push(assign(temp, value.clone()));
        }
        for ((name, _), temp) in updates.iter().zip(&temps) {
            out.push(assign(name, id(temp)));
        }
        out
    }

    fn whole_chunks(&mut self, x: &Expansion) -> Vec<Stmt> {
        let w = x.lanes;
        let chunk = self.fresh("chunk");
        // Loading in halves is what a loop over two half-width loads would do, so moving
        // one of those to a stream doesn't change its code.
        let mut body = if w % 2 == 0 && w > x.register {
            let (lo, hi) = (self.fresh("lo"), self.fresh("hi"));
            vec![
                self.load(x, &lo, w / 2, id(OFFSET)),
                self.load(x, &hi, w / 2, bin(BinOp::Add, id(OFFSET), int(w as i64 / 2))),
                Stmt {
                    kind: StmtKind::Assign {
                        target: AssignTarget::Ident(chunk.clone()),
                        ty: None,
                        value: call("concat", vec![id(&lo), id(&hi)]),
                    },
                    span: x.span,
                },
            ]
        } else {
            vec![self.load(x, &chunk, w, id(OFFSET))]
        };
        if x.stream.tail == Tail::Masked {
            body.push(Stmt {
                kind: StmtKind::Assign {
                    target: AssignTarget::Ident(LEN.to_string()),
                    ty: Some(i64_type()),
                    value: int(w as i64),
                },
                span: x.span,
            });
        }
        body.extend(self.commit_with_carries(x, &chunk));
        body.push(self.set_offset(x, bin(BinOp::Add, id(OFFSET), int(w as i64))));
        let cond = bin(BinOp::LtEq, bin(BinOp::Add, id(OFFSET), int(w as i64)), id(&x.end));
        vec![Stmt { kind: StmtKind::While { cond, body }, span: x.span }]
    }

    /// Whole chunks, then a partial one with `chunk_len` valid lanes. Past a page end it
    /// loads the chunk that ends at the last byte instead, which starts in the page holding
    /// `chunk_offset`, and slides it down.
    fn masked(&mut self, x: &Expansion) -> Vec<Stmt> {
        let w = x.lanes;
        if w != 16 {
            panic!(
                "at {}: `tail masked` only supports 16-lane chunks so far, since sliding a partial chunk uses `tbl`; got {}",
                x.span, w
            );
        }
        let mut out = self.whole_chunks(x);
        let remaining = self.fresh("remaining");
        let chunk = self.fresh("chunk");
        let assign = |name: &str, ty: Option<Type>, value: Expr| Stmt {
            kind: StmtKind::Assign { target: AssignTarget::Ident(name.to_string()), ty, value },
            span: x.span,
        };
        out.push(assign(&remaining, Some(i64_type()), bin(BinOp::Sub, id(&x.end), id(OFFSET))));

        let address = bin(BinOp::Add, call("ptr_addr", vec![id(&x.stream.buffer)]), id(OFFSET));
        let crosses = unlikely(bin(BinOp::Gt, bin(BinOp::And, address, int(4095)), int(4080)));
        let last = bin(BinOp::Sub, bin(BinOp::Add, id(OFFSET), id(&remaining)), int(16));
        let lanes = Expr::VecLit { elem_type: "u8".to_string(), values: (0..16).collect() };
        let slide = bin(
            BinOp::Add,
            lanes,
            call("to_u8", vec![bin(BinOp::Sub, int(16), id(&remaining))]),
        );
        let loaded = Expr::Load {
            aligned: false,
            ty: u8_type(16),
            ptr: Box::new(id(&x.stream.buffer)),
            offset: Some(Box::new(last)),
        };
        let slid = vec![assign(&chunk, None, call("tbl", vec![loaded, slide]))];
        let plain = vec![self.load(x, &chunk, 16, id(OFFSET))];

        let mut partial = vec![
            self.if_else(x, crosses, slid, plain),
            assign(LEN, Some(i64_type()), id(&remaining)),
        ];
        partial.extend(self.commit_with_carries(x, &chunk));
        partial.push(self.set_offset(x, id(&x.end)));
        out.push(self.if_else(x, bin(BinOp::Gt, id(&remaining), int(0)), partial, vec![]));
        out
    }
}

/// Rules that hold for every width, with messages in terms of the source.
fn check_stream(stream: &RangeStream, span: Span, defined: &HashSet<String>, lanes: u64) {
    if stream.chunk_ty.name != "u8" {
        panic!("at {}: a stream over a range only supports u8 chunks so far, got {}", span, stream.chunk_ty.name);
    }
    if lanes == 0 {
        panic!("at {}: a stream's chunk can't have zero lanes", span);
    }
    match (stream.tail, &stream.until) {
        (Tail::Overlap, None) => panic!("at {}: `tail overlap` needs `until`, to say what stops the stream", span),
        (Tail::Overlap, Some(_)) => {
            if !stream.carry.is_empty() {
                panic!("at {}: `tail overlap` can read bytes twice, so it can't have carries", span);
            }
            if contains_break(&stream.body) {
                panic!("at {}: `tail overlap` can't `break`; use `until` to stop it", span);
            }
        }
        (_, Some(_)) => panic!("at {}: `until` only works with `tail overlap` so far", span),
        (_, None) => {
            if stream.stopped.is_some() {
                panic!("at {}: `stopped` needs an `until`", span);
            }
        }
    }
    if let Some(until) = &stream.until {
        let mut uses_offset = false;
        walk_expr(&mut until.clone(), &mut |e| {
            uses_offset |= matches!(e, Expr::Ident(name) if name == OFFSET);
        });
        if uses_offset {
            panic!("at {}: `until` can't use `chunk_offset`, since it may test two chunks at once", span);
        }
    }
    if contains_stream(&stream.body) || stream.stopped.as_deref().is_some_and(contains_stream) {
        panic!("at {}: a stream over a range can't contain another stream", span);
    }
    if contains_break(&stream.body) {
        panic!("at {}: `break` out of a stream over a range isn't supported yet; use `return`", span);
    }
    let mut assigned = Vec::new();
    collect_assigned(&stream.body, &mut assigned);
    for name in assigned {
        if name == OFFSET || name == LEN {
            panic!("at {}: `{}` is set by the stream and can't be assigned", span, name);
        }
        if defined.contains(&name) && !stream.carry.iter().any(|c| c.name == name) {
            panic!("at {}: '{}' is assigned in a stream, which won't change it outside; make it a carry", span, name);
        }
    }
    let mut updated = HashSet::new();
    for (name, _) in &stream.carry_updates {
        if !stream.carry.iter().any(|c| &c.name == name) {
            panic!("at {}: '{}' isn't one of this stream's carries", span, name);
        }
        if !updated.insert(name.clone()) {
            panic!("at {}: carry '{}' is updated twice", span, name);
        }
    }
}

fn contains_stream(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match &stmt.kind {
        StmtKind::Stream { .. } | StmtKind::RangeStream(_) => true,
        StmtKind::If { then_body, else_body, .. } => contains_stream(then_body) || contains_stream(else_body),
        StmtKind::While { body, .. } => contains_stream(body),
        _ => false,
    })
}

/// A `break` that would leave the stream, rather than a `while` inside it.
fn contains_break(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match &stmt.kind {
        StmtKind::Break => true,
        StmtKind::If { then_body, else_body, .. } => contains_break(then_body) || contains_break(else_body),
        _ => false,
    })
}
