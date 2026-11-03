//! `oxc-bench bench|tokens|prof|litmask <file.js> [iters]`, run from the simd-lang root.

#![allow(dead_code)]

#[path = "../../src/ast.rs"]
mod ast;
mod bench;
#[path = "../../src/codegen.rs"]
mod codegen;
#[path = "../../src/comptime.rs"]
mod comptime;
#[path = "../../src/desugar.rs"]
mod desugar;
#[path = "../../src/js.rs"]
mod js;
#[path = "../../src/lexer.rs"]
mod lexer;
#[path = "../../src/parser.rs"]
mod parser;
#[path = "../../src/stage1.rs"]
mod stage1;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(command), Some(file)) = (args.get(1), args.get(2)) else {
        eprintln!("Usage: oxc-bench bench|tokens|prof|litmask <file.js> [iters]");
        std::process::exit(1);
    };
    let iters = |default: usize| args.get(3).and_then(|s| s.parse().ok()).unwrap_or(default);
    match command.as_str() {
        "bench" => bench::run(file, iters(50)),
        "tokens" => bench::dump(file),
        "prof" => bench::prof(file, iters(3000)),
        "litmask" => bench::litmask(file, iters(100)),
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}
