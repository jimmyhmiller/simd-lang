#!/bin/sh
# Compile json_escape.simd to generated/. Extra options go to the compiler, for example
# --dispatch sse4.2,avx2, which also needs `cargo build --features dispatch`.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo build --release --quiet 2>/dev/null
./target/release/simd-lang compile json-escape/json_escape.simd -o json-escape/generated "$@"
