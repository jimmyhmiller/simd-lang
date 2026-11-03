#!/bin/sh
# Compile json_escape.simd to generated/.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python3 json-escape/gen_tables.py
cargo build --release --quiet 2>/dev/null
./target/release/simd-lang compile json-escape/json_escape.simd -o json-escape/generated
