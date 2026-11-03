#!/bin/sh
# Download the payloads upstream's benchmark uses: rxjs's sources and AFFiNE v0.24.2
# (see napi-rs/escape-simd: download-fixtures.js and benches/escape.rs).
# The files are only ever read as bytes; nothing here runs them.
set -eu
dir="$(cd "$(dirname "$0")" && pwd)/fixtures"
mkdir -p "$dir/rxjs" "$dir/affine"
curl -sSfL https://registry.npmjs.org/rxjs/-/rxjs-7.8.2.tgz | tar xz -C "$dir/rxjs"
curl -sSfL https://github.com/toeverything/AFFiNE/archive/refs/tags/v0.24.2/v0.24.2.tar.gz \
    | tar xz -C "$dir/affine" --strip-components=1
