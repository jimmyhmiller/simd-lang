#!/bin/sh
# Prove the high-level kernel costs nothing: json_escape.simd (streams, computed tables) and
# by_hand.simd (hand-written loops, literal tables) must compile to byte-identical object
# files, on every target. Exits non-zero if any differ.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
cargo build --release --quiet 2>/dev/null

case "$(uname -s)" in
    Darwin) ar="$(brew --prefix llvm@21)/bin/llvm-ar" ;;
    *) ar="${LLVM_AR:-/usr/lib/llvm-21/bin/llvm-ar}" ;;
esac

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

mkdir -p "$work/hand" "$work/stream"
cp json-escape/by_hand.simd "$work/hand/json_escape.simd"
cp json-escape/json_escape.simd "$work/stream/json_escape.simd"

status=0
for target in aarch64-apple-darwin x86_64-unknown-linux-gnu; do
    for kind in hand stream; do
        ./target/release/simd-lang compile "$work/$kind/json_escape.simd" \
            -o "$work/$kind/$target" --target "$target" > /dev/null 2> "$work/log" \
            || { cat "$work/log"; exit 2; }
        (cd "$work/$kind/$target" && "$ar" x libjson_escape.a)
    done
    hand="$(cksum < "$work/hand/$target/json_escape.o")"
    stream="$(cksum < "$work/stream/$target/json_escape.o")"
    if [ "$hand" = "$stream" ]; then
        echo "$target: identical ($stream)"
    else
        echo "$target: DIFFERENT (by hand: $hand, stream: $stream)"
        status=1
    fi
done
exit $status
