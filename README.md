> This is experimental software. It probably doesn't work.

# simd-lang

A little experiment: what if there was a language where every value is a SIMD vector?

The idea was to see how far you can get with a SIMD-first language that compiles to native code via MLIR. The test case is a JSON structural parser — it hits about 3.5 GB/s on Apple M1 for stage 1 (~73% of simdjson) and ~1.15 GB/s for a full DOM parse (~76% of simdjson).

The whole thing is pretty rough around the edges. It was built iteratively as an exploration, not designed upfront.

## Building & running

The compiler needs LLVM/MLIR 21, and compiling a `.simd` file also runs that install's
`mlir-translate`, `clang` and `llvm-ar`.

- **macOS:** `brew install llvm@21`. `.cargo/config.toml` points the build at it.
- **Linux:** install LLVM 21 with its MLIR libraries from apt.llvm.org, and override that
  path: `export MLIR_SYS_210_PREFIX=/usr/lib/llvm-21 TABLEGEN_210_PREFIX=/usr/lib/llvm-21`.

```bash
# Compile a .simd file to a static library + Rust bindings
cargo run --release -- compile examples/json_stage1.simd -o output/

# Run tests
cargo test --release

# Emit MLIR (useful for debugging)
cargo run --release -- compile examples/json_stage1.simd --emit-mlir

# Cross-compile: any host can target either architecture
cargo run --release -- compile examples/json_stage1.simd --target x86_64-unknown-linux-gnu -o output/
```

## Targets

| Target | Instructions | Register |
|---|---|---|
| `aarch64-*` | NEON | 16 bytes |
| `x86_64-*` | AVX2 and PCLMUL (`-march=x86-64-v3 -mpclmul`) | 32 bytes |
| `x86_64-*` with `--isa sse4.2` | SSE4.2 and PCLMUL (`x86-64-v2`) | 16 bytes |
| `x86_64-*` with `--isa avx512` | AVX-512 and PCLMUL (`x86-64-v4`) | 64 bytes |

A program means the same thing on both, and the test suite passes on both. `--cpu` overrides the CPU that code is generated for, such as `--cpu znver5`. That can use newer instructions, but the language still treats a register as the size in the table.

`--dispatch sse4.2,avx2` builds one library holding each listed level. The first call checks the CPU and the OS, and later calls jump straight to the best level, so a binary runs on any x86-64 CPU that has one of them. It's opt-in because the indirect jump isn't free, and it only handles `extern fn`s taking scalars and pointers. The library also gets `<name>_isa()`, which says which level this CPU got. A CPU with none of the listed levels traps on the first call.

## The language

Every value is a fixed-width vector, and a "scalar" is a width-1 vector. `T[1]` lowers to a plain machine scalar: integers live in general-purpose registers and are passed and returned as ordinary C integers.

```
-- types
u8[64]       -- 64 bytes
i32[4]       -- 4 ints
bool[64]     -- 64 booleans
u64[1]       -- "scalar" u64
u64          -- the same, for short
ptr[u8]      -- pointer to a buffer
```

### Functions

```
fn add(a: f32[8], b: f32[8]) -> f32[8] {
    return a + b
}
```

### Operators

All element-wise: `+` `-` `*` `/` `&` `|` `^` `~` `>>` `<<` `>` `<` `>=` `<=` `==` `!=`

`>>` and `<<` are bit shifts, by a literal, a scalar, or per lane. `>>` is arithmetic on signed types and logical on unsigned ones. For sliding elements within a vector, use `lane_shr(v, n)` / `lane_shl(v, n)`.

### Literals & broadcasting

Literals broadcast to match the other operand:

```
result = chunk + 1       -- 1 broadcasts to chunk's width
mask = chunk == '"'      -- char literal broadcasts
inverted = data ^ ~0     -- ~0 broadcasts to all-ones
```

A literal takes its type wherever one is known: function arguments, `return`, reassigning a variable, values stored through a pointer, struct fields. Constant expressions like `0 - 1` or `1 << 4` count as literals. A literal that doesn't fit its type is an error, though either reading of the bits is fine, so `u8` accepts both `255` and `-1`.

Hex and binary work too: `0xFF`, `0b101`. A constant is evaluated in the type it takes, so `~0 >> 4` is 15 as a `u8`. Ordering comparisons are stricter than bit operations, so `x > -1` on a `u8` is an error.

Otherwise, add a type annotation:

```
all_ones: u64[1] = ~0
zero: u8[64] = 0
```

Vector literals:

```
table = [u8: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
zeros = [u8: 0; 64]     -- repeat syntax
```

### Types

Operands must have the same lane size and kind, or it's a compile error. Convert with `to_u8`, `to_i64` and friends, which extend by the source's signedness. Mixing signed and unsigned is an error where it would change the answer (ordering, `/`, `min`, `max`) and otherwise takes the left operand's type. Both branches of an `if` must agree exactly on a variable's type, and a variable can't change type in a `while`. An annotation reinterprets a value of the same size: `x: u8[1] = a` makes an `i8` read as unsigned.

### Masked expressions

Per-lane select:

```
result = [mask] value_if_true : value_if_false
result = [mask] value_if_true          -- inactive lanes are zero
```

### Reductions & scans

```
total = +/ values                  -- sum reduction
prefix_xor = scan.xor(bits, seed) -- inclusive prefix XOR with carry
```

Reductions: `+/` `*/` `|/` `&/` `max/` `min/`. They follow the lane type's signedness.

`any` of `x <= K` and `x == K` compares on `u8` lanes takes the lane-wise minimum first, so `any(is_escape(a) | is_escape(b))` needs one compare of each kind however wide `a` and `b` are. This sees through `private` functions that are a single `return`.

Scans: `scan.add` `scan.xor` `scan.max` `scan.preceding_any`. `scan.preceding_any` shifts by one lane, so each lane holds the previous lane's value.

#### `scan_dfa` — prefix scan over a finite automaton

```
states = scan_dfa(table, classes, seed, num_states)
```

A prefix scan whose combine operator is **DFA transition-function composition** —
the general case of the `scan.*` family (`scan.xor` is a 2-state parity DFA). It returns the automaton's state after
each lane.

- `table: u8[256]` — `table[state*16 + class] = next_state` (≤16 states, ≤16 input classes)
- `classes: u8[N]` — each lane's input class (e.g. from a `tbl` byte-classifier)
- `seed: u8[1]` — carried start state (use `carry` for cross-chunk continuation)
- `num_states` — compile-time literal

It expresses the *regular* structure of a lexer in SIMD: multi-mode string/
comment region masks (a `/` inside a string is correctly *not* a comment — which
single-toggle `clmul` can't do), number/operator extents, even keyword tries.
`examples/js_litmask.simd` is the full 9-state JS literal DFA (strings, both
comment kinds, escapes), validated byte-for-byte against a scalar reference in
the codegen tests. It lowers, with **no gather**, to `num_states` state-vectors
composed through a log-step Hillis–Steele scan (`tbl` seeds the per-byte
transition functions; each step is `shuffle` + per-lane `cmpi`/`select`).

Two caveats found by building it out:
- **Lowering cost is O(k²) per step** (the per-lane select-compose). At k=9 the
  literal DFA runs ~167 MB/s — fine for a primitive, but *below* the scalar
  tokenizer's rate, so it doesn't currently accelerate it. A `vtbl`-based
  per-position compose (O(1)/position/step) would be much faster but needs a
  per-position-varying table lookup — the one codegen capability still missing.
- **For JS it's regex-poisoned**: stage-1 can't know which `/` is a regex literal
  (needs the previous *token* — parser context), and a quote-bearing regex flips
  the literal-parity for downstream bytes. So the mask is only sound for code
  without quote-bearing regexes — i.e. directly usable for **regex-free languages**
  (JSON, many DSLs), but not the JS hot path. The two non-regular parts of JS
  (regex-vs-division, template `${}` nesting) stay scalar by nature regardless.

### Streams

Process a buffer in chunks with carried state:

```
stream chunk: u8[64] over input carry (count: i32[1] = 0) {
    matches = chunk == 'x'
    carry count = count + popcount(matches)
}
return count
```

`chunk_offset` is automatically available as the byte offset of the current chunk. Carry variables are available after the stream with their final values.

`carry` updates come last in the body and apply together, using the values from before any of them. A carry with no update keeps its value. Assigning to an outer variable in the body is an error, because only carries leave a stream. A chunk's lane type must be the buffer's element type. The buffer's size must be a whole number of chunks, since a final partial chunk is read past the end of the buffer.

To process part of a buffer whose length is only known at runtime, give a range and say what happens to the input that doesn't fill a chunk:

```
stream chunk: u8[4 * REGISTER_BYTES] over src[0..n] tail overlap until is_escape(chunk) {
    store_at(dst, chunk_offset + 1, chunk)
} stopped {
    return slow_path(src, chunk_offset, n, dst)
}
return n
```

| Tail | What happens to the end of the input |
|---|---|
| `tail overlap` | The last chunk ends exactly at the end, and may re-read bytes. Input shorter than a chunk runs on narrower ones, down to one register, then one register testing only the lanes in range. |
| `tail masked` | The last chunk is partial, with `chunk_len` valid lanes. It never reads past the page holding the last byte. |
| `tail leave` | Stops before a partial chunk, so another stream can continue from `chunk_offset`. |

`until` is a per-lane test that can't write memory, and the body only runs on chunks where no lane is true. That lets the compiler test two chunks with one branch before running either chunk's body. `stopped` runs when the stream stops early, with everything from `chunk_offset` on still to be handled. That isn't always because a lane matched: the shortest input also goes there if a read would cross a page, so `stopped` has to cope with clean input. Without a `stopped` that returns, the code after the stream runs, and `chunk_offset` says where it stopped.

A range stream expands to the `while` and `if` you'd write by hand, so it costs nothing. `chunk_offset` here is an `i64[1]`, and stays set after the stream. The body can `return`, and carries work as above except with `tail overlap`, which sees some bytes twice and so rejects them.

### If/else

```
if cond {
    x = a
} else {
    x = b
}
```

`else if` chains work too. Only the taken branch runs, so stores inside it are conditional. Afterwards a variable holds its new value if either branch reassigned it; a name first defined inside the `if` is visible afterwards only if both branches define it.

`for i in a..b { ... }` counts up, and `for i in a..b step k` takes a positive step. It lowers to a `while`, so it costs the same. `a` and `b` are `i64`, `b` and `k` are evaluated once, and afterwards `i` holds the first value out of range.

`for lane in set_bits(bits)` visits the position of each 1 bit in a 64-bit integer, lowest first, as an `i64`. It lowers to the `ctz` and `clear_lowest_bit` loop, and `bits` itself isn't changed.

`x += e` works for `+ - * & | ^ << >>`, and means `x = x + e`, so `d += a - b` groups differently from `d = d + a - b`.

`while cond { ... }` is a real loop, `break` leaves the innermost one, and `return` works anywhere except inside a `stream`. `a && b` and `a || b` work on scalar bools and only evaluate `b` if needed; `&` and `|` work lane-wise and always evaluate both. `likely(cond)` and `unlikely(cond)` say which way a condition usually goes, so the other path is laid out away from the hot path.

Escapes compile to flags, which LLVM's jump threading removes: `json_escape`'s hot path has no flag checks left. A function returning a pointer can only `return` as its last statement.

### Slices

```
extern fn count(s: slice[u8]) -> i64 {
    stream c: u8[16] over s tail masked carry (seen: i64 = 0) {
        carry seen = seen + chunk_len
    }
    return seen
}
```

`slice[T]` is a pointer and a length. It has `s.len` and `s.ptr`, `s[i]` reads one element, and `s[a..b]`, `s[a..]` and `s[..b]` make a smaller slice, to assign to a name or pass to a slice parameter. Anywhere a pointer is expected, a slice works as its pointer. `over s` and `over s[a..]` run up to the slice's length.

A slice costs nothing itself, because it lowers to two plain values. An `extern fn` taking a slice has the C signature `(T *s, int64_t s_len)`, and a generated Rust binding takes it as one `&mut [T]`.

There are no bounds checks: `s[i]` and `s[a..b]` trust their indices. A stream over a slice stays in range by construction, apart from the deliberate reads past the end within a page described above. A slice can't be returned or be a struct field yet.

### Const tables

```
const TABLE = [u8: 10, 20, 30, 40]
```

Read-only static data. Inside functions `TABLE` is a pointer, and `TABLE[i]` reads one element. Indexing isn't bounds-checked.

A table can be computed at compile time, by a `comptime fn` that runs on integers while the program is parsed:

```
comptime fn count_bits(x: i64) -> i64 {
    if x == 0 { return 0 }
    return (x & 1) + count_bits(x >> 1)
}
const POPCOUNT8 = [u8: count_bits(x) for x in 0..256]
```

For a table of rows, give the row width and have the function `emit` each row. `TABLE[i]` then reads a whole row, as a `u8[16]` here. `fill` says what goes in the lanes a row doesn't emit, and without it a short row is an error:

```
comptime fn expansion(mask: i64) {
    for lane in 0..8 {
        if ((mask >> lane) & 1) == 1 { emit(8) }
        emit(lane)
    }
}
const EXPAND = [u8[16]: expansion(mask) for mask in 0..256 fill 255]
```

A `comptime fn` can use scalars, `if`, `for`, `while`, `break`, `return` and calls to other `comptime fn`s. Arithmetic that overflows, a loop that doesn't finish, and a value that doesn't fit the table are all compile errors. Comprehensions also work for local vectors, like `lanes = [u8: i for i in 0..16]`. Their range must be literal numbers, so it can't depend on `REGISTER_BYTES` or `N` yet.

### Width generics

```
fn any_escape[N](a: u8[N], b: u8[N]) -> bool[1] { ... }
fn copy[N](src: ptr[u8], dst: ptr[u8]) { ... }

any_escape(x, y)      -- N inferred from the arguments
copy[32](src, dst)    -- N given explicitly
```

Each set of widths gets its own copy, checked and compiled separately. Inside a body `N` is also a number: `load_at(p, N - 1)`. An `extern fn` can't be generic.

`REGISTER_BYTES` is the width of a vector register on the target, usable as a width or a number, and `[_]` is however many lanes of a type fill one: `u8[_]` is `u8[16]` on NEON and `u8[32]` on AVX2, and `f32[_]` is `f32[4]` and `f32[8]`.

### Function kinds

A public function that returns a `ptr` or a struct gets no Rust binding. Names starting with `__` are reserved, and code after a `return` or `break` is a compile error.

```
private fn helper(...)          -- no exported symbol or binding; LLVM may rewrite its signature
noinline private fn cold(...)   -- stays a call, keeping cold code and its constants off the hot path
extern fn entry(p: ptr[u8], n: i64[1]) -> i64[1]   -- C ABI: `p` is a plain pointer
```

A normal function takes each `ptr[T]` as a five-word memref descriptor. An `extern fn` takes a plain pointer instead, so it doesn't know the size and can't `stream` over it.

### Structs

```
struct Particle[N] {
    pos_x: f32[N]
    vel_x: f32[N]
}
```

Structs can be passed to and returned from functions, including generic ones, and assigned in `if` and `while`. `f(x).pos_x` reads a field of a returned struct, and `(a, b) = f(x)` binds its fields in order.

### Gather

```
chars = gather(input, positions)
chars = input.[positions]          -- equivalent
```

## Builtins

**Vector:** `iota(N)`, `extract(vec, lane)`, `compress(data, mask)`, `popcount(mask)`, `to_u8(vec)`, `to_i32(vec)`, `to_i64(vec)`, `sqrt(vec)`, `split_lo(vec)`, `split_hi(vec)`, `concat(a, b)`, `shuffle(a, b, [T: lanes...])` (constant lanes, indexing `a` then `b`), `min(a, b)`, `max(a, b)`, `fold_min(u8[N])` (down to one register: the minimum of each lane across registers), `any(bool_vec)`, `any_in_first(bool_vec, n)` (over one register)

**Lane shift:** `lane_shr(v, n)`, `lane_shl(v, n)`

**Bitmask:** `to_bitmask(bool_vec)`, `from_bitmask(u64, width)`, `mask8(bool_vec)` (true lanes become 0xFF), `nibblemask(u8[16])` (4 bits per lane, NEON's cheap movemask; lanes must be 0x00 or 0xFF), `clmul(a, b)`, `ctz(v)`, `clear_lowest_bit(v)`

**Table lookup:** `tbl(table_u8x16, indices)`, where an index of 16 or more gives 0

**Memory:** `gather(ptr, indices)`, `store(ptr, data)`, `store_at(ptr, offset, data)`, `compressstore(ptr, offset, data, mask)`, `load[Type](ptr)`, `loadu[Type](ptr)`, `load_at[Type](ptr, offset)`, `ptr_add(ptr, offset)`, `ptr_addr(ptr)`. A byte pointer can load and store any type at any alignment.

## JSON parser

The point of this whole thing. `examples/json_stage1.simd` is a ~50 line SIMD stage 1 that finds JSON structural characters using CLMUL prefix-XOR for string tracking. `src/json.rs` has a Rust tape builder (stage 2) that produces a simdjson-style DOM.

The DOM is verified by round-trip tests: parse → walk tape → reconstruct JSON → compare character-for-character with original.

On Apple M1, 3.35 MB JSON:

| | simdjson | simd-lang |
|---|---------|-----------|
| Stage 1 | 4.78 GB/s | 3.5 GB/s |
| Full DOM parse | 1.51 GB/s | 1.15 GB/s |

## JavaScript lexer

A second test case, same two-stage shape as JSON. `examples/js_stage1.simd` is a
SIMD pass that classifies every byte and emits a stream of **token-start
positions** — context-blind: word-starts (`[A-Za-z0-9_$]` whose predecessor
isn't a word char) plus every punctuator/quote/operator byte. The
"previous byte was a word char" relation is carried *exactly* across 64-byte
chunk boundaries via the bitmask (`(word_bits << 1) | carry`), not the
shift-in-zero approximation.

`src/js.rs` is the scalar Stage 2 that consumes that position stream and owns
the parts that are irreducibly sequential — the ~15% SIMD genuinely can't do:

- **strings / comments / template / regex interiors** — scanned, with the
  spurious stage-1 starts inside them skipped;
- **regex-vs-division** (`/ab+c/g` vs `a / b`) — disambiguated by the previous
  significant token, the classic lexer/parser feedback that makes JS impossible
  to lex context-free;
- **template `${ ... }` nesting** — a brace/template stack (a context-free,
  not regular, property);
- **longest-match multi-char operators** (`===`, `>>>=`, `=>`, …).

Verified the same way as the JSON DOM — a round-trip: the tokens plus the
whitespace gaps between them reconstruct the source byte-for-byte, proving every
byte is covered exactly once. See the tests in `src/js.rs`
(`cargo test --release js::`): cross-chunk boundary correctness, strings vs.
comments that contain each other's delimiters, nested templates, regex-vs-divide,
and number/keyword/identifier classification.

### Comparing against oxc

`cargo run --release --manifest-path oxc-bench/Cargo.toml -- tokens <file.js>` dumps both token
streams (ours → `/tmp/ours.tokens`, [oxc](https://oxc.rs) → `/tmp/oxc.tokens`)
and aligns them; `-- bench <file.js> N` (see `oxc-bench/`) measures throughput.

**Getting oxc to lex standalone.** oxc's lexer is a co-routine of the parser:
the parser resolves regex-vs-division and template `${}` nesting and feeds the
answers back. Run the lexer *alone* and it desyncs at the first regex literal —
on lodash it reads `/&(?:amp|lt|gt|quot|#39);/g` as division and dies at byte
4650 (0.9% of the file). So a naïve "oxc tokenizer" benchmark is meaningless.

The fix is to replicate that feedback. A small fork of oxc widens two methods to
`pub` — `next_regex` and `next_template_substitution_tail` — and `bench.rs` drives
them with the *same* prev-significant-token heuristic our stage-2 uses (plus a
template brace-depth stack). With that, **oxc lexes the entire file with 0 errors**,
and its stream matches ours almost exactly:

| file | our tokens | oxc tokens | exact span matches | differences |
|---|---|---|---|---|
| `lodash.js` (544 KB) | 42,191 (incl. 842 comments) | 41,377 | 41,327 / 41,349 | 22 |
| `three.min.js` (603 KB) | 193,056 | 193,266 | 192,955 | 98 |

Every remaining difference is oxc's *intentional* `>`-splitting (it emits `>`
singly so the parser can reassemble `>>`/`>=`/`>>>` in generics) — a third
feedback hook, not a disagreement about what the tokens are.

**Throughput** (Apple M1, both lexing 100% correctly, non-materializing on both
sides — tokens produced and discarded, matching how a parser consumes them):

| file | stage-1 SIMD | full (s1+s2) | oxc | stage-1 vs oxc | **full vs oxc** |
|---|---|---|---|---|---|
| `three.min.js` (minified, 603 KB) | ~8000 MB/s | 299–325 | 247–255 | ~31× | **1.21×** |
| `vue.global.prod.js` (minified, 148 KB) | ~8000 MB/s | 199–216 | 165–173 | ~48× | **1.20×** |
| `lodash.js` (readable, 544 KB) | ~8000 MB/s | 1063–1160 | 805–874 | ~10× | **1.32×** |

We **beat oxc on every file** while producing the *same* token stream (192,955 /
193,179 spans identical on three.min; the rest are oxc's intentional `>`-splitting),
including keyword classification — i.e. doing the same work oxc does.

How it got there (each step measured, output byte-identical throughout):

1. **Bitmaps, not a position array.** Stage-1 stopped `compressstore`-ing ~1 MB of
   i32 token positions and instead stores two bitmaps (`start_masks`, `is_word`,
   one u64 per 64-byte chunk). Stage-1 throughput **700 → 8000 MB/s** — the
   position array was a 10× memory-traffic bottleneck. Stage-1 is now irrelevant
   to the total; the whole cost is stage-2.
2. **Pull lexer + non-materializing driver.** `src/js.rs` exposes a `Lexer` a
   parser drives token-by-token (`next_token`/`peek` + `re_lex_regex` /
   `re_lex_template_tail` hooks); templates split into head/middle/tail. Counting
   tokens without storing them (as a parser does, and as oxc's loop does) removed
   the `Vec` materialization from both sides.
3. **memchr for comment/string interiors** (borrowed from oxc) — line/block
   comments scan to their end with SIMD `memchr`/`memmem`.
4. **Word-at-a-time bitmap iteration** (simdjson-style) — the current 64-bit word
   lives in a register and bits are cleared as tokens are consumed, instead of
   re-loading + re-masking per token. The single biggest stage-2 win.
5. **Cheap per-token bookkeeping** — a 256-entry lead-byte class table, a
   single-char-punctuator fast class (no look-ahead), a one-`bool` regex-context
   flag (replacing per-token `Option`+`from_utf8`), and a (length, first-letter)
   keyword pre-filter that rejects most identifiers without reading their bytes.

**Parser mode → 2×.** The numbers above do exactly what oxc does, including
keyword classification. But a *real parser* doesn't need the lexer to pre-classify
keywords — it resolves them during identifier interning (this is what V8 and
others do). In **parser mode** we emit `Ident` for keywords and consult the
keyword set only at the rare `/` (to keep regex-vs-division correct), so the
identifier body is never read for keyword classification. The token *boundaries
are byte-identical* to matched-work mode (asserted in `count_matches_tokenize`) —
only the keyword labels are deferred:

Measured across a 7-file real-world corpus (parser-mode MB/s vs oxc MB/s):

| file | type | **parser-mode vs oxc** | matched-work vs oxc |
|---|---|---|---|
| `jquery.js` | readable | **2.15×** | 1.41× |
| `lodash.js` | readable | **2.0×** | 1.41× |
| `react.development.js` | readable | **1.92×** | 1.38× |
| `three.min.js` | minified | 1.71× | 1.25× |
| `react-dom.production.min.js` | minified | 1.74× | 1.19× |
| `d3.min.js` | minified | 1.68× | 1.22× |
| `vue.global.prod.js` | minified | 1.6× | 1.23× |

Every file round-trips and matches oxc's token spans (only oxc's intentional
`>`-splitting differs). So **~2× on readable code, ~1.7× on minified**, with output
verified identical. Why the split: the
pure bitmap-iteration floor is ~100 GB/s (iteration is free); the remaining cost
is reading each token's bytes to classify it. Our structural edge over oxc is
*skipping* whitespace/comment bytes via the bitmap, which is a bigger win on
readable code than on minified (little whitespace to skip). Closing minified to a
full 2× would need **SIMD batch classification** — gathering and classifying many
token lead-bytes per vector op — which the variable token lengths + sequential
regex/template context make a substantial separate effort (profiling shows the
remaining cost is evenly spread across the per-token classify work, no single
hotspot).

Caveat in oxc's favor throughout: it produces full Unicode/escape/TS handling
where ours is ASCII-focused.

## JSON string escaping

[`json-escape/`](json-escape/) reimplements the [`json-escape-simd`](https://crates.io/crates/json-escape-simd) crate. On the crate's own file payloads it runs 2.6-3.4× faster on an Apple M2 Max and 1.5-2.9× faster on Zen 5, with byte-identical output. Its README has the full results and how to reproduce them.

## What's missing

- **Variable lane shifts.** `lane_shr` and `lane_shl` take literal amounts.
- **Seeded `scan.max`.** Only `scan.add` and `scan.xor` take a seed.
- **Leaving a buffer `stream`.** `return` and `break` can't leave one. A range stream can `return`, but not `break`.
- **Other range stream shapes.** Chunks must be `u8`. `until` only works with `tail overlap`, whose chunks must be 1, 2, 4... registers. `tail masked` only takes 16-lane chunks. Streams can't nest.
- **Early return of a pointer.** A function returning a `ptr` can only `return` as its last statement.

All of these are compile errors, not wrong results.

`any_in_first(mask, n)` needs `n` from 1 to the lane count, which is checked only when `n` is a literal.

A generic function is checked once per set of widths it's called with. One that nothing calls is tried at widths 1, 2, 4 ... 64, and is an error only if it fails at all of them. So a mistake that doesn't depend on the width is always caught, and one that only shows up at certain widths waits until a call uses them. A generic that only works at some other width, such as `u8[N + 1]` passed to `tbl`, is rejected until something calls it.

## How it works

```
.simd source → lexer → parser → stream expansion → desugar → type checker → MLIR codegen → mlir-translate → clang -O3 → .a + .rs
```

Errors include source locations (`error at 3:5: undefined variable: foo`). The type checker catches width mismatches, bad builtin args, etc. before codegen.

The codegen emits MLIR's vector dialect operations which LLVM lowers to NEON or AVX2 instructions. The `stream` construct becomes an `scf.for` loop with carried values, `while` an `scf.while`, and `if` an `scf.if`.
