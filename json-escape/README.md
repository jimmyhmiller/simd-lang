# JSON string escaping in simd-lang

A reimplementation of the [`json-escape-simd`](https://crates.io/crates/json-escape-simd) crate (3.1.3, the sonic-rs NEON kernel) in simd-lang. Its output is byte-identical to the crate and to `serde_json`.

## Throughput

Measured on one Apple M2 Max performance core, in input bytes per second, using `escape_into` with a reused buffer. Each figure comes from the median of three runs. AFFiNE, rxjs and the short strings are the payloads from the crate's own benchmark.

| Workload | Crate | simd-lang |
|---|---|---|
| AFFiNE sources, 24 MB | 2.5 GB/s | **8.5 GB/s** |
| rxjs sources, 814 KB | 3.2 GB/s | **7.8 GB/s** |
| Upstream's 4 short strings | 8.6 GB/s | 8.3 GB/s |
| Clean, 1 MB | 34.7 GB/s | **45.1 GB/s** |
| Clean, 4 KB | 32.5 GB/s | **44.7 GB/s** |
| Clean, 32 B | 7.8 GB/s | **13.9 GB/s** |
| 64 KB, 2% escapes | 6.2 GB/s | **9.7 GB/s** |
| 64 KB, 10% escapes | 1.5 GB/s | **8.9 GB/s** |
| 64 KB, 50% escapes | 0.31 GB/s | **9.0 GB/s** |
| 64 KB, 100% `\u00XX` | 1.9 GB/s | **2.4 GB/s** |

Short strings depend heavily on length because each call has a fixed cost, so per-call time is the better measure there: a clean 32 B string takes 2.3 ns here and 4.1 ns in the crate.

## Where it loses

simd-lang wins 58 of the 61 rows the benchmark measures. It loses these three:

- **One 29 B string with one escape, repeated (0.81×).** The path after the first escape is branchless, about 130 instructions, while the crate's branches cost little once perfectly predicted. On unpredictable strings the same choice wins by up to 1.24×.
- **Upstream's four short strings together (0.97×).** That 29 B string is one of the four; the other three are 1.64-1.81× faster alone. Through the allocating `escape()`, which upstream's published numbers use, `malloc` dominates and the two tie (1.02×, ranging 0.95-1.04).
- **A fixed list of 4,096 clean 1-32 B strings, replayed (0.91×).** The branch predictor memorizes the crate's branches over the repeated sequence. With too many unique strings to memorize, simd-lang wins 1.13×.

## How it works

1. **Optimistic copy.** The hot path assumes nothing needs escaping and uses only immediate constants. The first escape tail-calls an outlined routine, so clean strings never pay for its table addresses.
2. **Overlapping loads.** Lengths 16-32 take two registers and 33-64 take four, and longer strings finish with one overlapping 64 bytes, so varying lengths have no remainder loops to mispredict.
3. **Branchless escapes.** A 256-entry `tbl` shuffle turns 8 bytes into up to 16, so 2-byte escapes cost the same at any density. The crate handles each escape in a scalar loop and then rescans.
4. **A cheaper classifier.** `v ^ 2 <= 32` catches the control bytes and `"` with one compare, and lane-wise minimums across four registers leave one compare of each kind.

## Limitations

- It's NEON-only, while the crate also has x86 kernels.
- A short input can be read up to 15 bytes past its end, or up to 15 bytes before its start near a page boundary, always within mapped pages. The crate over-reads the same way. This is safe on hardware, but AddressSanitizer would flag it.
- The output buffer needs room for `6n + 35` bytes, the same as the crate.

## Files

| File | Purpose |
|---|---|
| `json_escape.simd` | The kernel: hand-written code, then generated tables |
| `gen_tables.py` | Generates the tables at the bottom of the kernel |
| `src/lib.rs` | Rust `escape` and `escape_into` API, matching the crate's |
| `src/main.rs` | The head-to-head benchmark |
| `tests/differential.rs` | Tests against the crate and `serde_json`, including guard pages |
| `build-kernel.sh` | Regenerates the tables and compiles the kernel into `generated/` |
| `fetch-fixtures.sh` | Downloads upstream's rxjs and AFFiNE payloads |

## Reproducing

```bash
./json-escape/fetch-fixtures.sh    # upstream's payloads, about 140 MB
./json-escape/build-kernel.sh      # needs the compiler's LLVM 21 setup
cd json-escape
cargo test --release               # differential and guard-page tests
cargo run --release --bin bench 2>&1 >/dev/null          # every row
cargo run --release --bin bench -- "density" 2>&1 >/dev/null   # groups matching a substring
```
