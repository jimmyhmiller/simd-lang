# JSON string escaping in simd-lang

A reimplementation of the [`json-escape-simd`](https://crates.io/crates/json-escape-simd) crate (3.1.3, the sonic-rs kernels) in simd-lang. One source file, `json_escape.simd`, compiles to NEON and to AVX2. Its output is byte-identical to the crate and to `serde_json`.

## Throughput

In input bytes per second on one core, using `escape_into` with a reused buffer, as the median of five runs. AFFiNE, rxjs and the short strings are the payloads from the crate's own benchmark. The machines are an Apple M2 Max and an AMD Ryzen AI Max+ 395 (Zen 5), and the crate uses its default NEON and AVX2 kernels.

| Workload | M2 crate | M2 simd-lang | M2 | Zen 5 crate | Zen 5 simd-lang | Zen 5 |
|---|---|---|---|---|---|---|
| AFFiNE sources, 24 MB | 2.6 GB/s | **8.6 GB/s** | 3.35× | 4.3 GB/s | **12.4 GB/s** | 2.86× |
| rxjs sources, 814 KB | 3.2 GB/s | **8.3 GB/s** | 2.58× | 8.8 GB/s | **12.8 GB/s** | 1.46× |
| Upstream's 4 short strings | 8.4 GB/s | 8.5 GB/s | 1.01× | 8.5 GB/s | **9.7 GB/s** | 1.14× |
| Clean, 1 MB | 34.2 GB/s | **44.5 GB/s** | 1.31× | 66.7 GB/s | 66.2 GB/s | 0.99× |
| Clean, 4 KB | 32.2 GB/s | **44.2 GB/s** | 1.38× | 76.6 GB/s | **117.3 GB/s** | 1.53× |
| Clean, 32 B | 7.3 GB/s | **13.9 GB/s** | 1.90× | 9.3 GB/s | **14.3 GB/s** | 1.54× |
| 64 KB, 2% escapes | 6.3 GB/s | **10.0 GB/s** | 1.61× | 11.1 GB/s | **16.6 GB/s** | 1.49× |
| 64 KB, 10% escapes | 1.6 GB/s | **9.8 GB/s** | 6.17× | 2.9 GB/s | **14.1 GB/s** | 4.88× |
| 64 KB, 50% escapes | 0.32 GB/s | **9.9 GB/s** | 31.03× | 0.91 GB/s | **14.2 GB/s** | 15.64× |
| 64 KB, 100% `\u00XX` | 1.9 GB/s | **2.4 GB/s** | 1.28× | 2.5 GB/s | **3.3 GB/s** | 1.32× |

The crate also has optional AVX-512 kernels. Against those on the Zen 5, simd-lang's AVX2 build is still 2.36× faster on AFFiNE and 1.51× faster on rxjs. Tuning it for the CPU with `--cpu znver5` makes clean bulk faster, to 140 GB/s at 4 KB, but escapes slower, at 11.9 GB/s against 13.9, so the default stays at `x86-64-v3`.

Short strings depend heavily on length because each call has a fixed cost, so per-call time is the better measure there: a clean 32 B string takes 2.3 ns on the M2 against the crate's 4.4 ns, and 2.2 ns on Zen 5 against 3.4 ns.

## Where it doesn't win

Of the 61 rows the benchmark measures, simd-lang wins 58 on the M2 and 55 on Zen 5.

**It loses two on each machine:**

- **One 29 B string with one escape, repeated** (0.82× on the M2, 0.84× on Zen 5). The path after the first escape is branchless, while the crate's branches cost little once perfectly predicted. On strings too varied to predict, the same choice wins by up to 1.27× and 1.28×.
- **A fixed list of strings, replayed.** The branch predictor memorizes the crate's branches over the repeated sequence: 0.95× for 4,096 clean 1-32 B strings on the M2, and 0.90× for 4,096 strings of 1-200 B with 2% escapes on Zen 5. With too many unique strings to memorize, the first becomes 1.15×.

**It ties these,** within 2%:

- Upstream's four short strings on the M2 (1.01×), because that 29 B string is one of them. The other three are 1.63-1.81× faster alone. On Zen 5 the batch is 1.14×.
- Clean 1 KB and 1 MB on Zen 5 (1.00× and 0.99×). The cause isn't established: simd-lang wins at 256 B (1.22×), 4 KB (1.53×) and 64 KB (1.23×) on either side.
- 0.1% density on Zen 5 (0.99× for 2-byte escapes and 1.01× for `\u00XX`). The gap opens as escapes become more common, reaching 1.49× at 2%.

Zen 5 timings vary between processes on some short-string rows, and in rare runs the crate or simd-lang is much slower across a whole row group, which looks like an address-layout effect. The table uses medians across restarts for this reason.

## How it works

1. **Optimistic copy.** The hot path assumes nothing needs escaping and uses only immediate constants. The first escape returns through an outlined routine, so clean strings never pay for its table addresses.
2. **Register-sized short path.** A string shorter than a register takes one path, whatever its length, so there's no length to mispredict.
3. **Two overlapping registers.** `copy2_if_clean[N]` loads, tests and stores two N-byte registers that may overlap. That one function covers every length up to twice a register, the clean loop, and the final overlapping chunk, with no remainder loops.
4. **Branchless escapes.** A 256-entry shuffle table turns 8 bytes into up to 16, so 2-byte escapes cost the same at any density. The crate handles each escape in a scalar loop and then rescans.
5. **A cheaper classifier.** `v ^ 2 <= 32` catches the control bytes and `"` with one compare, and `fold_min` reduces any width to one register first, leaving one compare of each kind.

## Limitations

- The x86 build requires AVX2 and has no runtime dispatch, while the crate detects the CPU and also runs on older ones.
- A short input can be read past its end, up to one register minus a byte, always within its last page. The crate over-reads the same way, with the same page check.
- The output buffer needs room for `6n + 35` bytes, the same as the crate.

## Testing

`tests/differential.rs` compares every result with the crate and with `serde_json`: every ASCII byte at every position, every length to 600, 200,000 random strings, and inputs and outputs flush against guard pages. To check that those tests can fail, 22 deliberate bugs were injected into the kernel one at a time, and the suite caught all 22 on both architectures.

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
./json-escape/fetch-fixtures.sh    # upstream's payloads, about 150 MB
./json-escape/build-kernel.sh      # see the root README for the LLVM 21 setup
cd json-escape
cargo test --release               # differential and guard-page tests
cargo run --release --bin bench 2>&1 >/dev/null                # every row
cargo run --release --bin bench -- "density" 2>&1 >/dev/null   # groups matching a substring
cargo run --release --bin bench --features crate-avx512 2>&1 >/dev/null   # against AVX-512
```

On Linux, pin the benchmark to a core with `taskset`.
