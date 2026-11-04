# JSON string escaping in simd-lang

A reimplementation of the [`json-escape-simd`](https://crates.io/crates/json-escape-simd) crate (3.1.3, the sonic-rs kernels) in simd-lang. One source file, `json_escape.simd`, compiles to NEON and to AVX2, and it's written with `stream`s and tables computed at compile time, rather than hand-written loops and a generator script. That compiles to the same object file as the [hand-written kernel](by_hand.simd), byte for byte. Its output is byte-identical to the crate and to `serde_json`.

## Throughput

In input bytes per second on one core, using `escape_into` with a reused buffer, as the median of twelve runs. AFFiNE, rxjs and the short strings are the payloads from the crate's own benchmark. The machines are an Apple M2 Max and an AMD Ryzen AI Max+ 395 (Zen 5), and the crate uses its default NEON and AVX2 kernels.

| Workload | M2 crate | M2 simd-lang | M2 | Zen 5 crate | Zen 5 simd-lang | Zen 5 |
|---|---|---|---|---|---|---|
| AFFiNE sources, 24 MB | 2.4 GB/s | **8.1 GB/s** | 3.42× | 4.3 GB/s | **12.4 GB/s** | 2.85× |
| rxjs sources, 814 KB | 3.0 GB/s | **7.9 GB/s** | 2.61× | 8.8 GB/s | **12.7 GB/s** | 1.45× |
| Upstream's 4 short strings | 8.0 GB/s | 8.1 GB/s | 1.02× | 8.7 GB/s | **10.3 GB/s** | 1.18× |
| Clean, 1 MB | 31.9 GB/s | **41.2 GB/s** | 1.29× | 66.8 GB/s | 66.0 GB/s | 0.99× |
| Clean, 4 KB | 31.1 GB/s | **42.6 GB/s** | 1.37× | 76.5 GB/s | **115.4 GB/s** | 1.51× |
| Clean, 32 B | 7.0 GB/s | **13.3 GB/s** | 1.89× | 9.2 GB/s | **14.4 GB/s** | 1.56× |
| 64 KB, 2% escapes | 6.1 GB/s | **9.8 GB/s** | 1.60× | 11.1 GB/s | **17.9 GB/s** | 1.60× |
| 64 KB, 10% escapes | 1.5 GB/s | **9.0 GB/s** | 6.05× | 2.9 GB/s | **14.2 GB/s** | 4.94× |
| 64 KB, 50% escapes | 0.30 GB/s | **9.1 GB/s** | 30.14× | 0.90 GB/s | **14.0 GB/s** | 15.64× |
| 64 KB, 100% `\u00XX` | 1.8 GB/s | **2.3 GB/s** | 1.27× | 2.4 GB/s | **3.3 GB/s** | 1.36× |

The crate also has optional AVX-512 kernels. Against those on the Zen 5, simd-lang's AVX2 build is still 2.36× faster on AFFiNE and 1.51× faster on rxjs. Tuning it for the CPU with `--cpu znver5` makes clean bulk faster, to 140 GB/s at 4 KB, but escapes slower, at 11.9 GB/s against 13.9, so the default stays at `x86-64-v3`.

Short strings depend heavily on length because each call has a fixed cost, so per-call time is the better measure there: a clean 32 B string takes 2.4 ns on the M2 against the crate's 4.6 ns, and 2.2 ns on Zen 5 against 3.5 ns.

## Where it doesn't win

Of the 61 rows the benchmark measures, simd-lang wins 58 on the M2 and 55 on Zen 5.

**It loses two on each machine:**

- **One 29 B string with one escape, repeated** (0.82× on the M2, 0.81× on Zen 5). The path after the first escape is branchless, while the crate's branches cost little once perfectly predicted. On strings too varied to predict, the same choice wins by up to 1.27× and 1.28×.
- **A fixed list of strings, replayed.** The branch predictor memorizes the crate's branches over the repeated sequence: 0.89× for 4,096 clean 1-32 B strings on the M2, and 0.91× for 4,096 strings of 1-200 B with 2% escapes on Zen 5. With too many unique strings to memorize, the first becomes 1.15×.

**It ties these,** within 2%:

- Upstream's four short strings on the M2 (1.02×), because that 29 B string is one of them. The other three are 1.64-1.80× faster alone. On Zen 5 the batch is 1.18×.
- Clean 64 KB and 1 MB on Zen 5 (1.00× and 0.99×). The cause isn't established, and simd-lang wins at 4 KB (1.51×).
- Unique 16-32 B and 1-32 B strings with 2% escapes on Zen 5 (1.01× each).

Zen 5 timings vary between processes on some short-string rows, and in rare runs the crate or simd-lang is much slower across a whole row group, which looks like an address-layout effect. The table uses medians across restarts for this reason.

## Other x86 builds

The default is one direct AVX2 build, because the alternatives measured slower on Zen 5:

| Build | Compared with default AVX2 | Why you'd use it |
|---|---|---|
| `--isa avx512` | 13% slower overall. Clean 256 B and 4 KB are 37% and 21% faster, clean 1-24 B is 2-3× slower, and 10% escapes is 19% slower | Mostly clean strings of 256 B or more |
| `--isa sse4.2` | Runs on CPUs without AVX2. Slower wherever wide registers help, such as clean 4 KB at 2.3× the time | CPUs without AVX2 |
| `--dispatch sse4.2,avx2` | 2% slower overall, from the indirect jump on every call. Worst row is 19% on replayed 1-32 B strings, or 10% going by the fastest run of each | One binary for any x86-64 CPU made since about 2011 |

All three pass the differential tests, including guard pages. To use dispatch, run `./json-escape/build-kernel.sh --dispatch sse4.2,avx2` and build with `--features dispatch`. `isa()` then says which level was picked. AVX-512 isn't in that list because it would slow down most workloads on this CPU. The cause of the slowdown on short strings isn't established.

## How it works

1. **Optimistic copy.** The hot path assumes nothing needs escaping and uses only immediate constants. The first escape returns through an outlined routine, so clean strings never pay for its table addresses.
2. **Register-sized short path.** A string shorter than a register takes one path, whatever its length, so there's no length to mispredict.
3. **Two overlapping registers, one branch.** `stream ... tail overlap until is_escape(chunk)` tests two registers that may overlap, and only then stores both. That covers every length up to twice a register, the clean loop, and the final chunk, with no remainder loops.
4. **Branchless escapes.** A 256-entry shuffle table turns 8 bytes into up to 16, so 2-byte escapes cost the same at any density. The crate handles each escape in a scalar loop and then rescans.
5. **A cheaper classifier.** `v ^ 2 <= 32` catches the control bytes and `"` with one compare. The compiler reduces `any` of such compares to one register first, leaving one compare of each kind.

## Limitations

- The default x86 build requires AVX2, while the crate detects the CPU and also runs on older ones. See [Other x86 builds](#other-x86-builds).
- A short input can be read past its end, up to one register minus a byte, always within its last page. The crate over-reads the same way, with the same page check.
- The output buffer needs room for `6n + 35` bytes on ARM, the same as the crate. On x86 it's `6n + 67`, which allows for AVX-512's wider stores in every build.

## Testing

`tests/differential.rs` compares every result with the crate and with `serde_json`: every ASCII byte at every position, every length to 600, 200,000 random strings, and inputs and outputs flush against guard pages. To check that those tests can fail, 22 deliberate bugs were injected into the kernel one at a time, and the suite caught all 22 on both architectures.

`check-same-code.sh` compiles `json_escape.simd` and `by_hand.simd` for both architectures and fails unless the object files match exactly, so streams, computed tables, slices and `set_bits` cost nothing. It doesn't freeze the kernel: a change to the algorithm goes in both files, and has to show no slower rows in the benchmark. The hand-written kernel is 777 lines, mostly literal tables, and the high-level one is 198. Its entry point went from 34 lines of code to 12.

## Files

| File | Purpose |
|---|---|
| `json_escape.simd` | The kernel, using streams and tables computed at compile time |
| `by_hand.simd` | The same kernel with hand-written loops and literal tables, kept as the reference |
| `check-same-code.sh` | Fails unless those two compile to identical object files |
| `src/lib.rs` | Rust `escape` and `escape_into` API, matching the crate's |
| `src/main.rs` | The head-to-head benchmark |
| `tests/differential.rs` | Tests against the crate and `serde_json`, including guard pages |
| `build-kernel.sh` | Compiles the kernel into `generated/` |
| `fetch-fixtures.sh` | Downloads upstream's rxjs and AFFiNE payloads |

## Reproducing

```bash
./json-escape/fetch-fixtures.sh    # upstream's payloads, about 150 MB
./json-escape/build-kernel.sh      # see the root README for the LLVM 21 setup
./json-escape/check-same-code.sh   # streams and hand-written loops give identical code
cd json-escape
cargo test --release               # differential and guard-page tests
cargo run --release --bin bench 2>&1 >/dev/null                # every row
cargo run --release --bin bench -- "density" 2>&1 >/dev/null   # groups matching a substring
cargo run --release --bin bench --features crate-avx512 2>&1 >/dev/null   # against AVX-512
```

On Linux, pin the benchmark to a core with `taskset`.
