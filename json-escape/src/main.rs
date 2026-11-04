//! Head-to-head: simd-lang's kernel vs the json-escape-simd crate.
//!
//! Each round times one implementation then the other, flipping the order every
//! round, so thermal and scheduler drift hits both alike. Reports medians.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

struct Row {
    group: &'static str,
    name: String,
    bytes: usize,
    crate_ns: f64,
    ours_ns: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn race(mut a: impl FnMut(), mut b: impl FnMut(), budget_ms: f64, rounds: usize) -> (f64, f64) {
    // Pick an iteration count that fills the per-round budget.
    let t = Instant::now();
    a();
    let once = t.elapsed().as_secs_f64() * 1e3;
    let iters = ((budget_ms / once.max(1e-6)) as usize).clamp(1, 2_000_000);

    let (mut xs, mut ys) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
    let timed = |f: &mut dyn FnMut() -> ()| {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        t.elapsed().as_secs_f64() * 1e9 / iters as f64
    };
    for round in 0..rounds {
        if round % 2 == 0 {
            xs.push(timed(&mut a));
            ys.push(timed(&mut b));
        } else {
            ys.push(timed(&mut b));
            xs.push(timed(&mut a));
        }
    }
    (median(xs), median(ys))
}

fn kernel(rows: &mut Vec<Row>, group: &'static str, name: &str, sources: &[String], budget: f64, rounds: usize) {
    if skipped(group) {
        return;
    }
    let bytes: usize = sources.iter().map(|s| s.len()).sum();
    let cap = sources.iter().map(|s| simd_json_escape::scratch_len(s.len())).max().unwrap_or(64);
    let mut buf_a: Vec<u8> = Vec::with_capacity(cap);
    let mut buf_b: Vec<u8> = Vec::with_capacity(cap);
    let (crate_ns, ours_ns) = race(
        || {
            for s in sources {
                buf_a.clear();
                json_escape_simd::escape_into(black_box(s.as_str()), &mut buf_a);
                black_box(buf_a.as_ptr());
            }
        },
        || {
            for s in sources {
                buf_b.clear();
                simd_json_escape::escape_into(black_box(s.as_str()), &mut buf_b);
                black_box(buf_b.as_ptr());
            }
        },
        budget,
        rounds,
    );
    rows.push(Row { group, name: name.to_string(), bytes, crate_ns, ours_ns });
}

fn allocating(rows: &mut Vec<Row>, group: &'static str, name: &str, sources: &[String], budget: f64, rounds: usize) {
    if skipped(group) {
        return;
    }
    let bytes: usize = sources.iter().map(|s| s.len()).sum();
    let (crate_ns, ours_ns) = race(
        || {
            for s in sources {
                black_box(json_escape_simd::escape(black_box(s.as_str())));
            }
        },
        || {
            for s in sources {
                black_box(simd_json_escape::escape(black_box(s.as_str())));
            }
        },
        budget,
        rounds,
    );
    rows.push(Row { group, name: name.to_string(), bytes, crate_ns, ours_ns });
}

/// `bench <substring>` runs only the groups whose names contain it.
fn skipped(group: &str) -> bool {
    std::env::args().nth(1).is_some_and(|only| !group.contains(&only))
}

fn collect(dir: &Path, exts: &[&str], out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let Ok(meta) = std::fs::symlink_metadata(&p) else { continue };
        if meta.is_dir() {
            collect(&p, exts, out);
        } else if meta.is_file() && p.extension().and_then(|x| x.to_str()).is_some_and(|x| exts.contains(&x)) {
            out.push(p);
        }
    }
}

fn load(dir: &Path, exts: &[&str]) -> Vec<String> {
    let mut paths = Vec::new();
    collect(dir, exts, &mut paths);
    paths.sort();
    paths.iter().filter_map(|p| std::fs::read_to_string(p).ok()).collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// `len` bytes of lowercase text with escapes at random positions, at `per_mille`/1000.
fn with_density(len: usize, per_mille: u64, escapes: &[u8], seed: u64) -> String {
    let mut rng = Rng(seed);
    let bytes: Vec<u8> = (0..len)
        .map(|_| {
            if rng.below(1000) < per_mille {
                escapes[rng.below(escapes.len() as u64) as usize]
            } else {
                b'a' + rng.below(26) as u8
            }
        })
        .collect();
    String::from_utf8(bytes).unwrap()
}

fn main() {
    // Keep the benchmark on the performance cores.
    #[cfg(target_os = "macos")]
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let mut rows = Vec::new();
    let (budget, rounds) = (4.0, 41);

    // Upstream's three payloads, from crates/json-escape-simd/benches/escape.rs.
    let short: Vec<String> = [
        "Hello, world!",
        "abcdefghijklmnopqrstuvwxyz .*? hello world escape json string",
        "normal string 🥹",
        "中文 English 🚀 \n❓ 𝄞",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let rxjs = load(&root.join("rxjs/package/src"), &["ts"]);
    let affine = load(&root.join("affine"), &["ts", "tsx", "js", "mjs", "cjs"]);
    assert!(!rxjs.is_empty() && !affine.is_empty(), "fixtures missing under {}", root.display());

    for (name, src, b, r) in [
        ("short strings", &short, budget, rounds),
        ("rxjs (251 files)", &rxjs, 20.0, 21),
        ("fixtures/AFFiNE", &affine, 60.0, 11),
    ] {
        kernel(&mut rows, "upstream payloads: escape_into, reused buffer", name, src, b, r);
        allocating(&mut rows, "upstream payloads: escape(), allocating String", name, src, b, r);
    }

    for s in &short {
        let label: String = s.chars().take(14).collect::<String>().replace('\n', " ");
        kernel(&mut rows, "upstream short strings, one at a time", &format!("{} B {label}", s.len()), std::slice::from_ref(s), budget, rounds);
    }

    for len in [0usize, 1, 2, 4, 7, 8, 12, 15, 16, 17, 24, 31, 32, 48, 64, 100, 256, 1024, 4096, 65536, 1 << 20] {
        let s = vec![with_density(len, 0, b"", 7)];
        kernel(&mut rows, "clean input, by length", &format!("{len} B"), &s, budget, rounds);
    }

    // Mixed lengths defeat the branch predictor, as real JSON keys and values do.
    let mut rng = Rng(99);
    let keys: Vec<String> = (0..4096).map(|i| with_density(1 + rng.below(32) as usize, 0, b"", i + 1)).collect();
    kernel(&mut rows, "replayed short strings, mixed lengths", "1-32 B clean", &keys, budget, rounds);
    let keys: Vec<String> = (0..4096).map(|i| with_density(1 + rng.below(32) as usize, 20, b"\"\\\n", i + 1)).collect();
    kernel(&mut rows, "replayed short strings, mixed lengths", "1-32 B, 2% escapes", &keys, budget, rounds);
    let keys: Vec<String> = (0..4096).map(|i| with_density(1 + rng.below(200) as usize, 20, b"\"\\\n", i + 1)).collect();
    kernel(&mut rows, "replayed short strings, mixed lengths", "1-200 B, 2% escapes", &keys, budget, rounds);

    // Long enough that the branch predictor can't memorize the sequence of lengths.
    let mixed = |lo: u64, hi: u64, rng: &mut Rng, per_mille: u64| -> Vec<String> {
        (0..(1 << 18)).map(|i| with_density((lo + rng.below(hi - lo + 1)) as usize, per_mille, b"\"\\\n", i + 1)).collect()
    };
    if !skipped("unique short strings") {
        for (lo, hi) in [(1u64, 15u64), (16, 32), (33, 64), (1, 32), (1, 64)] {
            let keys = mixed(lo, hi, &mut rng, 0);
            kernel(&mut rows, "unique short strings, too many to memorize", &format!("{lo}-{hi} B clean"), &keys, 8.0, 15);
            let keys = mixed(lo, hi, &mut rng, 20);
            kernel(&mut rows, "unique short strings, too many to memorize", &format!("{lo}-{hi} B, 2%"), &keys, 8.0, 15);
        }
    }

    for per_mille in [0u64, 1, 5, 10, 20, 50, 100, 250, 500, 1000] {
        let s = vec![with_density(64 * 1024, per_mille, b"\"\\\n", 11)];
        let label = format!("{:.1}%", per_mille as f64 / 10.0);
        kernel(&mut rows, "64 KB, density of 2-byte escapes", &label, &s, budget, rounds);
    }
    for per_mille in [1u64, 10, 50, 100, 250, 500, 1000] {
        let s = vec![with_density(64 * 1024, per_mille, &[0x01, 0x1b, 0x07], 13)];
        let label = format!("{:.1}%", per_mille as f64 / 10.0);
        kernel(&mut rows, "64 KB, density of 6-byte \\u00XX escapes", &label, &s, budget, rounds);
    }

    let mut group = "";
    println!("{{\"rows\":[");
    let mut first = true;
    for r in &rows {
        if r.group != group {
            group = r.group;
            eprintln!("\n{group}\n{:<22} {:>13} {:>13} {:>10} {:>10} {:>8}", "", "crate ns", "simd-lang ns", "crate MB/s", "ours MB/s", "speedup");
        }
        let mbs = |ns: f64| r.bytes as f64 / ns * 1e3;
        eprintln!(
            "{:<22} {:>13.1} {:>13.1} {:>10.0} {:>10.0} {:>7.2}x",
            r.name, r.crate_ns, r.ours_ns, mbs(r.crate_ns), mbs(r.ours_ns), r.crate_ns / r.ours_ns
        );
        println!(
            "{}{{\"group\":{:?},\"name\":{:?},\"bytes\":{},\"crate_ns\":{:.2},\"ours_ns\":{:.2}}}",
            if first { "" } else { "," },
            r.group, r.name, r.bytes, r.crate_ns, r.ours_ns
        );
        first = false;
    }
    println!("]}}");
}
