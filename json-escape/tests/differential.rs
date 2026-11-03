//! Output must be byte-identical to json-escape-simd and to serde_json.

use simd_json_escape::{escape, escape_into, escape_raw, scratch_len};

fn check(s: &str) {
    let ours = escape(s);
    assert_eq!(ours, json_escape_simd::escape(s), "differs from crate on {s:?}");
    assert_eq!(ours, serde_json::to_string(s).unwrap(), "differs from serde_json on {s:?}");
}

/// A small deterministic generator, so failures reproduce.
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

#[test]
fn every_ascii_byte_at_every_position() {
    for c in 0u8..128 {
        for len in 1..=130usize {
            for pos in [0, len / 2, len - 1] {
                let mut bytes = vec![b'a'; len];
                bytes[pos] = c;
                check(std::str::from_utf8(&bytes).unwrap());
            }
        }
    }
}

#[test]
fn every_length_all_clean_and_all_escaped() {
    for len in 0..=600usize {
        for fill in [b'x', b'"', b'\\', b'\n', 0x00, 0x1f, 0x7f] {
            check(std::str::from_utf8(&vec![fill; len]).unwrap());
        }
    }
}

#[test]
fn control_bytes_mixed_with_short_escapes_in_one_block() {
    for a in 0u8..32 {
        for b in [b'"', b'\\', b'\n', b'\t', b'z'] {
            for len in [3usize, 15, 16, 17, 31, 32, 33, 64, 65, 80] {
                let mut bytes = vec![b'q'; len];
                bytes[0] = b;
                bytes[len / 2] = a;
                bytes[len - 1] = b;
                check(std::str::from_utf8(&bytes).unwrap());
            }
        }
    }
}

#[test]
fn random_strings() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let alphabets: [&[char]; 5] = [
        &['a', 'b', ' ', '"', '\\', '\n'],
        &['\u{0}', '\u{1}', '\u{1b}', '\u{1f}', 'x', '\u{7f}'],
        &['中', '文', '🚀', 'é', '\t', '"', 'k'],
        &['\u{8}', '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', '\u{e}', 'y'],
        &['p'],
    ];
    for _ in 0..200_000 {
        let alphabet = alphabets[rng.below(5) as usize];
        // Mostly clean runs punctuated by alphabet characters, at varying density.
        let density = [1, 2, 8, 64, 512][rng.below(5) as usize];
        let len = match rng.below(4) {
            0 => rng.below(20),
            1 => rng.below(70),
            2 => rng.below(300),
            _ => rng.below(1500),
        };
        let mut s = String::new();
        for _ in 0..len {
            if rng.below(density) == 0 {
                s.push(alphabet[rng.below(alphabet.len() as u64) as usize]);
            } else {
                s.push((b'a' + rng.below(26) as u8) as char);
            }
        }
        check(&s);
    }
}

#[test]
fn appends_to_existing_contents() {
    let mut dst = b"prefix:".to_vec();
    escape_into("a\"b", &mut dst);
    assert_eq!(dst, b"prefix:\"a\\\"b\"");
}

/// Maps three pages with the first and last inaccessible, so a read or write
/// outside the middle page faults and fails the test run.
struct Guarded {
    base: *mut u8,
    page: usize,
}
impl Guarded {
    fn new() -> Self {
        unsafe {
            let page = libc::sysconf(libc::_SC_PAGESIZE) as usize;
            let base = libc::mmap(
                std::ptr::null_mut(),
                page * 3,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            ) as *mut u8;
            assert!(base as isize != -1, "mmap failed");
            assert_eq!(libc::mprotect(base as _, page, libc::PROT_NONE), 0);
            assert_eq!(libc::mprotect(base.add(page * 2) as _, page, libc::PROT_NONE), 0);
            Guarded { base, page }
        }
    }
    fn page(&self) -> *mut u8 {
        unsafe { self.base.add(self.page) }
    }
}
impl Drop for Guarded {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base as _, self.page * 3) };
    }
}

fn escape_at(src: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; scratch_len(src.len())];
    let len = unsafe { escape_raw(src, out.as_mut_ptr()) };
    out.truncate(len);
    out
}

#[test]
fn input_flush_against_either_page_edge() {
    let g = Guarded::new();
    for len in 0..=200usize {
        for fill in [b'x', b'"', 0x01u8] {
            let expected = serde_json::to_string(std::str::from_utf8(&vec![fill; len]).unwrap()).unwrap();
            unsafe {
                // Ends exactly at the guard page above.
                let end = std::slice::from_raw_parts_mut(g.page().add(g.page - len), len);
                end.fill(fill);
                assert_eq!(escape_at(end), expected.as_bytes(), "at page end, len {len}");
                // Starts exactly at the guard page below.
                let start = std::slice::from_raw_parts_mut(g.page(), len);
                start.fill(fill);
                assert_eq!(escape_at(start), expected.as_bytes(), "at page start, len {len}");
            }
        }
    }
}

#[test]
fn output_never_passes_the_documented_bound() {
    let g = Guarded::new();
    for len in 0..=(g.page - 35) / 6 {
        for fill in [b'x', b'"', 0x01u8, b'\n'] {
            let src = vec![fill; len];
            let expected = serde_json::to_string(std::str::from_utf8(&src).unwrap()).unwrap();
            unsafe {
                // The scratch region ends exactly at the guard page.
                let dst = g.page().add(g.page - scratch_len(len));
                let n = escape_raw(&src, dst);
                let out = std::slice::from_raw_parts(dst, n);
                assert_eq!(out, expected.as_bytes(), "len {len}");
            }
        }
    }
}
