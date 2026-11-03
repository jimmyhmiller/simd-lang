//! JSON string escaping backed by the simd-lang kernel `json_escape.simd`.
//! Same API and same output as the `json-escape-simd` crate.

extern "C" {
    fn json_escape(src: *const u8, n: i64, dst: *mut u8) -> i64;
}

/// Bytes `value` may touch in the output buffer: the kernel's stores overshoot.
#[inline(always)]
pub const fn scratch_len(input_len: usize) -> usize {
    input_len * 6 + 32 + 3
}

/// # Safety
/// `dst` must be valid for writes of `scratch_len(value.len())` bytes.
#[inline(always)]
pub unsafe fn escape_raw(value: &[u8], dst: *mut u8) -> usize {
    json_escape(value.as_ptr(), value.len() as i64, dst) as usize
}

/// Escapes `value` as a JSON string, including the surrounding `"` quotes.
pub fn escape(value: &str) -> String {
    let mut buf = Vec::with_capacity(scratch_len(value.len()));
    unsafe {
        let len = escape_raw(value.as_bytes(), buf.as_mut_ptr());
        buf.set_len(len);
        String::from_utf8_unchecked(buf)
    }
}

/// Escapes `value` (including the surrounding `"`) and appends the result to `dst`.
pub fn escape_into<S: AsRef<str>>(value: S, dst: &mut Vec<u8>) {
    let value = value.as_ref();
    dst.reserve(scratch_len(value.len()));
    let old_len = dst.len();
    unsafe {
        let len = escape_raw(value.as_bytes(), dst.as_mut_ptr().add(old_len));
        dst.set_len(old_len + len);
    }
}
