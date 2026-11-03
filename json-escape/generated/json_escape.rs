// Auto-generated bindings for json_escape.simd
// Do not edit — regenerate with: simd-lang compile

#[link(name = "json_escape", kind = "static")]
extern "C" {
    #[link_name = "json_escape"]
    fn __simd_json_escape(src: *mut u8, n: i64, dst: *mut u8) -> i64;
}

/// `json_escape` from json_escape.simd
pub fn json_escape(src: &mut [u8], n: i64, dst: &mut [u8]) -> i64 {
    unsafe {
        __simd_json_escape(src.as_mut_ptr(), n, dst.as_mut_ptr())
    }
}

