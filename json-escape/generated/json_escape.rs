// Auto-generated bindings for json_escape.simd
// Do not edit — regenerate with: simd-lang compile

#[link(name = "json_escape", kind = "static")]
extern "C" {
    #[link_name = "json_escape"]
    fn __simd_json_escape(src: *mut u8, src__len: i64, dst: *mut u8) -> i64;
}

/// `json_escape` from json_escape.simd
pub fn json_escape(src: &mut [u8], dst: &mut [u8]) -> i64 {
    unsafe {
        __simd_json_escape(src.as_mut_ptr(), src.len() as i64, dst.as_mut_ptr())
    }
}

