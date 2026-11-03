//! Point the linker at the AOT-compiled `libjson_escape.a`.
use std::path::PathBuf;

fn main() {
    let generated = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("generated");
    let lib = generated.join("libjson_escape.a");
    if !lib.exists() {
        panic!("missing {}\n\nRun ./json-escape/build-kernel.sh first", lib.display());
    }
    println!("cargo:rustc-link-search=native={}", generated.display());
    println!("cargo:rustc-link-lib=static=json_escape");
    println!("cargo:rerun-if-changed={}", lib.display());
}
