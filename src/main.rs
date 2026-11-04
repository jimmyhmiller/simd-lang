mod ast;
mod codegen;
mod comptime;
mod consteval;
mod desugar;
mod dispatch;
mod streams;
#[cfg(test)]
mod json;
mod lexer;
mod parser;
mod slices;
mod typecheck;

use melior::ir::operation::OperationLike;
use std::collections::HashMap;
use std::path::PathBuf;

fn main() {
    // The compiler reports errors in programs by panicking with a message, so print just
    // that. RUST_BACKTRACE=1 restores the full report, for debugging the compiler itself.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::panic::set_hook(Box::new(|info| {
            if typecheck::trying_widths() {
                return;
            }
            let payload = info.payload();
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| info.to_string());
            eprintln!("error: {}", message);
        }));
    }

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: simd-lang compile <file.simd> [-o <output_dir>] [--target <triple>] [--cpu <cpu>] [--isa <level>] [--dispatch <levels>]");
        eprintln!("       simd-lang compile <file.simd> --emit-mlir");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "compile" => cmd_compile(&args[2..]),
        "jit-stage1" => {
            let file = args.get(2).map(String::as_str).unwrap_or_else(|| {
                eprintln!("Usage: simd-lang jit-stage1 <file.js> [iters]");
                std::process::exit(1);
            });
            let iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
            cmd_jit_stage1(file, iters);
        }
        other => {
            eprintln!("Unknown command: {}", other);
            std::process::exit(1);
        }
    }
}

/// JIT-compile `examples/js_stage1.simd` and time *only* the stage-1 kernel on
/// `path`, reporting MB/s in the same shape `aot-bench` prints — so the AOT and
/// JIT numbers line up directly. This is the "JIT version" the AOT artifact is
/// measured against. No oxc, no stage-2 — just bytes → two bitmaps.
fn cmd_jit_stage1(path: &str, iters: usize) {
    use std::time::Instant;

    // Same memref ABI as the AOT-generated bindings / bench::Stage1Fn.
    type Stage1Fn = extern "C" fn(
        *const u8, *const u8, i64, i64, i64, // input       memref<?xi8>
        *mut u64, *mut u64, i64, i64, i64,   // start_masks memref<?xi64>
        *mut u64, *mut u64, i64, i64, i64,   // word_masks  memref<?xi64>
    ) -> i32; // token-start count

    let raw = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let bytes = raw.len();
    let mb = bytes as f64 / 1_000_000.0;
    // Pad to a multiple of 64 so the kernel reads whole chunks.
    let mut padded = raw.clone();
    while padded.len() % 64 != 0 {
        padded.push(0);
    }

    // ── JIT-compile js_stage1 (timed: this is the cost paid every launch) ──
    let source = std::fs::read_to_string("examples/js_stage1.simd")
        .expect("cannot read examples/js_stage1.simd (run from project root)");
    let t_compile = Instant::now();
    let ctx = codegen::create_context();
    let items = parser::parse(&source);
    let mut module = codegen::compile_module(&ctx, &items, &HashMap::new());
    codegen::lower_to_llvm(&ctx, &mut module).expect("lowering failed");
    let engine = melior::ExecutionEngine::new(&module, 3, &[], false);
    let fptr = engine.lookup("js_stage1");
    assert!(!fptr.is_null(), "js_stage1 symbol not found");
    let stage1: Stage1Fn = unsafe { std::mem::transmute(fptr) };
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1e3;

    let nchunks = padded.len() / 64 + 1;
    let mut start_masks = vec![0u64; nchunks];
    let mut word_masks = vec![0u64; nchunks];

    let call = |start: &mut [u64], word: &mut [u64]| -> i32 {
        stage1(
            padded.as_ptr(), padded.as_ptr(), 0, padded.len() as i64, 1,
            start.as_mut_ptr(), start.as_mut_ptr(), 0, start.len() as i64, 1,
            word.as_mut_ptr(), word.as_mut_ptr(), 0, word.len() as i64, 1,
        )
    };

    let token_starts = call(&mut start_masks, &mut word_masks);
    for _ in 0..5 {
        std::hint::black_box(call(&mut start_masks, &mut word_masks));
    }

    let mut best = 0.0f64;
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(call(&mut start_masks, &mut word_masks));
        let mbps = mb / t.elapsed().as_secs_f64();
        best = best.max(mbps);
        samples.push(mbps);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];

    println!("=== JIT (MLIR ExecutionEngine, O3, host CPU) ===");
    println!("file            : {path}  ({bytes} bytes, {:.2} MB)", mb);
    println!("token starts    : {token_starts}");
    println!("JIT compile time: {compile_ms:.1} ms  (paid every launch)");
    println!("stage-1 MB/s    : {median:.0} median   {best:.0} best   ({iters} iters)");
}

fn cmd_compile(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: simd-lang compile <file.simd> [-o <output_dir>] [--emit-mlir] [--target <triple>] [--cpu <cpu>]");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[0]);
    let stem = input_path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let mut output_dir = PathBuf::from(".");
    let mut emit_mlir = false;
    let mut triple = codegen::Arch::host_triple().to_string();
    let mut cpu: Option<String> = None;
    let mut isa: Option<codegen::X86Isa> = None;
    let mut dispatch: Option<Vec<codegen::X86Isa>> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                output_dir = PathBuf::from(&args[i]);
            }
            "--emit-mlir" => {
                emit_mlir = true;
            }
            "--target" => {
                i += 1;
                triple = args[i].clone();
            }
            "--cpu" => {
                i += 1;
                cpu = Some(args[i].clone());
            }
            "--isa" => {
                i += 1;
                isa = Some(codegen::X86Isa::from_name(&args[i]).unwrap_or_else(|| {
                    eprintln!("Unknown --isa {}: use sse4.2, avx2 or avx512", args[i]);
                    std::process::exit(1);
                }));
            }
            "--dispatch" => {
                i += 1;
                let isas = args[i]
                    .split(',')
                    .map(|name| {
                        codegen::X86Isa::from_name(name).unwrap_or_else(|| {
                            eprintln!("Unknown --dispatch level {}: use sse4.2, avx2 or avx512", name);
                            std::process::exit(1);
                        })
                    })
                    .collect();
                dispatch = Some(isas);
            }
            other => {
                eprintln!("Unknown option: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Read source
    let source = std::fs::read_to_string(&input_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", input_path.display(), e);
        std::process::exit(1);
    });

    let arch = codegen::Arch::of_triple(&triple).unwrap_or_else(|| {
        eprintln!("Unsupported target: {} (expected aarch64-* or x86_64-*)", triple);
        std::process::exit(1);
    });
    let arch = match (arch, isa) {
        (arch, None) => arch,
        (codegen::Arch::X86_64(_), Some(isa)) => codegen::Arch::X86_64(isa),
        (codegen::Arch::Aarch64, Some(_)) => {
            eprintln!("--isa only applies to x86_64 targets");
            std::process::exit(1);
        }
    };

    // Parse
    let items = parser::parse(&source);

    // Create output directory
    if !emit_mlir {
        std::fs::create_dir_all(&output_dir).unwrap_or_else(|e| {
            eprintln!("Error creating output directory: {}", e);
            std::process::exit(1);
        });
    }

    let mut objects = Vec::new();
    if let Some(isas) = &dispatch {
        if !matches!(arch, codegen::Arch::X86_64(_)) {
            eprintln!("--dispatch only applies to x86_64 targets");
            std::process::exit(1);
        }
        if emit_mlir || cpu.is_some() || isa.is_some() {
            eprintln!("--dispatch can't be combined with --emit-mlir, --cpu or --isa");
            std::process::exit(1);
        }
        for &level in isas {
            let renamed = dispatch::with_isa_suffix(&items, level);
            let suffix = dispatch::isa_suffix(level);
            let obj_path = output_dir.join(format!("{}_{}.o", stem, suffix));
            let unit = CompileUnit {
                input_path: &input_path,
                output_dir: &output_dir,
                stem: &format!("{}_{}", stem, suffix),
                obj_path: &obj_path,
                triple: &triple,
                arch: codegen::Arch::X86_64(level),
                cpu: None,
            };
            compile_unit(&renamed, &unit, false);
            objects.push(obj_path);
        }
        let c_path = output_dir.join(format!("{}_dispatch.c", stem));
        let obj_path = output_dir.join(format!("{}_dispatch.o", stem));
        std::fs::write(&c_path, dispatch::dispatcher_source(&stem, &items, isas)).unwrap();
        let target_arg = format!("--target={}", triple);
        let status = std::process::Command::new(llvm_tool("clang"))
            .args(["-O3", "-fno-stack-protector", target_arg.as_str(), "-c"])
            .args([c_path.to_str().unwrap(), "-o", obj_path.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| {
                eprintln!("cannot run clang: {}", e);
                std::process::exit(1);
            });
        if !status.success() {
            eprintln!("compiling the dispatcher failed");
            std::process::exit(1);
        }
        if std::env::var_os("SIMD_LANG_KEEP_IR").is_none() {
            let _ = std::fs::remove_file(&c_path);
        }
        objects.push(obj_path);
    } else {
        let obj_path = output_dir.join(format!("{}.o", stem));
        let unit = CompileUnit {
            input_path: &input_path,
            output_dir: &output_dir,
            stem: &stem,
            obj_path: &obj_path,
            triple: &triple,
            arch,
            cpu: cpu.as_deref(),
        };
        compile_unit(&items, &unit, emit_mlir);
        if emit_mlir {
            return;
        }
        objects.push(obj_path);
    }

    // Archive into static library
    let lib_path = output_dir.join(format!("lib{}.a", stem));
    // `rcs` adds to an archive that exists, which would keep members of an older build.
    let _ = std::fs::remove_file(&lib_path);
    let archiver = llvm_tool("llvm-ar");
    let status = std::process::Command::new(&archiver)
        .args(["rcs", lib_path.to_str().unwrap()])
        .args(objects.iter().map(|o| o.to_str().unwrap()))
        .status()
        .unwrap_or_else(|e| {
            eprintln!("cannot run {}: {}", archiver.display(), e);
            std::process::exit(1);
        });
    if !status.success() {
        eprintln!("{} failed", archiver.display());
        std::process::exit(1);
    }

    // Generate Rust bindings
    let rs_path = output_dir.join(format!("{}.rs", stem));
    let bindings = generate_rust_bindings(&stem, &items);
    std::fs::write(&rs_path, bindings).unwrap();

    // Clean up object files
    for object in &objects {
        let _ = std::fs::remove_file(object);
    }

    eprintln!("Compiled {} →", input_path.display());
    eprintln!("  {}", lib_path.display());
    eprintln!("  {}", rs_path.display());
}

/// One object to produce: the program, built for one instruction set.
struct CompileUnit<'a> {
    input_path: &'a std::path::Path,
    output_dir: &'a std::path::Path,
    stem: &'a str,
    obj_path: &'a std::path::Path,
    triple: &'a str,
    arch: codegen::Arch,
    cpu: Option<&'a str>,
}

/// Type check and compile `items` to an object file, or print the MLIR instead.
fn compile_unit(items: &[ast::Item], unit: &CompileUnit, emit_mlir: bool) {
    if let Err(errors) = typecheck::typecheck_for(items, &HashMap::new(), unit.arch) {
        for e in &errors {
            eprintln!("{}: {}", unit.input_path.display(), e);
        }
        std::process::exit(1);
    }

    let ctx = codegen::create_context();
    let mut module = codegen::compile_module_for(&ctx, items, &HashMap::new(), unit.arch);

    if !module.as_operation().verify() {
        eprintln!("MLIR verification failed:");
        eprintln!("{}", module.as_operation());
        std::process::exit(1);
    }

    if emit_mlir {
        println!("{}", module.as_operation());
        return;
    }

    codegen::lower_to_llvm_for(&ctx, &mut module, unit.triple).unwrap_or_else(|e| {
        eprintln!("LLVM lowering failed: {}", e);
        std::process::exit(1);
    });

    // Produce the object file with the LLVM tools rather than the JIT's
    // ExecutionEngine: the JIT uses the large code model, where every constant-pool
    // and table access goes through the GOT, and it adds a `_mlir_` wrapper per fn.
    compile_to_object(&module, unit.output_dir, unit.stem, unit.obj_path, unit.triple, unit.arch, unit.cpu)
        .unwrap_or_else(|e| {
            eprintln!("object generation failed: {}", e);
            std::process::exit(1);
        });
}

/// The LLVM install matching the MLIR this compiler links against.
fn llvm_tool(name: &str) -> PathBuf {
    let default = if cfg!(target_os = "macos") { "/opt/homebrew/opt/llvm@21" } else { "/usr/lib/llvm-21" };
    let prefix = option_env!("MLIR_SYS_210_PREFIX").unwrap_or(default);
    PathBuf::from(prefix).join("bin").join(name)
}

/// Lowered MLIR → LLVM IR (`mlir-translate`) → optimized object (`clang -O3`).
fn compile_to_object(
    module: &melior::ir::Module,
    output_dir: &std::path::Path,
    stem: &str,
    obj_path: &std::path::Path,
    triple: &str,
    arch: codegen::Arch,
    cpu: Option<&str>,
) -> Result<(), String> {
    let mlir_path = output_dir.join(format!("{}.llvm.mlir", stem));
    let ll_path = output_dir.join(format!("{}.ll", stem));
    std::fs::write(&mlir_path, module.as_operation().to_string())
        .map_err(|e| format!("cannot write {}: {}", mlir_path.display(), e))?;

    let run = |tool: &str, args: &[&str]| -> Result<(), String> {
        let path = llvm_tool(tool);
        let status = std::process::Command::new(&path)
            .args(args)
            .status()
            .map_err(|e| format!("cannot run {}: {}", path.display(), e))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("{} failed: {}", path.display(), status))
        }
    };
    run(
        "mlir-translate",
        &["--mlir-to-llvmir", mlir_path.to_str().unwrap(), "-o", ll_path.to_str().unwrap()],
    )?;
    let target_arg = format!("--target={}", triple);
    let cpu_args: Vec<String> = match (arch, cpu) {
        (codegen::Arch::Aarch64, Some(cpu)) => vec![format!("-mcpu={}", cpu)],
        (codegen::Arch::Aarch64, None) if triple.contains("apple") => vec!["-mcpu=apple-m1".to_string()],
        (codegen::Arch::Aarch64, None) => vec![],
        (codegen::Arch::X86_64(_), Some(cpu)) => vec![format!("-march={}", cpu)],
        // The instructions `Arch::X86_64` promises.
        (codegen::Arch::X86_64(isa), None) => vec![format!("-march={}", isa.march()), "-mpclmul".to_string()],
    };
    let mut clang_args = vec!["-x", "ir", "-O3", "-Wno-override-module", target_arg.as_str()];
    clang_args.extend(cpu_args.iter().map(String::as_str));
    clang_args.extend(["-c", ll_path.to_str().unwrap(), "-o", obj_path.to_str().unwrap()]);
    run("clang", &clang_args)?;
    if std::env::var_os("SIMD_LANG_KEEP_IR").is_none() {
        let _ = std::fs::remove_file(&mlir_path);
        let _ = std::fs::remove_file(&ll_path);
    }
    Ok(())
}

/// Generate safe Rust wrapper code for all functions in a .simd file.
fn generate_rust_bindings(module_name: &str, items: &[ast::Item]) -> String {
    let mut out = String::new();
    // Regular `//` comments (not `//!`): these bindings are meant to be pulled in
    // via `include!`, where inner doc comments mid-module are a compile error.
    out.push_str(&format!(
        "// Auto-generated bindings for {}.simd\n",
        module_name
    ));
    out.push_str("// Do not edit — regenerate with: simd-lang compile\n\n");

    // Collect function signatures
    let fns: Vec<&ast::FnDef> = items
        .iter()
        .filter_map(|item| match item {
            ast::Item::Fn(f) if !f.private && f.comptime_params.is_empty() && has_binding(f) => Some(f),
            _ => None,
        })
        .collect();

    if fns.is_empty() {
        return out;
    }

    // Extern block with mangled names to avoid conflicts with safe wrappers
    out.push_str(&format!(
        "#[link(name = \"{}\", kind = \"static\")]\n",
        module_name
    ));
    out.push_str("extern \"C\" {\n");

    for f in &fns {
        let (extern_params, _) = build_extern_params(f);
        let extern_ret = build_extern_return(f);
        out.push_str(&format!(
            "    #[link_name = \"{}\"]\n    fn __simd_{}({}){};\n",
            f.name, f.name, extern_params, extern_ret
        ));
    }
    out.push_str("}\n\n");

    // Safe wrappers
    for f in &fns {
        generate_safe_wrapper(&mut out, f);
    }

    out
}

/// Pointers and structs come back through memory, which has no plain Rust signature.
fn has_binding(f: &ast::FnDef) -> bool {
    f.ret_ty.as_ref().is_none_or(|ret| {
        matches!(ret.name.as_str(), "f32" | "f64" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "bool")
    })
}

/// Determine whether a parameter is a ptr type (memref) or a vector type.
fn is_ptr_param(p: &ast::Param) -> bool {
    p.ty.name == "ptr"
}

/// Get the element type name for a ptr[T] parameter.
fn ptr_elem(p: &ast::Param) -> &str {
    match &p.ty.width {
        Some(ast::Width::Param(name)) => name.as_str(),
        _ => "u8",
    }
}

/// Map a simd-lang scalar type name to Rust type.
fn scalar_to_rust(name: &str) -> &str {
    match name {
        "u8" => "u8",
        "u16" => "u16",
        "u32" => "u32",
        "u64" => "u64",
        "i8" => "i8",
        "i16" => "i16",
        "i32" => "i32",
        "i64" => "i64",
        "f32" => "f32",
        "f64" => "f64",
        "bool" => "u8", // i1 in LLVM, passed as i8
        _ => "u8",
    }
}

/// Build the extern "C" parameter list for a function.
/// ptr[T] expands to 5 params (alloc, align, offset, size, stride).
/// Vector types are passed directly.
fn build_extern_params(f: &ast::FnDef) -> (String, Vec<String>) {
    let mut params = Vec::new();
    let mut param_names = Vec::new();

    for p in &f.params {
        if is_ptr_param(p) && f.extern_c {
            params.push(format!("{}: *mut {}", p.name, scalar_to_rust(ptr_elem(p))));
            param_names.push(p.name.clone());
        } else if is_ptr_param(p) {
            let elem = ptr_elem(p);
            let rust_ty = scalar_to_rust(elem);
            let is_mut = true; // conservative: treat all ptrs as mut
            let ptr_ty = if is_mut {
                format!("*mut {}", rust_ty)
            } else {
                format!("*const {}", rust_ty)
            };
            params.push(format!("{}_alloc: {}", p.name, ptr_ty));
            params.push(format!("{}_align: {}", p.name, ptr_ty));
            params.push(format!("{}_offset: i64", p.name));
            params.push(format!("{}_size: i64", p.name));
            params.push(format!("{}_stride: i64", p.name));
            param_names.push(p.name.clone());
        } else {
            // Vector param — for now, pass as the LLVM vector type
            // On ARM64, small vectors are passed in NEON registers
            let rust_ty = vector_to_rust_type(&p.ty);
            params.push(format!("{}: {}", p.name, rust_ty));
            param_names.push(p.name.clone());
        }
    }

    (params.join(", "), param_names)
}

/// Map a simd-lang type to Rust type for extern "C" ABI.
fn vector_to_rust_type(ty: &ast::Type) -> String {
    let width = match &ty.width {
        Some(ast::Width::Fixed(n)) => *n,
        _ => 1,
    };
    if width == 1 {
        // Width-1 values are plain scalars in the ABI
        scalar_to_rust(&ty.name).to_string()
    } else {
        // SIMD vectors — use fixed-size arrays for FFI
        // LLVM passes these in NEON registers
        let elem = scalar_to_rust(&ty.name);
        let total_bytes = width as usize * match ty.name.as_str() {
            "u8" | "i8" | "bool" => 1,
            "u16" | "i16" => 2,
            "u32" | "i32" | "f32" => 4,
            "u64" | "i64" | "f64" => 8,
            _ => 1,
        };
        // Use u128 for 16-byte vectors (common case)
        if total_bytes == 16 {
            "u128".to_string()
        } else {
            format!("[{}; {}]", elem, width)
        }
    }
}

/// Build the extern "C" return type.
fn build_extern_return(f: &ast::FnDef) -> String {
    match &f.ret_ty {
        None => String::new(),
        Some(ty) => {
            let rust_ty = vector_to_rust_type(ty);
            format!(" -> {}", rust_ty)
        }
    }
}

/// Generate a safe Rust wrapper function.
/// The parameters that are the length of the slice before them, which a Rust slice carries.
fn slice_lengths(f: &ast::FnDef) -> Vec<String> {
    f.params
        .windows(2)
        .filter(|pair| is_ptr_param(&pair[0]) && pair[1].name == format!("{}__len", pair[0].name))
        .map(|pair| pair[1].name.clone())
        .collect()
}

fn generate_safe_wrapper(out: &mut String, f: &ast::FnDef) {
    // Build safe parameter list
    let mut safe_params = Vec::new();
    let lengths = slice_lengths(f);
    for p in &f.params {
        if lengths.contains(&p.name) {
            continue;
        }
        if is_ptr_param(p) {
            let elem = ptr_elem(p);
            let rust_ty = scalar_to_rust(elem);
            safe_params.push(format!("{}: &mut [{}]", p.name, rust_ty));
        } else {
            let rust_ty = vector_to_rust_type(&p.ty);
            safe_params.push(format!("{}: {}", p.name, rust_ty));
        }
    }

    // Return type
    let safe_ret = match &f.ret_ty {
        None => String::new(),
        Some(ty) => {
            format!(" -> {}", vector_to_rust_type(ty))
        }
    };

    // Doc comment
    out.push_str(&format!("/// `{}` from {}.simd\n", f.name, f.name));
    out.push_str(&format!(
        "pub fn {}({}){} {{\n",
        f.name,
        safe_params.join(", "),
        safe_ret
    ));
    out.push_str("    unsafe {\n");

    // Build call arguments
    let mut call_args = Vec::new();
    for p in &f.params {
        if let Some(slice) = p.name.strip_suffix("__len").filter(|_| lengths.contains(&p.name)) {
            call_args.push(format!("{}.len() as i64", slice));
        } else if is_ptr_param(p) && f.extern_c {
            call_args.push(format!("{}.as_mut_ptr()", p.name));
        } else if is_ptr_param(p) {
            call_args.push(format!("{}.as_mut_ptr()", p.name));
            call_args.push(format!("{}.as_mut_ptr()", p.name));
            call_args.push("0".to_string());
            call_args.push(format!("{}.len() as i64", p.name));
            call_args.push("1".to_string());
        } else {
            call_args.push(p.name.clone());
        }
    }

    // Call and return
    let call = format!("__simd_{}({})", f.name, call_args.join(", "));
    match &f.ret_ty {
        None => {
            out.push_str(&format!("        {};\n", call));
        }
        Some(_) => {
            out.push_str(&format!("        {}\n", call));
        }
    }

    out.push_str("    }\n");
    out.push_str("}\n\n");
}

#[cfg(test)]
mod binding_tests {
    use super::*;

    #[test]
    fn a_slice_is_one_rust_argument_and_two_c_ones() {
        let items = parser::parse("extern fn f(s: slice[u8], n: i64, d: ptr[u8]) -> i64 { return s.len + n }");
        let bindings = generate_rust_bindings("m", &items);
        assert!(bindings.contains("fn __simd_f(s: *mut u8, s__len: i64, n: i64, d: *mut u8) -> i64"), "{bindings}");
        assert!(bindings.contains("pub fn f(s: &mut [u8], n: i64, d: &mut [u8]) -> i64"), "{bindings}");
        assert!(bindings.contains("__simd_f(s.as_mut_ptr(), s.len() as i64, n, d.as_mut_ptr())"), "{bindings}");
    }

    #[test]
    fn a_pointer_and_an_unrelated_integer_stay_two_rust_arguments() {
        let items = parser::parse("extern fn f(s: ptr[u8], n: i64) -> i64 { return n }");
        let bindings = generate_rust_bindings("m", &items);
        assert!(bindings.contains("pub fn f(s: &mut [u8], n: i64) -> i64"), "{bindings}");
    }
}
