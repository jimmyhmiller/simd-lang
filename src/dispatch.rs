//! Builds one program for several x86 instruction sets in one library. Each level's exported
//! functions get a suffix, and a small generated C file provides the plain names. The first
//! call checks the CPU and stores the best level in a pointer. Every call after that is a
//! single indirect jump, with no check.

use crate::ast::*;
use crate::codegen::X86Isa;
use crate::comptime::{for_each_expr, walk_expr};
use std::collections::HashSet;

pub fn isa_suffix(isa: X86Isa) -> &'static str {
    match isa {
        X86Isa::Sse42 => "sse42",
        X86Isa::Avx2 => "avx2",
        X86Isa::Avx512 => "avx512",
    }
}

fn exported(items: &[Item]) -> Vec<&FnDef> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(def) if !def.private && def.comptime_params.is_empty() => Some(def),
            _ => None,
        })
        .collect()
}

/// The same program, with every exported function renamed for this level. Other functions
/// stay internal to each level's object, so their names can't collide.
pub fn with_isa_suffix(items: &[Item], isa: X86Isa) -> Vec<Item> {
    let names: HashSet<String> = exported(items).iter().map(|def| def.name.clone()).collect();
    let suffix = isa_suffix(isa);
    let renamed = |name: &str| format!("{}__{}", name, suffix);
    items
        .iter()
        .map(|item| {
            let Item::Fn(def) = item else { return item.clone() };
            let mut def = def.clone();
            if names.contains(&def.name) {
                def.name = renamed(&def.name);
            }
            for_each_expr(&mut def.body, &mut |top| {
                walk_expr(top, &mut |expr| {
                    if let Expr::Call { func, .. } = expr {
                        match func.as_mut() {
                            Expr::Ident(name) | Expr::Instantiate { name, .. } if names.contains(name.as_str()) => {
                                *name = renamed(name);
                            }
                            _ => {}
                        }
                    }
                });
            });
            Item::Fn(def)
        })
        .collect()
}

fn c_type(ty: &Type, fn_name: &str) -> &'static str {
    if ty.name == "ptr" {
        return "void *";
    }
    if ty.width != Some(Width::Fixed(1)) {
        panic!("--dispatch only handles scalars and pointers, but '{}' uses {}", fn_name, ty.name);
    }
    match ty.name.as_str() {
        "i8" => "signed char",
        "u8" => "unsigned char",
        "i16" => "short",
        "u16" => "unsigned short",
        "i32" => "int",
        "u32" => "unsigned int",
        "i64" => "long long",
        "u64" => "unsigned long long",
        "f32" => "float",
        "f64" => "double",
        "bool" => "_Bool",
        other => panic!("--dispatch only handles scalars and pointers, but '{}' uses {}", fn_name, other),
    }
}

/// C for the plain names, and `<stem>_isa()`, which says which level this CPU got.
pub fn dispatcher_source(stem: &str, items: &[Item], isas: &[X86Isa]) -> String {
    let mut ordered: Vec<X86Isa> = isas.to_vec();
    ordered.sort_by_key(|isa| std::cmp::Reverse(isa_rank(*isa)));
    ordered.dedup();

    let mut out = String::from(DETECTION);
    out.push_str("static int chosen_level(void) {\n    int level = cpu_level();\n");
    for isa in &ordered {
        out.push_str(&format!("    if (level >= {}) return {};\n", isa_rank(*isa), isa_rank(*isa)));
    }
    // Nothing here can run. A trap is the honest answer, since every level needs these instructions.
    out.push_str("    __builtin_trap();\n}\n\n");

    out.push_str(&format!("const char *{}_isa(void) {{\n    switch (chosen_level()) {{\n", stem));
    for isa in &ordered {
        out.push_str(&format!("    case {}: return \"{}\";\n", isa_rank(*isa), isa_suffix(*isa)));
    }
    out.push_str("    }\n    __builtin_trap();\n}\n\n");

    for def in exported(items) {
        if !def.extern_c {
            panic!(
                "--dispatch only handles `extern fn`, since other functions take pointers as descriptors: '{}'",
                def.name
            );
        }
        let name = &def.name;
        let ret = def.ret_ty.as_ref().map_or("void", |ty| c_type(ty, name));
        let params: Vec<String> =
            def.params.iter().map(|p| format!("{} {}", c_type(&p.ty, name), p.name)).collect();
        let params = if params.is_empty() { "void".to_string() } else { params.join(", ") };
        let args: Vec<&str> = def.params.iter().map(|p| p.name.as_str()).collect();
        let args = args.join(", ");
        let do_return = if def.ret_ty.is_some() { "return " } else { "" };

        for isa in &ordered {
            out.push_str(&format!("{} {}__{}({});\n", ret, name, isa_suffix(*isa), params));
        }
        out.push_str(&format!("static {} {}__first_call({});\n", ret, name, params));
        out.push_str(&format!("static {} (*{}__chosen)({}) = {}__first_call;\n", ret, name, params, name));
        out.push_str(&format!("static {} {}__first_call({}) {{\n", ret, name, params));
        out.push_str("    switch (chosen_level()) {\n");
        for isa in &ordered {
            out.push_str(&format!(
                "    case {}: __atomic_store_n(&{}__chosen, {}__{}, __ATOMIC_RELAXED); break;\n",
                isa_rank(*isa),
                name,
                name,
                isa_suffix(*isa)
            ));
        }
        out.push_str("    }\n");
        out.push_str(&format!(
            "    {}__atomic_load_n(&{}__chosen, __ATOMIC_RELAXED)({});\n}}\n",
            do_return, name, args
        ));
        out.push_str(&format!(
            "{} {}({}) {{\n    {}__atomic_load_n(&{}__chosen, __ATOMIC_RELAXED)({});\n}}\n\n",
            ret, name, params, do_return, name, args
        ));
    }
    out
}

fn isa_rank(isa: X86Isa) -> u8 {
    match isa {
        X86Isa::Sse42 => 1,
        X86Isa::Avx2 => 2,
        X86Isa::Avx512 => 3,
    }
}

/// The highest level this CPU and OS can run: 0 for none, then the ranks above. The feature
/// lists are the x86-64-v2, v3 and v4 definitions, plus PCLMUL, which every level promises.
const DETECTION: &str = r#"// Generated by simd-lang --dispatch. Do not edit.
#include <cpuid.h>

static unsigned long long read_xcr0(void) {
    unsigned int low, high;
    __asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return ((unsigned long long)high << 32) | low;
}

static int cpu_level(void) {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return 0;
    unsigned int leaf1_ecx = ecx, leaf1_edx = edx;

    int ext_ecx = 0;
    if (__get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx)) ext_ecx = ecx;

    /* v2: CMPXCHG16B, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, then PCLMUL. And LAHF/SAHF. */
    unsigned int v2 = bit_CMPXCHG16B | bit_SSE3 | bit_SSSE3 | bit_SSE41 | bit_SSE42 | bit_POPCNT | bit_PCLMUL;
    if ((leaf1_ecx & v2) != v2 || !(ext_ecx & bit_LAHF_LM)) return 0;

    /* The OS has to save the wider registers, or using them faults. */
    if (!(leaf1_ecx & bit_OSXSAVE)) return 1;
    unsigned long long xcr0 = read_xcr0();
    if ((xcr0 & 0x6) != 0x6) return 1;

    if (__get_cpuid_max(0, 0) < 7) return 1;
    unsigned int leaf7_ebx, leaf7_ecx, leaf7_edx;
    __cpuid_count(7, 0, eax, leaf7_ebx, leaf7_ecx, leaf7_edx);

    /* v3: AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OSXSAVE. */
    unsigned int v3_leaf1 = bit_AVX | bit_F16C | bit_FMA | bit_MOVBE;
    unsigned int v3_leaf7 = bit_AVX2 | bit_BMI | bit_BMI2;
    if ((leaf1_ecx & v3_leaf1) != v3_leaf1 || (leaf7_ebx & v3_leaf7) != v3_leaf7 || !(ext_ecx & bit_LZCNT)) return 1;

    /* v4: AVX512F, BW, CD, DQ, VL, and the OS saving the mask and ZMM registers. */
    unsigned int v4_leaf7 = bit_AVX512F | bit_AVX512BW | bit_AVX512CD | bit_AVX512DQ | bit_AVX512VL;
    if ((leaf7_ebx & v4_leaf7) != v4_leaf7 || (xcr0 & 0xe0) != 0xe0) return 2;

    return 3;
}

"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    const PROGRAM: &str = r#"
        private fn helper(a: i64) -> i64 { return a + 1 }
        extern fn inner(a: i64) -> i64 { return helper(a) }
        extern fn outer(p: ptr[u8], a: i64) -> i64 { return inner(a) }
    "#;

    fn fn_names(items: &[Item]) -> Vec<String> {
        items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(def) => Some(def.name.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn exported_functions_and_calls_to_them_get_the_suffix() {
        let items = with_isa_suffix(&parser::parse(PROGRAM), X86Isa::Avx512);
        assert_eq!(fn_names(&items), ["helper", "inner__avx512", "outer__avx512"]);
        let Item::Fn(outer) = &items[2] else { panic!("expected a function") };
        assert!(format!("{:?}", outer.body).contains("inner__avx512"), "{:?}", outer.body);
        let Item::Fn(inner) = &items[1] else { panic!("expected a function") };
        assert!(format!("{:?}", inner.body).contains("\"helper\""), "private calls keep their name");
    }

    #[test]
    fn the_dispatcher_only_names_the_levels_it_was_given_best_first() {
        let source = dispatcher_source("lib", &parser::parse(PROGRAM), &[X86Isa::Sse42, X86Isa::Avx512]);
        assert!(source.contains("long long inner(long long a)"));
        assert!(source.contains("long long outer(void * p, long long a)"));
        assert!(source.contains("inner__avx512") && source.contains("inner__sse42"));
        assert!(!source.contains("inner__avx2"));
        assert!(source.contains("lib_isa"));
        let best = source.find("if (level >= 3) return 3").expect("avx512 is checked");
        let next = source.find("if (level >= 1) return 1").expect("sse4.2 is checked");
        assert!(best < next, "the best level comes first");
    }

    #[test]
    fn the_dispatcher_passes_a_slice_as_a_pointer_and_length() {
        let source = dispatcher_source("lib", &parser::parse("extern fn f(s: slice[u8], n: i64) -> i64 { return s.len + n }"), &[X86Isa::Avx2]);
        assert!(source.contains("long long f(void * s, long long s__len, long long n)"), "{source}");
    }

    #[test]
    #[should_panic(expected = "only handles `extern fn`")]
    fn the_dispatcher_rejects_functions_with_descriptor_pointers() {
        dispatcher_source("lib", &parser::parse("fn f(p: ptr[u8]) -> i64 { return 0 }"), &[X86Isa::Avx2]);
    }

    #[test]
    #[should_panic(expected = "only handles scalars and pointers")]
    fn the_dispatcher_rejects_vector_arguments() {
        dispatcher_source("lib", &parser::parse("extern fn f(a: u8[16]) -> i64 { return 0 }"), &[X86Isa::Avx2]);
    }
}
