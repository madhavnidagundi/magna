#[cfg(feature = "ti")]
fn has_tidl_ort_symbols(lib_dir: &str) -> bool {
    let lib = std::path::Path::new(lib_dir).join("libonnxruntime.so");

    let output = match std::process::Command::new("nm")
        .args(["-D", lib.to_string_lossy().as_ref()])
        .output()
    {
        Ok(output) => output,
        Err(e) => {
            println!("cargo:warning=Could not run nm to verify TIDL symbols: {e}");
            return false;
        }
    };

    if !output.status.success() {
        println!(
            "cargo:warning=Could not inspect ONNX Runtime library for TIDL symbols: {}",
            lib.display()
        );
        return false;
    }

    let symbols = String::from_utf8_lossy(&output.stdout);

    let has_default = symbols
        .lines()
        .any(|line| line.contains("OrtSessionsOptionsSetDefault_Tidl"));
    let has_append = symbols
        .lines()
        .any(|line| line.contains("OrtSessionOptionsAppendExecutionProvider_Tidl"));

    if !has_default || !has_append {
        println!(
            "cargo:warning=ONNX Runtime found, but required TIDL Execution Provider symbols are missing"
        );
    }

    has_default && has_append
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/backends/nvidia/trt_c_api.cpp");
    println!("cargo:rerun-if-changed=src/backends/qualcomm/qnn_c_api.cpp");
    println!("cargo:rerun-if-changed=proto/magna.proto");
    println!("cargo::rustc-check-cfg=cfg(have_qnn_headers)");
    println!("cargo::rustc-check-cfg=cfg(have_cuda)");
    println!("cargo::rustc-check-cfg=cfg(have_dlr)");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
    println!("cargo:rerun-if-changed=cpu_ort_c_api.c");
    println!("cargo:rerun-if-env-changed=ORT_CPU_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=ORT_CPU_LIB_DIR");
    println!("cargo:rerun-if-env-changed=QNN_SDK_ROOT");
    println!("cargo:rerun-if-env-changed=QAIRT_SDK_ROOT");
    println!("cargo:rerun-if-env-changed=MAGNA_DOCS_BUILD");

    // -------------------------------------------------------------------------
    // CPU backend — direct ONNX Runtime C API wrapper (bypasses the `ort` crate,
    // whose Rust session-builder path deadlocks during CreateSession on this
    // board's aarch64/Yocto onnxruntime 1.15.0 build). Only compiled with the
    // `cpu` feature.
    // -------------------------------------------------------------------------
    #[cfg(feature = "cpu-system-ort")]
    {
        let ort_include = std::env::var("ORT_CPU_INCLUDE_DIR").ok().or_else(|| {
            ["/usr/include", "/usr/local/include"]
                .into_iter()
                .find(|p| {
                    std::path::Path::new(&format!(
                        "{p}/onnxruntime/core/session/onnxruntime_c_api.h"
                    ))
                    .exists()
                })
                .map(String::from)
        });

        let ort_lib_dir = std::env::var("ORT_CPU_LIB_DIR").ok().or_else(|| {
            ["/usr/lib", "/usr/local/lib"]
                .into_iter()
                .find(|p| std::path::Path::new(&format!("{p}/libonnxruntime.so")).exists())
                .map(String::from)
        });

        match (ort_include, ort_lib_dir) {
            (Some(inc), Some(lib_dir)) => {
                println!("cargo:warning=CPU backend: ONNX Runtime headers at: {inc}");
                println!("cargo:warning=CPU backend: ONNX Runtime library at: {lib_dir}");

                cc::Build::new()
                    .file("cpu_ort_c_api.c")
                    .include(&inc)
                    .flag("-O2")
                    .compile("cpu_ort_c_api");

                println!("cargo:rustc-link-search=native={lib_dir}");
                println!("cargo:rustc-link-lib=onnxruntime");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            }
            _ => {
                if std::env::var_os("DOCS_RS").is_some()
                    || std::env::var_os("MAGNA_DOCS_BUILD").is_some()
                {
                    println!(
                        "cargo:warning=Skipping system ONNX Runtime wrapper for documentation build"
                    );
                    return build_protobuf();
                }
                return Err(
                    "cpu-system-ort requires ONNX Runtime headers and libonnxruntime; set \
                     ORT_CPU_INCLUDE_DIR and ORT_CPU_LIB_DIR"
                        .into(),
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // NVIDIA TensorRT C++ wrapper — only compiled with the `nvidia` feature
    // and only on Linux where CUDA is available.
    // -------------------------------------------------------------------------
    #[cfg(feature = "nvidia")]
    {
        fn find_cuda_installation() -> Option<String> {
            let mut paths = Vec::new();
            let check_dirs = ["/usr/local", "/opt"];
            for dir in check_dirs {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        if let Ok(file_type) = entry.file_type() {
                            if file_type.is_dir() {
                                let file_name = entry.file_name();
                                let name = file_name.to_string_lossy();
                                if name.starts_with("cuda") {
                                    paths.push(entry.path());
                                }
                            }
                        }
                    }
                }
            }
            paths.sort();
            paths.pop().map(|p| p.to_string_lossy().to_string())
        }

        fn validate_path(path: &str, name: &str) -> bool {
            if !std::path::Path::new(path).exists() {
                println!(
                    "cargo:warning=Missing dependency: {} not found at {}",
                    name, path
                );
                return false;
            }
            true
        }

        // 1. Try environment variables first
        let cuda_root = std::env::var("CUDA_ROOT")
            .or_else(|_| std::env::var("CUDA_PATH"))
            .unwrap_or_else(|_| {
                // 2. Auto-detect from common paths
                find_cuda_installation().unwrap_or_else(|| {
                    println!("cargo:warning=CUDA not found. Set CUDA_ROOT environment variable.");
                    "/usr/local/cuda".to_string() // fallback
                })
            });

        let cuda_include = format!("{}/include", cuda_root);
        let cuda_lib = format!("{}/lib64", cuda_root);

        // 3. Detect architecture for TensorRT system paths
        let arch = if cfg!(target_arch = "aarch64") {
            "aarch64-linux-gnu"
        } else {
            "x86_64-linux-gnu"
        };

        let tensorrt_include =
            std::env::var("TENSORRT_INCLUDE").unwrap_or_else(|_| format!("/usr/include/{}", arch));

        let tensorrt_lib =
            std::env::var("TENSORRT_LIB").unwrap_or_else(|_| format!("/usr/lib/{}", arch));

        // 4. Validate paths exist
        let cuda_ok = validate_path(&cuda_include, "CUDA headers");
        let trt_ok = validate_path(&tensorrt_include, "TensorRT headers");

        if cuda_ok && trt_ok {
            println!("cargo:rustc-cfg=have_cuda");
            println!("[build.rs] nvidia feature enabled — compiling TRT C++ wrapper");
            cc::Build::new()
                .cpp(true)
                .flag("-Wno-deprecated-declarations")
                .flag("-Wno-unused-parameter")
                .file("src/backends/nvidia/trt_c_api.cpp")
                .include(&cuda_include)
                .include(&tensorrt_include)
                .compile("trt_wrapper");

            println!("cargo:rustc-link-search=native={}", cuda_lib);
            println!("cargo:rustc-link-search=native={}", tensorrt_lib);
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=nvinfer");
            println!("cargo:rustc-link-lib=nvonnxparser");
            // rpath-link allows the linker to resolve transitive .so dependencies
            // (e.g., libnvonnxparser.so depends on libnvinfer_plugin.so)
            println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", tensorrt_lib);
            println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", cuda_lib);
        } else {
            println!("cargo:warning=Skipping TensorRT wrapper compilation because CUDA/TensorRT headers were not found");
        }
    }

    #[cfg(not(feature = "nvidia"))]
    {
        println!("[build.rs] nvidia feature NOT enabled — skipping TRT C++ compilation");
    }

    // -------------------------------------------------------------------------
    // Qualcomm QNN backend
    // -------------------------------------------------------------------------
    #[cfg(feature = "qualcomm")]
    {
        println!("[build.rs] qualcomm feature enabled — configuring QNN");
        let qnn_root = std::env::var("QAIRT_SDK_ROOT")
            .or_else(|_| std::env::var("QNN_SDK_ROOT"))
            .unwrap_or_default();
        let qnn_include = format!("{}/include/QNN", qnn_root);

        if std::path::Path::new(&qnn_include).exists() {
            println!("cargo:rustc-cfg=have_qnn_headers");
            cc::Build::new()
                .cpp(true)
                .file("src/backends/qualcomm/qnn_c_api.cpp")
                .include(&qnn_include)
                .compile("qnn_wrapper");

            // Note: we load libQnnHtp.so dynamically in qnn_c_api.cpp via dlopen,
            // so we don't need to link it here at build time.
        } else {
            println!(
                "cargo:warning=QNN SDK headers not found at {}. Native Qualcomm inference will be unavailable.",
                qnn_include
            );
        }
    }

    // -------------------------------------------------------------------------
    // TI Backend (DLR + C shim)
    // -------------------------------------------------------------------------
    #[cfg(feature = "ti")]
    {
        println!("cargo:warning=Building TI backend (DLR)");
        println!("cargo:rerun-if-env-changed=TDA4_SYSROOT");
        println!("cargo:rerun-if-env-changed=DLR_INCLUDE_DIR");
        println!("cargo:rerun-if-env-changed=DLR_LIB_DIR");
        println!("cargo:rerun-if-env-changed=EDGEAI_INCLUDE_DIR");
        println!("cargo:rerun-if-env-changed=MAGNA_TI_FORCE_DLR");
        println!("cargo:rerun-if-changed=dlr_c_api.c");

        // --- Resolve DLR include dir ---
        // Priority: env var > sysroot layout > system paths
        let sysroot = std::env::var("TDA4_SYSROOT").ok();

        let dlr_include = std::env::var("DLR_INCLUDE_DIR").ok().or_else(|| {
            // Sysroot layout (cross-compile)
            if let Some(ref sr) = sysroot {
                let p = format!("{sr}/usr/include");
                if std::path::Path::new(&format!("{p}/dlr.h")).exists() {
                    return Some(p);
                }
            }
            // Native: DLR Python package (TDA4 default install)
            let native = "/usr/lib/python3.12/site-packages/dlr/include";
            if std::path::Path::new(&format!("{native}/dlr.h")).exists() {
                return Some(native.into());
            }
            // Native: standard system include
            if std::path::Path::new("/usr/include/dlr.h").exists() {
                return Some("/usr/include".into());
            }
            None
        });

        // --- Resolve Edge AI SDK include dir ---
        let edgeai_include = std::env::var("EDGEAI_INCLUDE_DIR").ok().or_else(|| {
            if let Some(ref sr) = sysroot {
                let p = format!("{sr}/edgeai/include");
                if std::path::Path::new(&p).exists() {
                    return Some(p);
                }
            }
            // Native: TDA4 default install
            let native = "/opt/edgeai-dl-inferer/dl_inferer/include";
            if std::path::Path::new(native).exists() {
                return Some(native.into());
            }
            None
        });

        // --- Resolve DLR library dir ---
        let dlr_lib = std::env::var("DLR_LIB_DIR").ok().or_else(|| {
            if let Some(ref sr) = sysroot {
                // Check sysroot paths
                for sub in ["edgeai/lib", "usr/lib"] {
                    let p = format!("{sr}/{sub}");
                    if std::path::Path::new(&format!("{p}/libdlr.so")).exists() {
                        return Some(p);
                    }
                }
            }
            // Native: system lib
            if std::path::Path::new("/usr/lib/libdlr.so").exists() {
                return Some("/usr/lib".into());
            }
            None
        });

        // DLR (TVM-compiled) and ONNX Runtime TIDL EP (onnxrt-tidl artifact exports,
        // e.g. `artifacts_folder`/`param.yaml`-style exports) are two different TI
        // compilation toolchains and are not interchangeable at runtime. Since a
        // system may have both SDKs installed, default to the ONNX Runtime TIDL EP
        // path (the more common artifact export format) unless the caller opts into
        // DLR explicitly via MAGNA_TI_FORCE_DLR=1.
        let force_dlr = std::env::var("MAGNA_TI_FORCE_DLR").is_ok();
        if dlr_include.is_some() && !force_dlr {
            println!("cargo:warning=DLR runtime detected but not forced (set MAGNA_TI_FORCE_DLR=1 to use it for TVM-compiled models).");
            println!("cargo:warning=Defaulting to the ONNX Runtime TIDL Execution Provider backend (matches onnxrt-tidl artifact exports).");
        }

        if let Some(ref dlr_inc) = dlr_include.clone().filter(|_| force_dlr) {
            println!("cargo:warning=DLR headers found at: {dlr_inc}");

            let mut build = cc::Build::new();
            build.file("dlr_c_api.c").include(dlr_inc).flag("-O2");

            if let Some(ref eai) = edgeai_include {
                println!("cargo:warning=Edge AI headers at: {eai}");
                build.include(eai);
            }

            build.compile("dlr_c_api");

            if let Some(ref lib_dir) = dlr_lib {
                println!("cargo:warning=DLR lib at: {lib_dir}");
                println!("cargo:rustc-link-search=native={lib_dir}");
            }

            println!("cargo:rustc-link-lib=dlr");
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-cfg=have_dlr");
        } else {
            println!("cargo:warning=DLR not found. Falling back to ONNX Runtime TIDL Execution Provider backend.");
            println!("cargo:rerun-if-env-changed=ORT_TIDL_INCLUDE_DIR");
            println!("cargo:rerun-if-env-changed=ORT_TIDL_LIB_DIR");
            println!("cargo:rerun-if-changed=tidl_onnx_c_api.c");

            // --- Resolve ONNX Runtime (TIDL-enabled) include dir ---
            let ort_include = std::env::var("ORT_TIDL_INCLUDE_DIR").ok().or_else(|| {
                let candidates = [
                    "/usr/include/onnxruntime/include",
                    "/usr/local/include/onnxruntime",
                ];
                candidates
                    .into_iter()
                    .find(|p| {
                        std::path::Path::new(&format!(
                            "{p}/onnxruntime/core/session/onnxruntime_c_api.h"
                        ))
                        .exists()
                    })
                    .map(String::from)
            });

            // --- Resolve ONNX Runtime library dir ---
            let ort_lib_dir = std::env::var("ORT_TIDL_LIB_DIR").ok().or_else(|| {
                ["/usr/lib", "/usr/local/lib"]
                    .into_iter()
                    .find(|p| std::path::Path::new(&format!("{p}/libonnxruntime.so")).exists())
                    .map(String::from)
            });

            match (ort_include, ort_lib_dir) {
                (Some(inc), Some(lib_dir)) if has_tidl_ort_symbols(&lib_dir) => {
                    println!("cargo:warning=TIDL-enabled ONNX Runtime headers at: {inc}");
                    println!("cargo:warning=ONNX Runtime library at: {lib_dir}");

                    cc::Build::new()
                        .file("tidl_onnx_c_api.c")
                        .include(&inc)
                        .flag("-O2")
                        .compile("tidl_onnx_c_api");

                    println!("cargo:rustc-link-search=native={lib_dir}");
                    println!("cargo:rustc-link-lib=onnxruntime");
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
                    println!("cargo:rustc-cfg=have_dlr");
                }
                _ => {
                    println!(
                        "cargo:warning=Neither DLR nor a TIDL-enabled ONNX Runtime were found."
                    );
                    println!("cargo:warning=Set DLR_INCLUDE_DIR/DLR_LIB_DIR for on-board DLR deployment,");
                    println!("cargo:warning=or ORT_TIDL_INCLUDE_DIR/ORT_TIDL_LIB_DIR for host TIDL emulation.");
                }
            }
        }
    }

    build_protobuf()
}

fn build_protobuf() -> Result<(), Box<dyn std::error::Error>> {
    // -------------------------------------------------------------------------
    // gRPC protobuf compilation
    // -------------------------------------------------------------------------
    // tonic-build requires `protoc` on the system PATH.
    // If protoc is not installed, we generate a helpful message and skip.
    let proto_path = "proto/magna.proto";
    if std::path::Path::new(proto_path).exists() {
        match tonic_build::compile_protos(proto_path) {
            Ok(_) => println!("[build.rs] protobuf compiled successfully"),
            Err(e) => {
                println!("cargo:warning=Failed to compile protobuf: {e}");
                println!("cargo:warning=Ensure `protoc` is installed and on your PATH.");
                println!("cargo:warning=On Ubuntu: sudo apt install -y protobuf-compiler");
                println!("cargo:warning=On macOS:  brew install protobuf");
                println!("cargo:warning=On Windows: choco install protoc / winget install protoc");
                // Don't fail the build — the grpc module will use a fallback.
            }
        }
    } else {
        println!("cargo:warning=proto/magna.proto not found — skipping protobuf generation");
    }

    Ok(())
}
