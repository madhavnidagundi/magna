// =============================================================================
// Smoke Tests — NVIDIA Backend
// =============================================================================
//! Minimal smoke tests to validate NVIDIA backend on actual hardware.
//!
//! These tests verify:
//! 1. NVIDIA backend can be instantiated
//! 2. TensorRT libraries are accessible
//! 3. Basic middleware initialization works
//! 4. Binary executes without crashes
//!
//! NOT tested (deferred to full test suite):
//! - Actual inference accuracy
//! - Performance benchmarks
//! - Model loading from specific files

use magna_middleware::api::public_api::Middleware;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::init_test_logger;

#[cfg(feature = "nvidia")]
#[test]
fn smoke_test_nvidia_backend_instantiation() {
    init_test_logger();

    println!("🔍 Smoke test: NVIDIA backend instantiation");

    // Test 1: Can we create middleware instance?
    let mw = Middleware::new();
    println!("  ✅ Middleware instance created");

    // Test 3: Can we initialize with NVIDIA backend config?
    let config = MiddlewareConfig {
        fallback_precision: Precision::FP16,
        warmup_runs: 0,
        debug: true,
        labels_path: None,
        model_path: None,
        grpc_address: String::new(),
    };

    match mw.initialize(config) {
        Ok(_) => {
            println!("  ✅ Middleware initialized with NVIDIA backend");
        }
        Err(e) => {
            // It's OK if initialization fails without a model - we're just checking it doesn't crash
            println!(
                "  ⚠️  Middleware initialization returned error (expected without model): {}",
                e
            );
            println!("  ✅ No crash occurred - smoke test passed");
        }
    }

    println!("🎉 NVIDIA backend smoke test completed successfully");
}

#[cfg(not(feature = "nvidia"))]
#[test]
fn smoke_test_nvidia_feature_not_enabled() {
    println!("⚠️  NVIDIA feature not enabled - smoke tests skipped");
    println!("   Run with: cargo test --features nvidia");
}

#[test]
fn smoke_test_basic_binary_execution() {
    init_test_logger();

    println!("🔍 Smoke test: Basic binary execution");

    // Test that basic middleware can be created regardless of backend
    let mw = Middleware::new();
    println!("  ✅ Middleware created");

    // Test with CPU backend (should always work)
    let config = MiddlewareConfig {
        fallback_precision: Precision::FP32,
        warmup_runs: 0,
        debug: true,
        labels_path: None,
        model_path: None,
        grpc_address: String::new(),
    };

    match mw.initialize(config) {
        Ok(_) => println!("  ✅ CPU backend initialized"),
        Err(e) => println!("  ⚠️  CPU backend initialization: {}", e),
    }

    println!("🎉 Basic binary execution smoke test passed");
}

#[cfg(feature = "nvidia")]
#[test]
fn smoke_test_assets_available() {
    init_test_logger();

    println!("🔍 Smoke test: Verify production assets are accessible");

    let model_path = if let Ok(p) = std::env::var("TEST_ENGINE_PATH") {
        std::path::PathBuf::from(p)
    } else {
        std::path::PathBuf::from("../models/mobilenetv4_nvidia_optimized_int8.engine")
    };
    let image_path = std::path::Path::new("../assets/sample.jpg");
    let labels_path = std::path::Path::new("../assets/imagenet_class_index.json");

    // Verify model exists
    if !model_path.exists() {
        eprintln!(
            "SKIPPED [smoke_test_assets_available]: engine not found at {:?}",
            model_path
        );
        return;
    }
    println!("  ✅ Model file exists: {:?}", model_path);

    // Verify image exists
    assert!(
        image_path.exists(),
        "Sample image not found: {:?}",
        image_path
    );
    println!("  ✅ Sample image exists: {:?}", image_path);

    // Verify labels exist
    assert!(labels_path.exists(), "Labels not found: {:?}", labels_path);
    println!("  ✅ Labels file exists: {:?}", labels_path);

    // Check model file is readable and non-empty
    let metadata = std::fs::metadata(&model_path).expect("Cannot read model file metadata");
    assert!(metadata.len() > 0, "Model file is empty");
    println!("  ✅ Model file size: {} MB", metadata.len() / 1_000_000);

    // Check image file is readable
    let img_metadata = std::fs::metadata(image_path).expect("Cannot read image file metadata");
    assert!(img_metadata.len() > 0, "Image file is empty");
    println!("  ✅ Image file size: {} KB", img_metadata.len() / 1_000);

    println!("🎉 Assets verification smoke test passed");
}

#[cfg(feature = "nvidia")]
#[test]
fn smoke_test_full_inference() {
    init_test_logger();

    println!("🔍 Smoke test: Full NVIDIA inference pipeline");
    println!("⚠️  IMPORTANT: TensorRT engines are NOT portable across versions!");
    println!("   This test requires rebuilding the engine on the target hardware.");
    println!("   To rebuild: Use TensorRT's trtexec or model conversion scripts.");
    println!();

    // Asset paths
    let model_path = if let Ok(p) = std::env::var("TEST_ENGINE_PATH") {
        p
    } else {
        "../models/mobilenetv4_nvidia_optimized_int8.engine".to_string()
    };
    let image_path = "../assets/sample.jpg";

    // Verify assets exist
    if !std::path::Path::new(&model_path).exists() {
        eprintln!(
            "SKIPPED [smoke_test_full_inference]: engine not found at {}",
            model_path
        );
        return;
    }
    assert!(
        std::path::Path::new(image_path).exists(),
        "Image not found: {}",
        image_path
    );
    println!("  ✅ Test assets verified");

    // Step 1: Create and initialize middleware
    let mw = Middleware::new();
    let config = MiddlewareConfig {
        fallback_precision: Precision::FP16, // INT8 model, but API uses FP16
        warmup_runs: 0,
        debug: true,
        labels_path: Some("../assets/imagenet_class_index.json".into()),
        model_path: None,
        grpc_address: String::new(),
    };

    mw.initialize(config)
        .expect("Failed to initialize middleware");
    println!("  ✅ Middleware initialized with NVIDIA backend");

    // Step 2: Load TensorRT engine
    let info = mw
        .load_engine(&model_path)
        .expect("Failed to load TensorRT engine");
    println!("  ✅ TensorRT engine loaded");

    // Step 3: Preprocess image using the engine's input precision
    let precision = info.inputs[0].precision;
    use magna_middleware::preprocess::imagenet;
    let mut input_tensor =
        imagenet::preprocess_image_file(image_path, precision).expect("Failed to preprocess image");
    input_tensor.name = info.inputs[0].name.clone();
    println!("  ✅ Image preprocessed: shape {:?}", input_tensor.shape);

    // Step 4: Run inference
    let output = mw.infer(&[input_tensor]).expect("Inference failed");
    println!("  ✅ Inference completed");
    println!("     Output shape: {:?}", output[0].shape);
    println!("     Output precision: {:?}", output[0].precision);

    // Step 5: Basic output validation
    assert!(!output.is_empty(), "Output is empty");
    assert!(
        output[0].shape.iter().product::<usize>() > 0,
        "Output has no elements"
    );
    println!("  ✅ Output validated (non-empty)");

    // Step 6: Shutdown
    mw.shutdown().expect("Shutdown failed");
    println!("  ✅ Middleware shutdown cleanly");

    println!("🎉 Full NVIDIA inference smoke test PASSED!");
    println!("   This confirms:");
    println!("   • TensorRT runtime works on target hardware");
    println!("   • Model loading succeeds");
    println!("   • Preprocessing pipeline works");
    println!("   • Inference executes without crashes");
    println!("   • Binary can access GPU and TensorRT libraries");
}

// =============================================================================
// Smoke Tests — Qualcomm QNN Backend (Radxa Q6A)
// =============================================================================
// See issue #54 ("Qualcomm QNN: Complete FFI execution path and address
// open issues"). Two tiers:
//
// 1. smoke_test_qualcomm_native_headers_available — hardware-independent.
//    Fails the test (not just a warning) if the `qualcomm` feature was
//    requested but the QAIRT SDK headers were not found at compile time
//    (`have_qnn_headers` cfg not set). Before this test existed, a runner
//    without the SDK produced a "green" workflow while silently compiling
//    only the Rust-side fallback error path — this closes that gap without
//    needing any CI YAML changes to enforce it.
// 2. smoke_test_qualcomm_native_load_infer_release /
//    smoke_test_qualcomm_native_repeated_load_release /
//    smoke_test_qualcomm_native_failed_reload_preserves_engine — real hardware
//    only. Set `TEST_QNN_CONTEXT_PATH` to a real, pre-generated `.bin`
//    context binary (see QUALCOMM.md §5) to exercise load → infer → release
//    against the actual Hexagon HTP DSP. With native headers enabled, missing
//    assets fail rather than silently skip. Run with --test-threads=1 because
//    these tests share the board's DSP session.

#[cfg(feature = "qualcomm")]
#[test]
fn smoke_test_qualcomm_native_headers_available() {
    #[cfg(have_qnn_headers)]
    {
        println!("✅ QNN native headers were available at compile time (have_qnn_headers set)");
    }
    #[cfg(not(have_qnn_headers))]
    {
        panic!(
            "The `qualcomm` feature was enabled but QNN SDK headers were not found at \
             build time (have_qnn_headers cfg not set) — native QNN execution is \
             unavailable, and this build only exercises the Rust-side fallback error \
             path. Set QAIRT_SDK_ROOT or QNN_SDK_ROOT to a QAIRT SDK whose \
             include/QNN directory exists before building (see QUALCOMM.md)."
        );
    }
}

#[cfg(not(feature = "qualcomm"))]
#[test]
fn smoke_test_qualcomm_feature_not_enabled() {
    println!("⚠️  Qualcomm feature not enabled - smoke tests skipped");
    println!("   Run with: cargo test --features qualcomm");
}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
#[test]
fn smoke_test_qualcomm_native_load_infer_release() {
    use magna_middleware::backends::qualcomm::adapter::QualcommAdapter;
    use magna_middleware::inference::traits::{InferenceBackend, TensorBuffer};

    init_test_logger();

    let context_path = std::env::var("TEST_QNN_CONTEXT_PATH").unwrap_or_else(|_| {
        panic!(
            "TEST_QNN_CONTEXT_PATH is required for this hardware test; \
            set it to a real generated_ctx/*.bin on the Radxa board \
            (see QUALCOMM.md §5)"
        )
    });

    println!("🔍 Smoke test: native QNN load → infer → release against real hardware");

    let adapter = QualcommAdapter::new();

    let info = adapter
        .load_engine(&context_path)
        .expect("Failed to load real QNN context binary");
    println!(
        "  ✅ Context binary loaded: {} input(s), {} output(s)",
        info.inputs.len(),
        info.outputs.len()
    );
    assert_eq!(
        info.inputs.len(),
        1,
        "expected exactly one input tensor (current wrapper scope — see QualcommAdapter docs)"
    );
    assert_eq!(
        info.outputs.len(),
        1,
        "expected exactly one output tensor (current wrapper scope — see QualcommAdapter docs)"
    );

    adapter.allocate_buffers().expect("allocate_buffers failed");
    assert!(adapter.is_ready());

    let input_spec = &info.inputs[0];
    let input_bytes = vec![
        0u8;
        input_spec
            .checked_byte_size()
            .expect("input spec byte size must be valid")
    ];

    // Repeated infer cycles on the same loaded context to catch leaks/crashes
    // across calls, per issue #54's "repeated load/infer/release" criterion.
    for i in 0..3 {
        let input = TensorBuffer {
            name: input_spec.name.clone(),
            data: input_bytes.clone(),
            shape: input_spec.shape.clone(),
            precision: input_spec.precision,
        };
        let output = adapter
            .infer(&[input])
            .unwrap_or_else(|e| panic!("infer() failed on iteration {i}: {e}"));
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].shape, info.outputs[0].shape);
        assert_eq!(
            output[0].data.len(),
            output[0]
                .checked_byte_size()
                .expect("output byte size must be valid"),
            "output byte count must match shape × element size"
        );
        println!(
            "  ✅ Inference {} completed, output shape {:?}",
            i + 1,
            output[0].shape
        );
    }

    adapter.release().expect("release failed");
    assert!(!adapter.is_ready());
    assert!(adapter.engine_info().is_none());
    println!("🎉 Native QNN load/infer/release smoke test PASSED on real hardware");
}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
#[test]
fn smoke_test_qualcomm_native_repeated_load_release() {
    use magna_middleware::backends::qualcomm::adapter::QualcommAdapter;
    use magna_middleware::inference::traits::InferenceBackend;

    let context_path = std::env::var("TEST_QNN_CONTEXT_PATH").unwrap_or_else(|_| {
        panic!(
            "TEST_QNN_CONTEXT_PATH is required for this hardware test; \
            set it to a real generated_ctx/*.bin on the Radxa board \
            (see QUALCOMM.md §5)"
        )
    });

    // Three independent full lifecycles on the same adapter instance exercise
    // normal teardown/reinitialization. Failed replacement without an intervening
    // release is covered separately below.
    let adapter = QualcommAdapter::new();
    for cycle in 0..3 {
        adapter
            .load_engine(&context_path)
            .unwrap_or_else(|e| panic!("load_engine failed on cycle {cycle}: {e}"));
        adapter.allocate_buffers().expect("allocate_buffers failed");
        adapter
            .release()
            .unwrap_or_else(|e| panic!("release failed on cycle {cycle}: {e}"));
        assert!(!adapter.is_ready());
    }
    println!("🎉 Repeated load/release cycles completed without error");
}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
#[test]
fn smoke_test_qualcomm_native_failed_reload_preserves_engine() {
    use magna_middleware::backends::qualcomm::adapter::QualcommAdapter;
    use magna_middleware::inference::traits::{InferenceBackend, TensorBuffer};
    use std::io::Write;

    let context_path = std::env::var("TEST_QNN_CONTEXT_PATH")
        .expect("set TEST_QNN_CONTEXT_PATH to a real QNN context binary on the Radxa board");
    let adapter = QualcommAdapter::new();
    let original = adapter
        .load_engine(&context_path)
        .expect("load original engine");
    adapter
        .allocate_buffers()
        .expect("allocate original buffers");
    assert!(adapter.is_ready());

    let input_spec = &original.inputs[0];
    let input = TensorBuffer {
        name: input_spec.name.clone(),
        data: vec![0; input_spec.checked_byte_size().unwrap()],
        shape: input_spec.shape.clone(),
        precision: input_spec.precision,
    };
    let before = adapter
        .infer(std::slice::from_ref(&input))
        .expect("initial inference");
    assert_eq!(before.len(), 1);

    // The replacement exists, so this cannot pass only through the missing-path
    // check. It must fail native initialization/loading without consuming the old
    // handle. Repeating the attempt also exercises cleanup of failed candidates.
    let mut invalid = tempfile::Builder::new().suffix(".bin").tempfile().unwrap();
    invalid
        .write_all(b"INVALID_QNN_CONTEXT_FOR_RELOAD_TEST")
        .unwrap();
    invalid.flush().unwrap();
    for _ in 0..3 {
        assert!(adapter
            .load_engine(invalid.path().to_str().unwrap())
            .is_err());
        assert!(adapter.is_ready(), "failed reload must preserve readiness");
        let retained = adapter
            .engine_info()
            .expect("old metadata must remain available");
        assert_eq!(retained.name, original.name);
        assert_eq!(retained.inputs, original.inputs);
        assert_eq!(retained.outputs, original.outputs);
        assert_eq!(retained.memory_bytes, original.memory_bytes);

        // Do not reallocate: failed replacement must preserve the ready engine.
        let after = adapter
            .infer(std::slice::from_ref(&input))
            .expect("original engine must still execute after failed replacement");
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].name, original.outputs[0].name);
        assert_eq!(after[0].shape, original.outputs[0].shape);
        assert_eq!(after[0].precision, original.outputs[0].precision);
        assert_eq!(
            after[0].data.len(),
            original.outputs[0].checked_byte_size().unwrap()
        );
    }

    adapter.release().unwrap();
    assert!(!adapter.is_ready());
    assert!(adapter.engine_info().is_none());
}
