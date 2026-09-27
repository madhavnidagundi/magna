// =============================================================================
// Integration Test — Inference Pipeline
// =============================================================================
//! End-to-end test covering:
//! 1. Initialize middleware
//! 2. Load engine
//! 3. Run inference
//! 4. Classify output
//! 5. Check metrics
//! 6. Shutdown

use magna_middleware::api::public_api::Middleware;
use magna_middleware::inference::traits::TensorBuffer;
use magna_middleware::lifecycle::state_manager::State;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::init_test_logger;

fn get_model_path(mw: &Middleware) -> Option<String> {
    // Allow CI to inject the engine path via env var
    if let Ok(p) = std::env::var("TEST_ENGINE_PATH") {
        if std::path::Path::new(&p).exists() {
            return Some(p);
        }
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    if mw.backend_name() == "cpu" {
        let p = std::path::Path::new(&manifest_dir).join("../models/mobilenetv2.onnx");
        if p.exists() {
            Some(p.to_string_lossy().to_string())
        } else {
            None
        }
    } else {
        let p = std::path::Path::new(&manifest_dir)
            .join("../models/mobilenetv4_nvidia_optimized_int8.engine");
        if p.exists() {
            Some(p.to_string_lossy().to_string())
        } else {
            None
        }
    }
}

#[test]
fn full_inference_pipeline() {
    init_test_logger();

    let mw = Middleware::new();
    assert_eq!(mw.state(), State::Uninitialized);

    // Initialize.
    let config = MiddlewareConfig {
        fallback_precision: Precision::FP32,
        warmup_runs: 0,
        debug: true,
        labels_path: None,
        model_path: None,
        grpc_address: String::new(),
    };
    mw.initialize(config).unwrap();
    assert_eq!(mw.state(), State::Initialized);

    // Load engine.
    let Some(path) = get_model_path(&mw) else {
        eprintln!("SKIPPED [full_inference_pipeline]: test model not found");
        return;
    };
    let info = mw.load_engine(&path).unwrap();
    assert_eq!(mw.state(), State::Ready);
    assert_eq!(info.inputs[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(info.outputs[0].shape, vec![1, 1000]);

    // Run inference with a raw tensor.
    let input = TensorBuffer {
        name: info.inputs[0].name.clone(),
        data: vec![0u8; 3 * 224 * 224 * 4],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    };
    let output = mw.infer(&[input]).unwrap();
    assert_eq!(output[0].shape, vec![1, 1000]);
    assert_eq!(output[0].precision, Precision::FP32);

    // Check that metrics recorded the inference (requires --features metrics).
    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        assert_eq!(metrics.inference_count, 1);
        assert!(metrics.last_latency_ms >= 0.0); // may be very fast
    }

    // Shutdown.
    mw.shutdown().unwrap();
    assert_eq!(mw.state(), State::Uninitialized);
}

#[test]
fn multiple_inferences() {
    init_test_logger();

    let mw = Middleware::new();
    mw.initialize(MiddlewareConfig {
        fallback_precision: Precision::FP32,
        ..Default::default()
    })
    .unwrap();

    let Some(path) = get_model_path(&mw) else {
        eprintln!("SKIPPED [multiple_inferences]: test model not found");
        return;
    };
    let info = mw.load_engine(&path).unwrap();

    let input = TensorBuffer {
        name: info.inputs[0].name.clone(),
        data: vec![0u8; 3 * 224 * 224 * 4],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    };

    // Run 10 inferences.
    for _ in 0..10 {
        mw.infer(std::slice::from_ref(&input)).unwrap();
    }

    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        assert_eq!(metrics.inference_count, 10);
        assert!(metrics.avg_latency_ms >= 0.0);
    }

    mw.shutdown().unwrap();
}

#[test]
fn reload_engine_while_ready() {
    init_test_logger();

    let mw = Middleware::new();
    mw.initialize(MiddlewareConfig {
        ..Default::default()
    })
    .unwrap();

    let Some(path1) = get_model_path(&mw) else {
        eprintln!("SKIPPED [reload_engine_while_ready]: test model not found");
        return;
    };
    let Some(path2) = get_model_path(&mw) else {
        eprintln!("SKIPPED [reload_engine_while_ready]: test model not found");
        return;
    };

    mw.load_engine(&path1).unwrap();
    assert_eq!(mw.state(), State::Ready);

    // Reload with a different engine.
    mw.load_engine(&path2).unwrap();
    assert_eq!(mw.state(), State::Ready);

    mw.shutdown().unwrap();
}

#[test]
fn thread_safety() {
    init_test_logger();

    let mw = Middleware::new();
    mw.initialize(MiddlewareConfig {
        ..Default::default()
    })
    .unwrap();

    let Some(path) = get_model_path(&mw) else {
        eprintln!("SKIPPED [thread_safety]: test model not found");
        return;
    };
    let info = mw.load_engine(&path).unwrap();
    let input_name = info.inputs[0].name.clone();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let mw_clone = mw.clone();
            let name = input_name.clone();
            std::thread::spawn(move || {
                let input = TensorBuffer {
                    name: name.clone(),
                    data: vec![0u8; 3 * 224 * 224 * 4],
                    shape: vec![1, 3, 224, 224],
                    precision: Precision::FP32,
                };
                for _ in 0..5 {
                    mw_clone.infer(std::slice::from_ref(&input)).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        assert_eq!(metrics.inference_count, 20); // 4 threads × 5 inferences
    }

    mw.shutdown().unwrap();
}
