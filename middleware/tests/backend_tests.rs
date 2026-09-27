// =============================================================================
// Integration Test — Backend Adapters
// =============================================================================
//! Tests that each backend adapter correctly implements the InferenceBackend
//! trait lifecycle.

use magna_middleware::backends::cpu::adapter::CpuAdapter;
#[cfg(feature = "nvidia")]
use magna_middleware::backends::nvidia::adapter::NvidiaAdapter;
#[cfg(feature = "nvidia")]
use magna_middleware::backends::nvidia::NvidiaHardware;
#[cfg(feature = "qualcomm")]
use magna_middleware::backends::qualcomm::adapter::QualcommAdapter;
#[cfg(feature = "ti")]
use magna_middleware::backends::ti::adapter::TiAdapter;
use magna_middleware::inference::traits::{InferenceBackend, TensorBuffer};
use magna_middleware::utils::errors::Precision;

/// Resolves the TensorRT test engine path.
/// Checks TEST_ENGINE_PATH env var first, then falls back to the default location.
#[cfg(feature = "nvidia")]
fn nvidia_test_engine() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("TEST_ENGINE_PATH") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let p = std::path::Path::new(&manifest_dir)
        .join("../models/mobilenetv4_nvidia_optimized_int8.engine");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

#[cfg(feature = "qualcomm")]
fn make_dummy_file(ext: &str) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut f = tempfile::Builder::new()
        .suffix(&format!(".{}", ext))
        .tempfile()
        .unwrap();
    f.write_all(b"DUMMY_ENGINE_DATA").unwrap();
    f
}

/// Generic lifecycle test for any backend.
fn test_backend_lifecycle(adapter: &mut dyn InferenceBackend, engine_file: &str) {
    // Not ready initially.
    assert!(!adapter.is_ready());

    // Load engine.
    let info = adapter.load_engine(engine_file).unwrap();
    assert!(!info.name.is_empty());
    assert_eq!(info.inputs[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(info.outputs[0].shape, vec![1, 1000]);

    // Allocate buffers.
    adapter.allocate_buffers().unwrap();
    assert!(adapter.is_ready());

    // Build input tensor using the actual input name from the engine info.
    let input = TensorBuffer {
        name: info.inputs[0].name.clone(),
        data: vec![0u8; 3 * 224 * 224 * 4],
        shape: vec![1, 3, 224, 224],
        precision: Precision::FP32,
    };

    // Run inference.
    let output = adapter.infer(&[input]).unwrap();
    assert_eq!(output[0].shape, vec![1, 1000]);
    assert_eq!(output[0].precision, Precision::FP32);
    assert!(!output[0].data.is_empty());

    // Verify the output is interpretable as f32.
    let scores = output[0].try_as_f32_slice().unwrap();
    assert_eq!(scores.len(), 1000);

    // Check engine info.
    assert!(adapter.engine_info().is_some());

    // Release.
    adapter.release().unwrap();
    assert!(!adapter.is_ready());
    assert!(adapter.engine_info().is_none());
}

#[test]
#[cfg(feature = "nvidia")]
fn nvidia_adapter_lifecycle() {
    let Some(p) = nvidia_test_engine() else {
        eprintln!("SKIPPED [nvidia_adapter_lifecycle]: TensorRT test engine not found (set TEST_ENGINE_PATH)");
        return;
    };
    let mut adapter = NvidiaAdapter::new(NvidiaHardware::Orin);
    test_backend_lifecycle(&mut adapter, p.to_string_lossy().as_ref());
}

#[test]
#[cfg(feature = "thor")]
fn nvidia_adapter_with_thor() {
    let Some(p) = nvidia_test_engine() else {
        eprintln!("SKIPPED [nvidia_adapter_with_thor]: TensorRT test engine not found (set TEST_ENGINE_PATH)");
        return;
    };
    let mut adapter = NvidiaAdapter::new(NvidiaHardware::Thor);
    assert_eq!(adapter.max_precision(), Precision::FP8);
    test_backend_lifecycle(&mut adapter, p.to_string_lossy().as_ref());
}

#[test]
#[cfg(feature = "qualcomm")]
fn qualcomm_dummy_engine_is_rejected() {
    let file = make_dummy_file("bin");
    let adapter = QualcommAdapter::new();

    let result = adapter.load_engine(file.path().to_str().unwrap());

    assert!(
        result.is_err(),
        "DUMMY engine must never be accepted as a real Qualcomm engine"
    );
    assert!(
        !adapter.is_ready(),
        "Rejected dummy engine must not leave the adapter ready"
    );
}

#[test]
fn cpu_adapter_lifecycle() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let p = std::path::Path::new(&manifest_dir).join("../models/mobilenetv2.onnx");

    if !p.exists() {
        eprintln!(
            "SKIPPED [cpu_adapter_lifecycle]: ONNX test model not found at {}",
            p.display()
        );
        return;
    }

    let mut adapter = CpuAdapter::new();
    test_backend_lifecycle(&mut adapter, p.to_string_lossy().as_ref());
}

#[test]
fn backend_names() {
    #[cfg(feature = "nvidia")]
    assert_eq!(
        NvidiaAdapter::new(NvidiaHardware::Orin).backend_name(),
        "nvidia"
    );
    #[cfg(feature = "qualcomm")]
    assert_eq!(QualcommAdapter::new().backend_name(), "qualcomm");
    #[cfg(feature = "ti")]
    assert_eq!(TiAdapter::new().backend_name(), "ti");
    assert_eq!(CpuAdapter::new().backend_name(), "cpu");
}

#[cfg(all(feature = "nvidia", feature = "qualcomm", feature = "ti"))]
#[test]
fn load_nonexistent_fails_all_adapters() {
    #[cfg(feature = "nvidia")]
    let nv = NvidiaAdapter::new(NvidiaHardware::Orin);
    #[cfg(feature = "qualcomm")]
    let qc = QualcommAdapter::new();
    #[cfg(feature = "ti")]
    let ti = TiAdapter::new();
    let cpu = CpuAdapter::new();

    #[cfg(feature = "nvidia")]
    assert!(nv.load_engine("/does/not/exist.engine").is_err());
    #[cfg(feature = "qualcomm")]
    assert!(qc.load_engine("/does/not/exist.bin").is_err());
    #[cfg(feature = "ti")]
    assert!(ti.load_engine("/does/not/exist.tidl").is_err());
    assert!(cpu.load_engine("/does/not/exist.onnx").is_err());
}

#[cfg(all(feature = "nvidia", feature = "qualcomm", feature = "ti"))]
#[test]
fn infer_without_load_fails() {
    #[cfg(feature = "nvidia")]
    let nv = NvidiaAdapter::new(NvidiaHardware::Orin);
    #[cfg(feature = "qualcomm")]
    let qc = QualcommAdapter::new();
    #[cfg(feature = "ti")]
    let ti = TiAdapter::new();
    let cpu = CpuAdapter::new();
    let input = dummy_input();

    #[cfg(feature = "nvidia")]
    assert!(nv.infer(std::slice::from_ref(&input)).is_err());
    #[cfg(feature = "qualcomm")]
    assert!(qc.infer(std::slice::from_ref(&input)).is_err());
    #[cfg(feature = "ti")]
    assert!(ti.infer(std::slice::from_ref(&input)).is_err());
    assert!(cpu.infer(std::slice::from_ref(&input)).is_err());
}

#[test]
fn default_trait_implementations() {
    #[cfg(feature = "nvidia")]
    let _nv = NvidiaAdapter::default();
    #[cfg(feature = "qualcomm")]
    let _qc = QualcommAdapter::default();
    #[cfg(feature = "ti")]
    let _ti = TiAdapter::default();
    let _cpu = CpuAdapter::default();
}
