//! TI runtime selection tests. Native inference is opt-in and requires real assets;
//! see docs/hardware_smoke_tests.md for the setup and feature matrix.
#![cfg(feature = "ti")]

use magna_middleware::backends::ti::adapter::TiAdapter;
use magna_middleware::inference::traits::{InferenceBackend, TensorBuffer};
#[cfg(all(not(have_dlr), not(feature = "mock-ti")))]
use magna_middleware::utils::errors::MiddlewareError;
use magna_middleware::utils::errors::Precision;

#[cfg(not(have_dlr))]
fn dummy_model() -> tempfile::NamedTempFile {
    use std::io::Write;

    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(b"DUMMY_ENGINE_DATA").unwrap();
    file
}

#[test]
#[cfg(all(not(have_dlr), not(feature = "mock-ti")))]
fn ti_missing_runtime_returns_backend_unavailable() {
    // An existing file ensures this exercises runtime selection, not path validation.
    let file = dummy_model();
    let adapter = TiAdapter::new();
    let error = adapter
        .load_engine(file.path().to_str().unwrap())
        .unwrap_err();
    assert!(matches!(error, MiddlewareError::BackendUnavailable(_)));
    assert!(adapter.engine_info().is_none());
    assert!(!adapter.is_ready());

    let input = TensorBuffer::from_f32("input", &[0.0], vec![1]);
    assert!(matches!(
        adapter.infer(&[input]),
        Err(MiddlewareError::EngineNotLoaded)
    ));
    adapter.release().unwrap();
    assert!(!adapter.is_ready());
}

#[test]
#[cfg(all(not(have_dlr), feature = "mock-ti"))]
fn ti_mock_lifecycle_returns_explicit_mock_output() {
    // mock-ti is a fallback, not an override of an installed native runtime.
    let file = dummy_model();
    let adapter = TiAdapter::new();
    assert!(!adapter.is_ready());

    let info = adapter.load_engine(file.path().to_str().unwrap()).unwrap();
    assert_eq!(info.name, "ti-mock");
    assert_eq!(info.inputs.len(), 1);
    assert_eq!(info.outputs.len(), 1);
    assert_eq!(info.inputs[0].name, "input");
    assert_eq!(info.inputs[0].shape, vec![1, 3, 224, 224]);
    assert_eq!(info.inputs[0].precision, Precision::FP32);
    assert_eq!(info.outputs[0].name, "output");
    assert_eq!(info.outputs[0].shape, vec![1, 1000]);
    assert_eq!(info.outputs[0].precision, Precision::FP32);
    assert!(!adapter.is_ready());

    adapter.allocate_buffers().unwrap();
    assert!(adapter.is_ready());
    let input = TensorBuffer::from_f32("input", &vec![0.0; 3 * 224 * 224], vec![1, 3, 224, 224]);
    let outputs = adapter.infer(&[input]).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "output");
    assert_eq!(outputs[0].shape, vec![1, 1000]);
    assert_eq!(outputs[0].precision, Precision::FP32);
    assert_eq!(outputs[0].data.len(), 1000 * Precision::FP32.element_size());
    let scores = outputs[0].try_as_f32_slice().unwrap();
    assert_eq!(scores.len(), 1000);
    assert!(scores.iter().all(|&score| (score - 0.001).abs() < 1e-7));

    adapter.release().unwrap();
    assert!(adapter.engine_info().is_none());
    assert!(!adapter.is_ready());
}

#[test]
#[ignore = "requires native DLR/TIDL, TEST_TI_MODEL_PATH, TEST_TI_INPUT_PATH and TEST_TI_OUTPUT_ELEMS"]
fn ti_native_lifecycle() {
    // Keep this test compiled even without have_dlr: explicitly requesting it
    // must fail, not silently run zero tests, if native runtime cfg regresses.
    let model = std::env::var("TEST_TI_MODEL_PATH")
        .expect("set TEST_TI_MODEL_PATH to a real DLR model directory or TIDL ONNX model");
    let input_path = std::env::var("TEST_TI_INPUT_PATH")
        .expect("set TEST_TI_INPUT_PATH to a preprocessed FP32 input tensor");
    let expected_output_elems: usize = std::env::var("TEST_TI_OUTPUT_ELEMS")
        .expect("set TEST_TI_OUTPUT_ELEMS to the model's known output element count")
        .parse()
        .expect("TEST_TI_OUTPUT_ELEMS must be a positive integer");
    assert!(expected_output_elems > 0);

    let adapter = TiAdapter::new();
    let info = adapter
        .load_engine(&model)
        .expect("native TI model must load");
    assert_ne!(
        info.name, "ti-mock",
        "native test must never use mock inference"
    );
    assert_eq!(info.inputs.len(), 1);
    assert_eq!(info.outputs.len(), 1);
    assert_eq!(info.inputs[0].precision, Precision::FP32);
    assert_eq!(info.outputs[0].precision, Precision::FP32);
    assert_eq!(
        info.outputs[0].shape.iter().product::<usize>(),
        expected_output_elems
    );

    let data = std::fs::read(input_path).expect("read the preprocessed FP32 input tensor");
    let input_elems = info.inputs[0].shape.iter().product::<usize>();
    assert!(input_elems > 0);
    assert_eq!(data.len(), input_elems * Precision::FP32.element_size());
    let input = TensorBuffer {
        name: info.inputs[0].name.clone(),
        data,
        shape: info.inputs[0].shape.clone(),
        precision: Precision::FP32,
    };

    adapter.allocate_buffers().unwrap();
    assert!(adapter.is_ready());
    let outputs = adapter
        .infer(&[input])
        .expect("native TI inference must succeed");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, info.outputs[0].shape);
    assert_eq!(outputs[0].precision, Precision::FP32);
    assert_eq!(
        outputs[0].data.len(),
        expected_output_elems * Precision::FP32.element_size()
    );
    let values = outputs[0].try_as_f32_slice().unwrap();
    assert_eq!(values.len(), expected_output_elems);
    assert!(values.iter().all(|value| value.is_finite()));

    adapter.release().unwrap();
    assert!(!adapter.is_ready());
    assert!(adapter.engine_info().is_none());
}
