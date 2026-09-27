//! CPU inference through a system-provided ONNX Runtime C API.
//!
//! This implementation exists for boards where the Rust `ort` session builder
//! deadlocks. Enable it explicitly with the `cpu-system-ort` feature.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};

use tracing::{debug, info};

use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

extern "C" {
    fn cpu_ort_init() -> c_int;
    fn cpu_ort_load_model(model_path: *const c_char) -> c_int;
    fn cpu_ort_get_input_name(buf: *mut c_char, buf_len: c_int) -> c_int;
    fn cpu_ort_get_output_name(buf: *mut c_char, buf_len: c_int) -> c_int;
    fn cpu_ort_run(
        input_data: *const c_float,
        input_elems: i64,
        output_data: *mut c_float,
        output_buf_capacity: i64,
        output_count: *mut i64,
    ) -> c_int;
    fn cpu_ort_destroy();
}

// This board-specific fallback currently supports ImageNet classifiers with
// MobileNet-style I/O. Use the portable `cpu` feature for arbitrary models.
const INPUT_SHAPE: [usize; 4] = [1, 3, 224, 224];
const INPUT_ELEMS: usize = 3 * 224 * 224;
const OUTPUT_CAPACITY: usize = 1000;

pub struct CpuAdapter {
    state: parking_lot::RwLock<CpuState>,
}

struct CpuState {
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
    initialized: bool,
}

impl CpuAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(CpuState {
                engine_info: None,
                buffers_allocated: false,
                initialized: false,
            }),
        }
    }
}

impl Default for CpuAdapter {
    fn default() -> Self {
        Self::new()
    }
}

fn read_c_name(
    get_name: unsafe extern "C" fn(*mut c_char, c_int) -> c_int,
) -> MiddlewareResult<String> {
    let mut buf = vec![0u8; 128];
    // SAFETY: `buf` is writable for `buf.len()` bytes and the shim guarantees
    // NUL termination on success.
    let rc = unsafe { get_name(buf.as_mut_ptr().cast(), buf.len() as c_int) };
    if rc != 0 {
        return Err(MiddlewareError::EngineLoadFailed(format!(
            "Failed to read tensor name (rc={rc})"
        )));
    }
    // SAFETY: The successful shim call above wrote a NUL-terminated string.
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr().cast()) };
    Ok(cstr.to_string_lossy().into_owned())
}

impl InferenceBackend for CpuAdapter {
    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        info!(backend = "cpu", engine_path = %path, "Loading model via system ONNX Runtime C API");

        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Model file not found: {path}"
            )));
        }

        // SAFETY: Initializes process-global state owned by the C shim.
        let rc = unsafe { cpu_ort_init() };
        if rc != 0 {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "cpu_ort_init failed (rc={rc})"
            )));
        }

        let c_path = CString::new(path)
            .map_err(|e| MiddlewareError::EngineLoadFailed(format!("Invalid model path: {e}")))?;

        // SAFETY: `c_path` is NUL-terminated and remains alive for the call.
        let rc = unsafe { cpu_ort_load_model(c_path.as_ptr()) };
        if rc != 0 {
            // SAFETY: Releases any state allocated by `cpu_ort_init`.
            unsafe { cpu_ort_destroy() };
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "cpu_ort_load_model failed (rc={rc})"
            )));
        }

        let input_name = read_c_name(cpu_ort_get_input_name)?;
        let output_name = read_c_name(cpu_ort_get_output_name)?;
        let info = EngineInfo {
            name: std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "cpu_model".into()),
            inputs: vec![TensorSpec {
                name: input_name,
                shape: INPUT_SHAPE.to_vec(),
                precision: Precision::FP32,
            }],
            outputs: vec![TensorSpec {
                name: output_name,
                shape: vec![OUTPUT_CAPACITY],
                precision: Precision::FP32,
            }],
            memory_bytes: 0,
        };

        let mut state = self.state.write();
        state.engine_info = Some(info.clone());
        state.initialized = true;
        info!(backend = "cpu", engine = %info.name, "Model loaded successfully");
        Ok(info)
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        self.state.write().buffers_allocated = true;
        debug!(backend = "cpu", "Buffers allocated (no-op for CPU)");
        Ok(())
    }

    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let state = self.state.read();
        if state.engine_info.is_none() || !state.initialized {
            return Err(MiddlewareError::EngineNotLoaded);
        }
        if !state.buffers_allocated {
            return Err(MiddlewareError::InferenceFailed(
                "Buffers not allocated".into(),
            ));
        }
        let input = inputs
            .first()
            .ok_or_else(|| MiddlewareError::InferenceFailed("No inputs provided".into()))?;
        input
            .validate()
            .map_err(|e| MiddlewareError::InferenceFailed(format!("Input validation: {e}")))?;
        if input.precision != Precision::FP32 {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Unsupported precision for CPU fallback: {:?}",
                input.precision
            )));
        }

        let f32_data: &[f32] = bytemuck::cast_slice(&input.data);
        if f32_data.len() != INPUT_ELEMS {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Unexpected input element count: got {}, expected {INPUT_ELEMS}",
                f32_data.len()
            )));
        }

        let mut output_data = vec![0f32; OUTPUT_CAPACITY];
        let mut output_count = 0i64;
        // SAFETY: Input and output slices are valid for the supplied lengths;
        // `output_count` is a valid writable pointer for the duration of the call.
        let rc = unsafe {
            cpu_ort_run(
                f32_data.as_ptr(),
                f32_data.len() as i64,
                output_data.as_mut_ptr(),
                OUTPUT_CAPACITY as i64,
                &mut output_count,
            )
        };
        if rc != 0 {
            return Err(MiddlewareError::InferenceFailed(format!(
                "cpu_ort_run failed (rc={rc})"
            )));
        }
        if !(0..=OUTPUT_CAPACITY as i64).contains(&output_count) {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Invalid output element count returned by ONNX Runtime: {output_count}"
            )));
        }

        output_data.truncate(output_count as usize);
        let output_name = state
            .engine_info
            .as_ref()
            .and_then(|info| info.outputs.first())
            .map(|spec| spec.name.clone())
            .unwrap_or_else(|| "output".into());

        Ok(vec![TensorBuffer {
            name: output_name,
            data: bytemuck::cast_slice(&output_data).to_vec(),
            shape: vec![output_count as usize],
            precision: Precision::FP32,
        }])
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        if state.initialized {
            // SAFETY: The shim was successfully initialized and has not yet
            // been destroyed by this adapter.
            unsafe { cpu_ort_destroy() };
        }
        state.engine_info = None;
        state.buffers_allocated = false;
        state.initialized = false;
        info!(backend = "cpu", "Backend released");
        Ok(())
    }

    fn engine_info(&self) -> Option<EngineInfo> {
        self.state.read().engine_info.clone()
    }

    fn is_ready(&self) -> bool {
        let state = self.state.read();
        state.engine_info.is_some() && state.buffers_allocated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_adapter_basics() {
        let adapter = CpuAdapter::new();
        assert_eq!(adapter.backend_name(), "cpu");
        assert!(!adapter.is_ready());
    }

    #[test]
    fn cpu_load_nonexistent_fails() {
        assert!(CpuAdapter::new()
            .load_engine("/nonexistent/model.onnx")
            .is_err());
    }
}
