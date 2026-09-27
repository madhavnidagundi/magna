// =============================================================================
// Magna Middleware — Texas Instruments TDA4 Backend Adapter
// =============================================================================
#[allow(unused_imports)]
use std::ffi::c_void;
use tracing::{debug, info};

#[cfg(any(all(feature = "ti", have_dlr), feature = "mock-ti"))]
use crate::inference::traits::TensorSpec;
use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

#[cfg(all(feature = "ti", have_dlr))]
extern "C" {
    fn tidl_rt_init() -> *mut c_void;
    fn tidl_rt_load_model(
        handle: *mut c_void,
        model_path: *const u8,
        model_path_len: usize,
        artifacts_dir: *const u8,
        artifacts_dir_len: usize,
    ) -> i32;
    fn tidl_rt_alloc_tensors(handle: *mut c_void) -> i32;
    fn tidl_rt_process(
        handle: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
    fn tidl_rt_get_input_elems(handle: *mut c_void) -> i32;
    fn tidl_rt_get_output_elems(handle: *mut c_void) -> i32;
    fn tidl_rt_destroy(handle: *mut c_void);
}

pub struct TiAdapter {
    state: parking_lot::RwLock<TiState>,
}

struct TiState {
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
    is_simulated: bool,
    #[cfg(all(feature = "ti", have_dlr))]
    tidl_handle: Option<*mut c_void>,
}

// SAFETY:
// TiState contains a raw pointer (*mut c_void) which is only accessed
// through a RwLock, ensuring proper synchronization.
// The C shim does not use thread-local or unsynchronized global state,
// so sharing across threads is safe.
unsafe impl Send for TiState {}
// SAFETY:
// TiState contains a raw pointer (*mut c_void) which is only accessed
// through a RwLock, ensuring proper synchronization.
// The C shim does not use thread-local or unsynchronized global state,
// so sharing across threads is safe.
unsafe impl Sync for TiState {}

impl TiAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(TiState {
                engine_info: None,
                buffers_allocated: false,
                is_simulated: true,
                #[cfg(all(feature = "ti", have_dlr))]
                tidl_handle: None,
            }),
        }
    }
}

impl Default for TiAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for TiAdapter {
    fn backend_name(&self) -> &str {
        "ti"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        #[allow(unused_mut, unused_variables)]
        let mut state = self.state.write();
        info!(backend = "ti", engine_path = %path, "Loading model");

        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Engine file not found: {}",
                path
            )));
        }

        #[cfg(all(feature = "ti", have_dlr))]
        {
            // SAFETY:
            // `tidl_rt_init` returns either a valid heap-allocated handle or null.
            // We immediately check for null before using it, ensuring the pointer
            // is valid for all subsequent FFI calls.
            let handle = unsafe { tidl_rt_init() };
            if handle.is_null() {
                return Err(MiddlewareError::EngineLoadFailed(
                    "DLR runtime init failed".into(),
                ));
            }

            let path_bytes = path.as_bytes();
            // The TIDL artifacts folder (contains allowedNode.txt, param.yaml,
            // etc.) is a separate directory from the model file itself and
            // cannot be reliably derived from the model path alone (layouts
            // vary: some deployments nest the model inside the artifacts
            // folder, others keep them as siblings). Prefer an explicit
            // env var; fall back to the model's parent directory as a
            // best-effort guess only if unset, with a loud warning since
            // that guess is frequently wrong and silently disables all
            // TIDL/DSP offload (the model still runs, just entirely on ARM).
            let artifacts_dir = std::env::var("MAGNA_TIDL_ARTIFACTS_DIR").unwrap_or_else(|_| {
                let guess = std::path::Path::new(path)
                    .parent()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                tracing::warn!(
                    backend = "ti",
                    guessed_dir = %guess,
                    "MAGNA_TIDL_ARTIFACTS_DIR not set; guessing artifacts folder from \
                     model path. If this is wrong, TIDL will silently run the \
                     entire model on ARM with zero DSP offload. Set \
                     MAGNA_TIDL_ARTIFACTS_DIR explicitly to avoid this."
                );
                guess
            });
            let artifacts_bytes = artifacts_dir.as_bytes();
            // SAFETY:
            // - `handle` is non-null (validated above).
            // - `path_bytes` and `artifacts_bytes` come from valid Rust strings,
            //   so both pointer/length pairs are valid for the duration of this call.
            // - The C function does not retain either pointer beyond this call.
            let ret = unsafe {
                tidl_rt_load_model(
                    handle,
                    path_bytes.as_ptr(),
                    path_bytes.len(),
                    artifacts_bytes.as_ptr(),
                    artifacts_bytes.len(),
                )
            };
            if ret != 0 {
                // SAFETY:
                // `handle` was returned by `tidl_rt_init` and has not yet been freed.
                // This call transfers ownership to the C side and must be called exactly once.
                unsafe {
                    tidl_rt_destroy(handle);
                }
                return Err(MiddlewareError::EngineLoadFailed(format!(
                    "DLR model load failed (error {})",
                    ret
                )));
            }
            // SAFETY:
            // `handle` is a valid initialized TIDL runtime handle,
            // and the function only reads metadata without modifying memory.
            let input_elems = unsafe { tidl_rt_get_input_elems(handle) } as usize;
            // SAFETY:
            // `handle` is valid and initialized, and the function returns
            // output tensor metadata without dereferencing invalid memory.
            let output_elems = unsafe { tidl_rt_get_output_elems(handle) } as usize;

            state.tidl_handle = Some(handle);
            state.is_simulated = false;

            let input_shape = vec![input_elems];
            let output_shape = vec![output_elems];

            let info = EngineInfo {
                name: std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "dlr_model".into()),
                inputs: vec![TensorSpec {
                    name: "input.1Net_IN".into(),
                    shape: input_shape,
                    precision: Precision::FP32,
                }],
                outputs: vec![TensorSpec {
                    name: "output".into(),
                    shape: output_shape,
                    precision: Precision::FP32,
                }],
                memory_bytes: 0,
            };
            state.engine_info = Some(info.clone());
            info!(backend = "ti", engine = %info.name, "DLR model loaded");
            return Ok(info);
        }

        // Production builds with no real TI runtime available (neither DLR nor a
        // TIDL-enabled ONNX Runtime were found at build time) must fail loudly
        // rather than fabricate a fake "sim" engine. A mock is only available
        // behind an explicit opt-in feature for testing (see below).
        #[cfg(feature = "mock-ti")]
        #[allow(unreachable_code)]
        {
            let info = EngineInfo {
                name: "ti-mock".into(),
                inputs: vec![TensorSpec {
                    name: "input".into(),
                    shape: vec![1, 3, 224, 224],
                    precision: Precision::FP32,
                }],
                outputs: vec![TensorSpec {
                    name: "output".into(),
                    shape: vec![1, 1000],
                    precision: Precision::FP32,
                }],
                memory_bytes: 0,
            };
            state.is_simulated = true;
            state.engine_info = Some(info.clone());
            return Ok(info);
        }

        #[allow(unreachable_code)]
        Err(MiddlewareError::BackendUnavailable(
            "TI runtime unavailable: neither DLR nor a TIDL-enabled ONNX Runtime \
             was found at build time. Set DLR_INCLUDE_DIR/DLR_LIB_DIR or \
             ORT_TIDL_INCLUDE_DIR/ORT_TIDL_LIB_DIR, or build with the \
             `mock-ti` feature for testing."
                .into(),
        ))
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        #[cfg(all(feature = "ti", have_dlr))]
        {
            if !state.is_simulated {
                if let Some(handle) = state.tidl_handle {
                    // SAFETY:
                    // `handle` is valid and initialized.
                    // The runtime allocates internal buffers and does not access invalid memory.
                    let ret = unsafe { tidl_rt_alloc_tensors(handle) };
                    if ret != 0 {
                        return Err(MiddlewareError::BufferAllocationFailed(format!(
                            "CMEM alloc failed (error {})",
                            ret
                        )));
                    }
                }
            }
        }
        state.buffers_allocated = true;
        debug!(backend = "ti", "Buffers allocated");
        Ok(())
    }

    /// Run inference via TI DLR (TIDL runtime).
    ///
    /// **NOTE:** The TIDL FFI uses positional byte-buffer binding (`tidl_rt_process`
    /// accepts raw pointers only). The `input.name` field is NOT sent to the
    /// runtime and has no effect on execution. Tensor names in `EngineInfo` are
    /// populated for metadata correctness but are not validated at inference time.
    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let state = self.state.read();
        let info = state
            .engine_info
            .as_ref()
            .ok_or(MiddlewareError::EngineNotLoaded)?;

        if !state.buffers_allocated {
            return Err(MiddlewareError::InferenceFailed(
                "Buffers not allocated".into(),
            ));
        }
        if inputs.is_empty() {
            return Err(MiddlewareError::InferenceFailed(
                "No inputs provided".into(),
            ));
        }

        #[allow(unused_variables)]
        let input = &inputs[0];
        let output_spec = &info.outputs[0];
        let output_elems: usize = output_spec.shape.iter().product();
        #[allow(unused_variables)]
        let output_bytes = output_elems * Precision::FP32.element_size();

        #[cfg(all(feature = "ti", have_dlr))]
        {
            if !state.is_simulated {
                if let Some(handle) = state.tidl_handle {
                    // Model expects uint8 NCHW [1,3,224,224]
                    // Pass input bytes directly — benchmark sends uint8 NCHW
                    let mut output_data = vec![0u8; output_bytes];
                    // SAFETY:
                    // - `handle` is valid and initialized.
                    // - `input.data` is a valid buffer for reads of `input.data.len()` bytes.
                    // - `output_data` is a valid mutable buffer for writes of `output_bytes` bytes.
                    // - The C function guarantees it writes at most `output_bytes`,
                    //   preventing buffer overflow.
                    let ret = unsafe {
                        tidl_rt_process(
                            handle,
                            input.data.as_ptr(),
                            input.data.len(),
                            output_data.as_mut_ptr(),
                            output_bytes,
                        )
                    };
                    if ret != 0 {
                        return Err(MiddlewareError::InferenceFailed(format!(
                            "DLR inference failed (error {})",
                            ret
                        )));
                    }
                    return Ok(vec![TensorBuffer {
                        name: output_spec.name.clone(),
                        data: output_data,
                        shape: output_spec.shape.clone(),
                        precision: Precision::FP32,
                    }]);
                }
            }
        }

        // Mock inference (uniform scores) is only reachable when `state.is_simulated`
        // was set by the opt-in `mock-ti` load path above; production builds can
        // never reach engine_info() != None with is_simulated == true otherwise,
        // since load_engine() now errors instead of fabricating a sim engine.
        #[cfg(feature = "mock-ti")]
        if state.is_simulated {
            let value = 1.0f32 / output_elems as f32;
            let mut output_data = vec![0u8; output_bytes];
            for chunk in output_data.as_chunks_mut::<4>().0 {
                chunk.copy_from_slice(&value.to_le_bytes());
            }
            return Ok(vec![TensorBuffer {
                name: output_spec.name.clone(),
                data: output_data,
                shape: output_spec.shape.clone(),
                precision: Precision::FP32,
            }]);
        }

        Err(MiddlewareError::BackendUnavailable(
            "TI runtime unavailable: cannot run inference without a loaded DLR/TIDL engine.".into(),
        ))
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        #[cfg(all(feature = "ti", have_dlr))]
        {
            if let Some(handle) = state.tidl_handle.take() {
                // SAFETY:
                // `handle` was previously initialized and not yet freed.
                // This call releases ownership and must only be called once.
                unsafe {
                    tidl_rt_destroy(handle);
                }
            }
        }
        state.engine_info = None;
        state.buffers_allocated = false;
        state.is_simulated = true;
        info!(backend = "ti", "Backend released");
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
    fn ti_adapter_basics() {
        let a = TiAdapter::new();
        assert_eq!(a.backend_name(), "ti");
        assert!(!a.is_ready());
    }

    #[test]
    fn ti_adapter_default() {
        let a = TiAdapter::default();
        assert!(a.engine_info().is_none());
    }

    #[test]
    fn ti_infer_without_load_fails() {
        let a = TiAdapter::new();
        let input = TensorBuffer::from_f32("test", &[0.0; 6], vec![1, 2, 3]);
        assert!(a.infer(&[input]).is_err());
    }
}
