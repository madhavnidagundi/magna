// =============================================================================
// Magna Middleware — NVIDIA TensorRT Backend Adapter
// =============================================================================
//! Implements [`InferenceBackend`] for NVIDIA Orin / Thor hardware using
//! TensorRT.
//!
//! On non-NVIDIA platforms (no `nvidia` feature), this module is not compiled,
//! ensuring that simulation paths and mock engines are completely separated.

use std::ffi::c_void;
use tracing::{debug, error, info};

use super::specs::NvidiaHardware;
use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

// ---------------------------------------------------------------------------
// FFI declarations (only linked when `nvidia` feature is enabled)
// ---------------------------------------------------------------------------

extern "C" {
    fn trt_load_engine(path: *const u8, path_len: usize) -> *mut c_void;
    fn trt_destroy_engine(ctx: *mut c_void);
    fn trt_allocate_buffers(ctx: *mut c_void) -> i32;
    fn trt_infer(
        ctx: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
    fn trt_get_device_memory(ctx: *mut c_void) -> i64;
    fn trt_get_input_elems(ctx: *mut c_void) -> i32;
    fn trt_get_output_elems(ctx: *mut c_void) -> i32;
    fn trt_get_input_elem_size(ctx: *mut c_void) -> i32;
    fn trt_get_output_elem_size(ctx: *mut c_void) -> i32;

    fn trt_build_engine(
        onnx_path: *const std::ffi::c_char,
        engine_path: *const std::ffi::c_char,
        precision: i32,
        calib_cache: *const std::ffi::c_char,
    ) -> bool;

    fn trt_capture_graph(ctx: *mut c_void) -> bool;
    fn trt_infer_graph(
        ctx: *mut c_void,
        input: *const u8,
        input_bytes: usize,
        output: *mut u8,
        output_bytes: usize,
    ) -> i32;
}

// ---------------------------------------------------------------------------
// TrtContext wrapper (Send but NOT Sync — TRT is not thread-safe)
// ---------------------------------------------------------------------------

struct TrtContext {
    ptr: *mut c_void,
}

// SAFETY: TensorRT contexts can be moved between threads.
unsafe impl Send for TrtContext {}
// SAFETY: `TrtContext` is never accessed directly across threads. All access goes through
// `NvidiaAdapter::state`, which is a `parking_lot::Mutex<NvidiaState>`, guaranteeing exclusive access.
unsafe impl Sync for TrtContext {}

impl Drop for TrtContext {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `self.ptr` is checked for null before being passed to `trt_destroy_engine`.
            // The pointer is exclusively owned by `TrtContext` and was validly initialized by `trt_load_engine`.
            unsafe {
                trt_destroy_engine(self.ptr);
            }
            info!(backend = "nvidia", "Engine resources released");
        }
    }
}

// ---------------------------------------------------------------------------
// NvidiaAdapter
// ---------------------------------------------------------------------------

/// NVIDIA TensorRT backend adapter.
pub struct NvidiaAdapter {
    hardware: NvidiaHardware,
    state: parking_lot::Mutex<NvidiaState>,
}

struct NvidiaState {
    context: Option<TrtContext>,
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
}

impl NvidiaAdapter {
    pub fn new(hardware: NvidiaHardware) -> Self {
        Self {
            hardware,
            state: parking_lot::Mutex::new(NvidiaState {
                context: None,
                engine_info: None,
                buffers_allocated: false,
            }),
        }
    }

    /// Maximum precision supported by this adapter.
    pub fn max_precision(&self) -> Precision {
        match self.hardware {
            NvidiaHardware::Thor => Precision::FP8,
            NvidiaHardware::Orin => Precision::FP16,
        }
    }
}

impl InferenceBackend for NvidiaAdapter {
    fn backend_name(&self) -> &str {
        "nvidia"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        info!(backend = "nvidia", engine_path = %path, "Loading engine");

        // Validate file exists
        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Engine file not found: {}",
                path
            )));
        }

        {
            let path_bytes = path.as_bytes();
            // SAFETY: `path_bytes` points to a valid byte slice representing the file path.
            // `trt_load_engine` safely reads the byte array up to the provided length and allocates
            // an opaque context pointer, returning null on failure.
            let ctx_ptr = unsafe { trt_load_engine(path_bytes.as_ptr(), path_bytes.len()) };

            if ctx_ptr.is_null() {
                return Err(MiddlewareError::EngineLoadFailed(
                    "TensorRT returned null context — engine deserialization failed".into(),
                ));
            }

            // SAFETY: `ctx_ptr` was successfully allocated by `trt_load_engine` and explicitly checked
            // to be non-null. The C library guarantees these accessors safely read engine properties
            // without mutating the context.
            let input_elems = unsafe { trt_get_input_elems(ctx_ptr) } as usize;
            // SAFETY: `ctx_ptr` is valid and non-null as verified above.
            let output_elems = unsafe { trt_get_output_elems(ctx_ptr) } as usize;
            // SAFETY: `ctx_ptr` is valid and non-null as verified above.
            let input_bytes = unsafe { trt_get_input_elem_size(ctx_ptr) } as usize;
            // SAFETY: `ctx_ptr` is valid and non-null as verified above.
            let output_bytes = unsafe { trt_get_output_elem_size(ctx_ptr) } as usize;
            // SAFETY: `ctx_ptr` is valid and non-null as verified above.
            let device_mem = unsafe { trt_get_device_memory(ctx_ptr) } as u64;

            let mut state = self.state.lock();
            state.context = Some(TrtContext { ptr: ctx_ptr });

            // Dynamically determine shape from the engine
            // (the C++ side should report actual shapes; these defaults are for
            // engines that don't report them)
            let precision_val = match input_bytes {
                4 => Precision::FP32,
                2 => Precision::FP16,
                1 => {
                    if matches!(self.hardware, NvidiaHardware::Thor) {
                        Precision::FP8
                    } else {
                        Precision::INT8
                    }
                }
                _ => Precision::FP32, // fallback
            };

            let output_precision_val = match output_bytes {
                4 => Precision::FP32,
                2 => Precision::FP16,
                1 => {
                    if matches!(self.hardware, NvidiaHardware::Thor) {
                        Precision::FP8
                    } else {
                        Precision::INT8
                    }
                }
                _ => Precision::FP32, // fallback
            };

            let info = EngineInfo {
                name: std::path::Path::new(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "engine".into()),
                inputs: vec![TensorSpec {
                    name: "input".to_string(),
                    shape: if input_elems > 0 {
                        // Try to infer shape from element count
                        infer_shape_from_elements(input_elems)
                    } else {
                        vec![1, 3, 224, 224] // fallback
                    },
                    precision: precision_val,
                }],
                outputs: vec![TensorSpec {
                    name: "output".to_string(),
                    shape: if output_elems > 0 {
                        vec![1, output_elems]
                    } else {
                        vec![1, 1000] // fallback
                    },
                    precision: output_precision_val,
                }],
                memory_bytes: device_mem,
            };

            info!(
                backend      = "nvidia",
                engine       = %info.name,
                memory_bytes = device_mem,
                "Engine loaded"
            );
            state.engine_info = Some(info.clone());
            Ok(info)
        }
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.lock();
        {
            if let Some(ref ctx) = state.context {
                // SAFETY: `ctx.ptr` is a valid initialized TensorRT context managed by `TrtContext`.
                // `trt_allocate_buffers` handles CUDA memory allocation internally and safely binds
                // it to the context.
                let result = unsafe { trt_allocate_buffers(ctx.ptr) };
                if result != 0 {
                    return Err(MiddlewareError::BufferAllocationFailed(format!(
                        "CUDA buffer allocation failed (error code {})",
                        result
                    )));
                }
            } else if state.engine_info.is_none() {
                return Err(MiddlewareError::EngineNotLoaded);
            }
        }

        state.buffers_allocated = true;

        // Thor Optimization: Capture CUDA Graph
        {
            if matches!(self.hardware, NvidiaHardware::Thor) {
                if let Some(ref ctx) = state.context {
                    info!(
                        backend = "nvidia",
                        "Thor detected: Capturing CUDA Graph for 0-latency launch"
                    );
                    // SAFETY: `ctx.ptr` is a valid, initialized TensorRT context managed by `TrtContext`.
                    unsafe { trt_capture_graph(ctx.ptr) };
                }
            }
        }

        debug!(backend = "nvidia", "Buffers allocated");
        Ok(())
    }

    /// Run inference via TensorRT.
    ///
    /// **NOTE:** TensorRT FFI uses positional byte-buffer binding (`trt_infer`
    /// accepts raw pointers only). The `input.name` field is NOT sent to the
    /// runtime and has no effect on execution. Tensor names in `EngineInfo` are
    /// populated for metadata correctness but are not validated at inference time.
    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let state = self.state.lock();
        let info = state
            .engine_info
            .as_ref()
            .ok_or(MiddlewareError::EngineNotLoaded)?;

        if !state.buffers_allocated {
            return Err(MiddlewareError::InferenceFailed(
                "Buffers not allocated — call allocate_buffers() first".into(),
            ));
        }

        if inputs.is_empty() {
            return Err(MiddlewareError::InferenceFailed(
                "No inputs provided".into(),
            ));
        }
        let input = &inputs[0];
        let input_spec = &info.inputs[0];
        let output_spec = &info.outputs[0];

        // Validate input tensor size
        let expected_input_bytes =
            input_spec.shape.iter().product::<usize>() * input_spec.precision.element_size();
        if input.data.len() != expected_input_bytes {
            // Allow FP32 input even if engine is FP16 (auto-conversion)
            let fp32_expected = input_spec.shape.iter().product::<usize>() * 4;
            if input.data.len() != fp32_expected {
                return Err(MiddlewareError::InferenceFailed(format!(
                    "Input tensor size mismatch: got {} bytes, expected {} or {} bytes",
                    input.data.len(),
                    expected_input_bytes,
                    fp32_expected
                )));
            }
        }

        let output_elems: usize = output_spec.shape.iter().product();
        let output_bytes = output_elems * output_spec.precision.element_size();

        {
            let mut output_data = vec![0u8; output_bytes];
            if let Some(ref ctx) = state.context {
                let result = if matches!(self.hardware, NvidiaHardware::Thor) {
                    // Thor: Use high-speed graph execution
                    // SAFETY: `ctx.ptr` is valid and non-null, and buffers are correctly allocated.
                    unsafe {
                        trt_infer_graph(
                            ctx.ptr,
                            input.data.as_ptr(),
                            input.data.len(),
                            output_data.as_mut_ptr(),
                            output_bytes,
                        )
                    }
                } else {
                    // Orin / Generic: Standard execution
                    // SAFETY: `ctx.ptr` is valid and non-null, and buffers are correctly allocated.
                    unsafe {
                        trt_infer(
                            ctx.ptr,
                            input.data.as_ptr(),
                            input.data.len(),
                            output_data.as_mut_ptr(),
                            output_bytes,
                        )
                    }
                };

                if result != 0 {
                    return Err(MiddlewareError::InferenceFailed(format!(
                        "TensorRT inference returned error code {}",
                        result
                    )));
                }
            } else {
                return Err(MiddlewareError::EngineNotLoaded);
            }
            Ok(vec![TensorBuffer {
                name: output_spec.name.clone(),
                data: output_data,
                shape: output_spec.shape.clone(),
                precision: Precision::FP32,
            }])
        }
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.lock();
        state.context = None; // TrtContext::drop will call trt_destroy_engine
        state.engine_info = None;
        state.buffers_allocated = false;
        info!(backend = "nvidia", "Backend released");
        Ok(())
    }

    fn build_engine(
        &self,
        onnx_path: &str,
        engine_path: &str,
        precision: Precision,
        calib_cache: Option<&str>,
    ) -> MiddlewareResult<()> {
        use std::ffi::CString;

        let c_onnx = CString::new(onnx_path)
            .map_err(|_| MiddlewareError::OptimizationFailed("Invalid ONNX path".into()))?;
        let c_engine = CString::new(engine_path)
            .map_err(|_| MiddlewareError::OptimizationFailed("Invalid engine path".into()))?;
        let c_calib = if let Some(cache) = calib_cache {
            CString::new(cache).map_err(|_| {
                MiddlewareError::OptimizationFailed("Invalid calib cache path".into())
            })?
        } else {
            CString::new("").unwrap()
        };

        // INT8 calibration cache validation
        if precision == Precision::INT8 {
            if let Some(cache_path) = calib_cache {
                if !std::path::Path::new(cache_path).exists() {
                    return Err(MiddlewareError::OptimizationFailed(format!(
                        "INT8 calibration cache file not found: {}",
                        cache_path
                    )));
                }
                info!(
                    calib_cache = %cache_path,
                    "INT8 calibration cache will be used for engine build"
                );
            } else {
                tracing::warn!(
                    "INT8 precision requested without calibration cache — \
                     TensorRT will use internal layer-wise fallback quantization"
                );
            }
        }

        // Precision mapping: 0: FP32, 1: FP16, 2: INT8, 3: FP8
        let p_int = match precision {
            Precision::FP32 => 0,
            Precision::FP16 => 1,
            Precision::INT8 => 2,
            Precision::FP8 => 3,
        };

        info!(
            onnx = %onnx_path,
            engine = %engine_path,
            precision = ?precision,
            "Building hardware-optimized engine"
        );

        // SAFETY: All pointer arguments are created from valid CStrings and are safe to pass to C.
        let success = unsafe {
            trt_build_engine(c_onnx.as_ptr(), c_engine.as_ptr(), p_int, c_calib.as_ptr())
        };

        if success {
            info!(engine = %engine_path, "Engine build complete");
            Ok(())
        } else {
            error!(onnx = %onnx_path, "Engine build failed");
            Err(MiddlewareError::Internal(
                "TensorRT engine build failed".into(),
            ))
        }
    }

    fn engine_info(&self) -> Option<EngineInfo> {
        self.state.lock().engine_info.clone()
    }

    fn is_ready(&self) -> bool {
        let state = self.state.lock();
        state.engine_info.is_some() && state.buffers_allocated
    }
}

impl Default for NvidiaAdapter {
    fn default() -> Self {
        Self::new(NvidiaHardware::Orin)
    }
}

// ---------------------------------------------------------------------------
// Helper: try to guess a shape from element count
// ---------------------------------------------------------------------------

fn infer_shape_from_elements(elems: usize) -> Vec<usize> {
    // Common ImageNet-like shapes
    if elems == 3 * 224 * 224 {
        return vec![1, 3, 224, 224];
    }
    if elems == 3 * 256 * 256 {
        return vec![1, 3, 256, 256];
    }
    if elems == 3 * 299 * 299 {
        return vec![1, 3, 299, 299];
    }
    if elems == 3 * 384 * 384 {
        return vec![1, 3, 384, 384];
    }
    if elems == 3 * 416 * 416 {
        return vec![1, 3, 416, 416];
    }
    if elems == 3 * 480 * 480 {
        return vec![1, 3, 480, 480];
    }
    if elems == 3 * 640 * 640 {
        return vec![1, 3, 640, 640];
    }
    // Generic: assume batch=1 and flatten
    vec![1, elems]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_adapter_create() {
        let a = NvidiaAdapter::new(NvidiaHardware::Orin);
        assert_eq!(a.backend_name(), "nvidia");
        assert!(!a.is_ready());
        assert_eq!(a.max_precision(), Precision::FP16);
    }

    #[test]
    fn nvidia_adapter_thor() {
        let a = NvidiaAdapter::new(NvidiaHardware::Thor);
        assert_eq!(a.max_precision(), Precision::FP8);
    }

    #[test]
    fn nvidia_adapter_default() {
        let a = NvidiaAdapter::default();
        assert_eq!(a.hardware, NvidiaHardware::Orin);
    }

    #[test]
    fn load_nonexistent_engine_fails() {
        let a = NvidiaAdapter::new(NvidiaHardware::Orin);
        assert!(a.load_engine("/does/not/exist.engine").is_err());
    }

    #[test]
    fn infer_without_load_fails() {
        let a = NvidiaAdapter::new(NvidiaHardware::Orin);
        let input = TensorBuffer::from_f32("test", &[0.0; 6], vec![1, 2, 3]);
        assert!(a.infer(&[input]).is_err());
    }
}
