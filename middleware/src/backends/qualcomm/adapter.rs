// =============================================================================
// Magna Middleware — Qualcomm Backend Adapter
// =============================================================================
//! Implements [`InferenceBackend`] for Qualcomm Snapdragon hardware using
//! SNPE (Snapdragon Neural Processing Engine) or QNN (Qualcomm AI Engine).
//!
//! When the `qualcomm` feature is enabled and the SNPE/QNN SDK is available,
//! this adapter uses the actual hardware DSP/HTP.  Production builds do not
//! use a simulation or synthetic inference fallback; loading a model requires
//! a valid native QNN context.

use tracing::{debug, info};

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
use std::ffi::CStr;

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
use crate::inference::traits::{BackendAvailability, TensorSpec};
use crate::inference::traits::{BackendCapabilities, EngineInfo, InferenceBackend, TensorBuffer};
#[cfg(all(feature = "qualcomm", have_qnn_headers))]
use crate::utils::errors::Precision;
use crate::utils::errors::{MiddlewareError, MiddlewareResult};

// ---------------------------------------------------------------------------
// FFI declarations for QNN (linked when `qualcomm` feature and headers available)
// ---------------------------------------------------------------------------

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
extern "C" {
    fn qnn_backend_init() -> *mut std::ffi::c_void;
    fn qnn_load_context(ctx: *mut std::ffi::c_void, path: *const u8) -> bool;
    fn qnn_execute_graph(
        ctx: *mut std::ffi::c_void,
        input: *const u8,
        input_bytes: i32,
        output: *mut u8,
        output_bytes: i32,
    ) -> bool;
    fn qnn_backend_destroy(ctx: *mut std::ffi::c_void);
    fn qnn_get_num_inputs(ctx: *mut std::ffi::c_void) -> i32;
    fn qnn_get_num_outputs(ctx: *mut std::ffi::c_void) -> i32;
    fn qnn_get_input_dims(
        ctx: *mut std::ffi::c_void,
        tensor_idx: i32,
        dims_out: *mut u32,
        max_dims: i32,
    ) -> i32;
    fn qnn_get_output_dims(
        ctx: *mut std::ffi::c_void,
        tensor_idx: i32,
        dims_out: *mut u32,
        max_dims: i32,
    ) -> i32;
    fn qnn_get_input_name(ctx: *mut std::ffi::c_void, tensor_idx: i32) -> *const std::ffi::c_char;
    fn qnn_get_output_name(ctx: *mut std::ffi::c_void, tensor_idx: i32) -> *const std::ffi::c_char;
    fn qnn_get_input_precision(ctx: *mut std::ffi::c_void, tensor_idx: i32) -> i32;
    fn qnn_get_output_precision(ctx: *mut std::ffi::c_void, tensor_idx: i32) -> i32;
    fn qnn_last_error_message() -> *const std::ffi::c_char;
}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
fn precision_from_qnn(code: i32) -> Option<Precision> {
    match code {
        0 => Some(Precision::FP32),
        1 => Some(Precision::FP16),
        2 => Some(Precision::INT8),
        _ => None,
    }
}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
fn tensor_name_from_qnn(ptr: *const std::ffi::c_char, fallback: &str) -> String {
    if ptr.is_null() {
        return fallback.to_string();
    }
    // SAFETY: QNN owns this null-terminated name for the retained system-context lifetime.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
fn qnn_last_error() -> Option<String> {
    // SAFETY: qnn_last_error_message returns a thread-local, null-terminated
    // string owned by the native wrapper.
    let ptr = unsafe { qnn_last_error_message() };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: ptr is non-null and points to the wrapper's thread-local,
    // null-terminated error string. No native call invalidates it before copying.
    let msg = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    (!msg.is_empty()).then_some(msg)
}

// ---------------------------------------------------------------------------
// QualcommAdapter
// ---------------------------------------------------------------------------

/// Qualcomm backend adapter supporting SNPE and QNN.
///
/// # Scope: single input / single output only
///
/// The native QNN wrapper (`qnn_c_api.cpp`) currently supports context
/// binaries with **exactly one input and one output tensor** — this is
/// enforced by `qnn_load_context()`, which rejects
/// `numGraphInputs != 1 || numGraphOutputs != 1` and fails the load rather
/// than executing against unexpected tensor counts. `load_engine()` on a
/// multi-input/output `.bin` therefore returns
/// `MiddlewareError::EngineLoadFailed`, not incorrect inference results.
/// Extending this to multi-tensor graphs would require reworking both the
/// C++ tensor bookkeeping (`QnnContextHandle::inputs`/`outputs` arrays are
/// already `Qnn_Tensor_t*`, so the storage is there) and the Rust
/// `EngineInfo`/`TensorBuffer` plumbing (currently `infer()` binds only
/// `inputs[0]`/`outputs[0]`).
///
/// The production Qualcomm path does not contain a content-triggered
/// FAKE/DUMMY model fallback. Engine loading therefore requires a valid
/// native QNN context when the Qualcomm backend is enabled.
pub struct QualcommAdapter {
    state: parking_lot::Mutex<QualcommState>,
}

struct QualcommState {
    engine_info: Option<EngineInfo>,
    engine_path: Option<String>,
    buffers_allocated: bool,
    /// Tracks which runtime is being used.
    #[cfg(all(feature = "qualcomm", have_qnn_headers))]
    runtime: QualcommRuntime,
    #[cfg(all(feature = "qualcomm", have_qnn_headers))]
    backend_handle: Option<*mut std::ffi::c_void>,
    #[cfg(all(feature = "qualcomm", have_qnn_headers))]
    context_handle: Option<*mut std::ffi::c_void>,
}

// SAFETY: QualcommState is exclusively accessed via Mutex by the adapter.
// While it holds raw FFI pointers, access to these is safely synchronized.
// QNN/SNPE engines can be safely moved between threads and shared securely.
unsafe impl Send for QualcommState {}

// SAFETY: While QualcommState holds raw FFI pointers, Sync is safe because:
// 1. The state is never exposed directly; it is fully encapsulated in QualcommAdapter.
// 2. All access to QualcommState is mediated through a parking_lot::Mutex.
// 3. The Mutex guarantees exclusive, synchronized access preventing concurrent FFI mutation.
unsafe impl Sync for QualcommState {}

#[cfg(all(feature = "qualcomm", have_qnn_headers))]
#[derive(Debug, Clone, Copy, PartialEq)]
enum QualcommRuntime {
    None,
    Qnn,
}

impl QualcommAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::Mutex::new(QualcommState {
                engine_info: None,
                engine_path: None,
                buffers_allocated: false,
                #[cfg(all(feature = "qualcomm", have_qnn_headers))]
                runtime: QualcommRuntime::None,
                #[cfg(all(feature = "qualcomm", have_qnn_headers))]
                backend_handle: None,
                #[cfg(all(feature = "qualcomm", have_qnn_headers))]
                context_handle: None,
            }),
        }
    }

    /// Resolve QNN SDK root from environment.
    ///
    /// Checks `QAIRT_SDK_ROOT` first, falls back to `QNN_SDK_ROOT` — matching
    /// `build.rs`'s precedence so the runtime "is QNN available" check
    /// reasons about the same SDK install the native wrapper was compiled
    /// against. Returns `None` when neither is set.
    #[cfg(all(feature = "qualcomm", have_qnn_headers))]
    fn qnn_sdk_root() -> Option<String> {
        std::env::var("QAIRT_SDK_ROOT")
            .or_else(|_| std::env::var("QNN_SDK_ROOT"))
            .ok()
    }

    /// Detect which Qualcomm runtime is available (QNN preferred over SNPE).
    #[cfg(all(feature = "qualcomm", have_qnn_headers))]
    fn detect_runtime() -> QualcommRuntime {
        if Self::qnn_sdk_root().is_some() {
            info!(backend = "qualcomm", runtime = "qnn", "Runtime selected");
            return QualcommRuntime::Qnn;
        }
        debug!(
            backend = "qualcomm",
            "QNN SDK not found. Set QNN_SDK_ROOT to enable."
        );
        QualcommRuntime::None
    }
}

impl Default for QualcommAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for QualcommAdapter {
    fn backend_name(&self) -> &str {
        "qualcomm"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        info!(backend = "qualcomm", engine_path = %path, "Loading model");

        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Model file not found: {}",
                path
            )));
        }

        #[cfg(all(feature = "qualcomm", have_qnn_headers))]
        {
            let mut state = self.state.lock();
            let runtime = Self::detect_runtime();
            if let QualcommRuntime::Qnn = runtime {
                let path_cstr = std::ffi::CString::new(path).map_err(|e| {
                    MiddlewareError::EngineLoadFailed(format!("Invalid path (null byte): {e}"))
                })?;
                // SAFETY: The wrapper allocates a new opaque handle without borrowing
                // Rust data. It is checked for null before use and owned by this load.
                let backend = unsafe { qnn_backend_init() };
                if backend.is_null() {
                    let reason = qnn_last_error()
                        .unwrap_or_else(|| "Failed to initialize QNN backend".into());
                    return Err(MiddlewareError::EngineLoadFailed(reason));
                }
                // SAFETY: backend is non-null and exclusively owned by this attempt;
                // path_cstr is null-terminated and remains alive throughout the call.
                if !unsafe { qnn_load_context(backend, path_cstr.as_ptr() as *const u8) } {
                    let reason = qnn_last_error()
                        .unwrap_or_else(|| "Failed to load QNN Context using native FFI".into());
                    // SAFETY: backend belongs to this failed attempt, has not been
                    // committed to state, and is destroyed exactly once.
                    unsafe { qnn_backend_destroy(backend) };
                    return Err(MiddlewareError::EngineLoadFailed(reason));
                }

                let metadata_result: MiddlewareResult<EngineInfo> = (|| {
                    let ctx = backend;
                    let mut input_shape = Vec::new();
                    let mut output_shape = Vec::new();

                    let mut dims = [0u32; 8];

                    // SAFETY: ctx is a valid loaded handle owned by this attempt;
                    // this query only reads its tensor count.
                    let num_inputs = unsafe { qnn_get_num_inputs(ctx) };
                    // SAFETY: ctx is still valid and exclusively owned; this query
                    // only reads the output tensor count.
                    let num_outputs = unsafe { qnn_get_num_outputs(ctx) };
                    if num_inputs != 1 || num_outputs != 1 {
                        return Err(MiddlewareError::EngineLoadFailed(format!(
                            "QNN context must expose exactly one input and one output; got {num_inputs}/{num_outputs}"
                        )));
                    }

                    if num_inputs > 0 {
                        // SAFETY: ctx is a valid loaded context handle; dims is a
                        // valid 8-element stack array matching max_dims=8, and the
                        // FFI guarantees no out-of-bounds writes beyond that.
                        let rank = unsafe { qnn_get_input_dims(ctx, 0, dims.as_mut_ptr(), 8) };
                        if rank > 0 {
                            input_shape =
                                dims[..rank as usize].iter().map(|&d| d as usize).collect();
                        } else {
                            return Err(MiddlewareError::EngineLoadFailed(
                                "QNN input tensor metadata has invalid rank".into(),
                            ));
                        }
                    }

                    if num_outputs > 0 {
                        // SAFETY: same contract as the qnn_get_input_dims call above —
                        // ctx is valid and dims/max_dims bound the write.
                        let rank = unsafe { qnn_get_output_dims(ctx, 0, dims.as_mut_ptr(), 8) };
                        if rank > 0 {
                            output_shape =
                                dims[..rank as usize].iter().map(|&d| d as usize).collect();
                        } else {
                            return Err(MiddlewareError::EngineLoadFailed(
                                "QNN output tensor metadata has invalid rank".into(),
                            ));
                        }
                    }

                    // SAFETY: The retained QNN system context owns both name pointers and
                    // the queried tensor metadata for the lifetime of `ctx`.
                    let input_name =
                        tensor_name_from_qnn(unsafe { qnn_get_input_name(ctx, 0) }, "input");
                    // SAFETY: Same lifetime and bounds contract as the input query above.
                    let output_name =
                        tensor_name_from_qnn(unsafe { qnn_get_output_name(ctx, 0) }, "output");
                    // SAFETY: `ctx` is valid and the native wrapper validated tensor index 0.
                    let input_precision =
                        precision_from_qnn(unsafe { qnn_get_input_precision(ctx, 0) }).ok_or_else(
                            || {
                                MiddlewareError::EngineLoadFailed(
                                    "Unsupported QNN input datatype".into(),
                                )
                            },
                        )?;
                    // SAFETY: `ctx` is valid and the native wrapper validated tensor index 0.
                    let output_precision =
                        precision_from_qnn(unsafe { qnn_get_output_precision(ctx, 0) })
                            .ok_or_else(|| {
                                MiddlewareError::EngineLoadFailed(
                                    "Unsupported QNN output datatype".into(),
                                )
                            })?;

                    let info = EngineInfo {
                        name: std::path::Path::new(path)
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "qnn_model".into()),
                        inputs: vec![TensorSpec {
                            name: input_name,
                            shape: input_shape,
                            precision: input_precision,
                        }],
                        outputs: vec![TensorSpec {
                            name: output_name,
                            shape: output_shape,
                            precision: output_precision,
                        }],
                        memory_bytes: 0,
                    };

                    info.inputs[0].checked_byte_size()?;
                    info.outputs[0].checked_byte_size()?;
                    Ok(info)
                })();

                let info = match metadata_result {
                    Ok(info) => info,
                    Err(e) => {
                        // SAFETY: backend is the newly-created handle for this
                        // load attempt and has not been committed to state.
                        unsafe { qnn_backend_destroy(backend) };
                        return Err(e);
                    }
                };

                if let Some(old_backend) = state.backend_handle.take() {
                    // SAFETY: old_backend was previously returned by qnn_backend_init
                    // and is owned by state until this take().
                    unsafe { qnn_backend_destroy(old_backend) };
                    state.context_handle = None;
                }

                state.backend_handle = Some(backend);
                state.context_handle = Some(backend);
                state.runtime = runtime;
                state.engine_path = Some(path.to_string());
                state.buffers_allocated = false;
                state.engine_info = Some(info.clone());
                return Ok(info);
            }
        }

        #[cfg(all(feature = "qualcomm", have_qnn_headers))]
        let reason = "Qualcomm QNN SDK not available. Set QAIRT_SDK_ROOT or QNN_SDK_ROOT.";
        #[cfg(not(all(feature = "qualcomm", have_qnn_headers)))]
        let reason = "Native Qualcomm QNN support was not compiled in. Enable the qualcomm feature and rebuild with QAIRT_SDK_ROOT or QNN_SDK_ROOT pointing to the SDK headers.";
        Err(MiddlewareError::EngineLoadFailed(reason.into()))
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.lock();
        // QNN/SNPE manage their own device buffers internally.
        state.buffers_allocated = true;
        debug!(backend = "qualcomm", "Buffers allocated");
        Ok(())
    }

    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let state = self.state.lock();

        if state.engine_info.is_none() {
            return Err(MiddlewareError::EngineNotLoaded);
        }

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

        // The native QNN wrapper binds exactly one input and one output tensor
        // per graph — enforced at load time in `qnn_c_api.cpp`, which rejects
        // context binaries whose `numGraphInputs`/`numGraphOutputs` != 1. Reject
        // multi-input calls here as well, so a caller that passes extra tensors
        // gets a clear error instead of having `inputs[1..]` silently dropped.
        // See the `QualcommAdapter` docs and issue #54.
        if inputs.len() != 1 {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Qualcomm backend supports exactly one input tensor per infer() call \
                 (got {}); multi-input graphs are not supported — see QUALCOMM.md \
                 §13 and the QualcommAdapter documentation",
                inputs.len()
            )));
        }

        #[cfg(all(feature = "qualcomm", have_qnn_headers))]
        {
            let info = state.engine_info.as_ref().unwrap();
            let input = &inputs[0];
            let output_spec = &info.outputs[0];
            let output_bytes = output_spec.checked_byte_size()?;

            let input_spec = &info.inputs[0];
            input.validate_against(input_spec)?;

            if let QualcommRuntime::Qnn = state.runtime {
                let mut output_data = vec![0u8; output_bytes];

                if let Some(ctx) = state.context_handle {
                    let input_bytes = i32::try_from(input.data.len()).map_err(|_| {
                        MiddlewareError::InferenceFailed("QNN input exceeds FFI size limit".into())
                    })?;
                    let output_bytes_i32 = i32::try_from(output_bytes).map_err(|_| {
                        MiddlewareError::InferenceFailed("QNN output exceeds FFI size limit".into())
                    })?;
                    // Native FFI execution
                    // SAFETY: qnn_execute_graph is safe because:
                    // 1. ctx is a valid context handle protected by Mutex.
                    // 2. input.data and output_data are valid memory slices.
                    // 3. Both lengths are exact byte counts, checked against tensor
                    //    metadata and converted to the FFI's i32 size type.
                    // 4. Memory regions are strictly disjoint (no overlapping pointers).
                    let success = unsafe {
                        qnn_execute_graph(
                            ctx,
                            input.data.as_ptr(),
                            input_bytes,
                            output_data.as_mut_ptr(),
                            output_bytes_i32,
                        )
                    };

                    if success {
                        return Ok(vec![TensorBuffer {
                            name: output_spec.name.clone(),
                            data: output_data,
                            shape: output_spec.shape.clone(),
                            precision: output_spec.precision,
                        }]);
                    }
                }

                let reason =
                    qnn_last_error().unwrap_or_else(|| "QNN native execution failed".into());
                return Err(MiddlewareError::FfiError {
                    backend: "qualcomm".into(),
                    message: reason,
                });
            }
        }

        Err(MiddlewareError::InferenceFailed(
            "Qualcomm QNN runtime not available".into(),
        ))
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.lock();
        #[cfg(feature = "qualcomm")]
        {
            #[cfg(have_qnn_headers)]
            if let Some(backend) = state.backend_handle.take() {
                // SAFETY: backend is a valid handle extracted from state via take(), ensuring
                // exclusive ownership. It is safe to destroy and will not be double-freed.
                unsafe {
                    qnn_backend_destroy(backend);
                }
            }
            #[cfg(have_qnn_headers)]
            {
                state.context_handle = None;
            }
        }
        state.engine_info = None;
        state.engine_path = None;
        state.buffers_allocated = false;
        #[cfg(all(feature = "qualcomm", have_qnn_headers))]
        {
            state.runtime = QualcommRuntime::None;
        }
        info!(backend = "qualcomm", "Backend released");
        Ok(())
    }

    fn engine_info(&self) -> Option<EngineInfo> {
        self.state.lock().engine_info.clone()
    }

    fn is_ready(&self) -> bool {
        let state = self.state.lock();
        state.engine_info.is_some() && state.buffers_allocated
    }

    fn capabilities(&self) -> BackendCapabilities {
        #[cfg(not(feature = "qualcomm"))]
        {
            BackendCapabilities::unavailable("qualcomm", "qualcomm feature is not enabled")
        }

        #[cfg(all(feature = "qualcomm", not(have_qnn_headers)))]
        {
            BackendCapabilities::unavailable(
                "qualcomm",
                "QNN headers were not available at build time",
            )
        }

        #[cfg(all(feature = "qualcomm", have_qnn_headers))]
        {
            let runtime = Self::detect_runtime();
            let availability = if matches!(runtime, QualcommRuntime::Qnn) {
                BackendAvailability::Available
            } else {
                BackendAvailability::Unavailable {
                    reason: "QAIRT_SDK_ROOT or QNN_SDK_ROOT is not set".into(),
                }
            };
            BackendCapabilities {
                backend_name: "qualcomm".into(),
                availability,
                max_inputs: 1,
                max_outputs: 1,
                supported_input_precisions: vec![Precision::FP32, Precision::FP16, Precision::INT8],
                supported_output_precisions: vec![
                    Precision::FP32,
                    Precision::FP16,
                    Precision::INT8,
                ],
                supports_engine_building: false,
                supports_dynamic_shapes: false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualcomm_adapter_basics() {
        let a = QualcommAdapter::new();
        assert_eq!(a.backend_name(), "qualcomm");
        assert!(!a.is_ready());
    }

    #[test]
    fn qualcomm_adapter_default() {
        let a = QualcommAdapter::default();
        assert!(a.engine_info().is_none());
    }

    #[test]
    fn qualcomm_load_nonexistent_fails() {
        let a = QualcommAdapter::new();
        assert!(a.load_engine("/does/not/exist.bin").is_err());
    }

    #[test]
    fn qualcomm_infer_without_load_fails() {
        let a = QualcommAdapter::new();
        let input = TensorBuffer::from_f32("test", &[0.0; 6], vec![1, 2, 3]);
        assert!(a.infer(&[input]).is_err());
    }

    #[test]
    fn qualcomm_dummy_engine_is_rejected() {
        use std::io::Write;

        let mut f = tempfile::Builder::new().suffix(".bin").tempfile().unwrap();
        f.write_all(b"DUMMY_ENGINE_DATA").unwrap();

        let a = QualcommAdapter::new();
        let result = a.load_engine(f.path().to_str().unwrap());

        assert!(
            result.is_err(),
            "DUMMY engine must never be accepted as a real Qualcomm engine"
        );
        assert!(
            !a.is_ready(),
            "Rejected dummy engine must not leave the adapter ready"
        );
    }

    #[cfg(not(have_qnn_headers))]
    #[test]
    fn qualcomm_without_sdk_rejects_existing_models_without_changing_state() {
        use crate::inference::traits::BackendAvailability;

        let model = tempfile::NamedTempFile::new().unwrap();
        let adapter = QualcommAdapter::new();
        let error = adapter
            .load_engine(model.path().to_str().unwrap())
            .unwrap_err();
        match error {
            MiddlewareError::EngineLoadFailed(reason) => {
                assert!(reason.contains("not compiled in"), "{reason}");
            }
            other => panic!("unexpected load error: {other}"),
        }
        assert!(adapter.engine_info().is_none());
        assert!(!adapter.is_ready());
        {
            let state = adapter.state.lock();
            assert!(state.engine_path.is_none());
            assert!(!state.buffers_allocated);
        }
        assert!(matches!(
            adapter.capabilities().availability,
            BackendAvailability::Unavailable { .. }
        ));
        let input = TensorBuffer::from_f32("input", &[0.0], vec![1]);
        assert!(matches!(
            adapter.infer(&[input]),
            Err(MiddlewareError::EngineNotLoaded)
        ));
        adapter.release().unwrap();
    }
}
