// =============================================================================
// Magna Middleware — CPU Fallback Backend
// =============================================================================
//! A simple CPU-based inference backend that serves as the universal fallback.
//!
//! This uses ONNX Runtime CPU EP via the `ort` crate to execute inference.

use tracing::{debug, info};

use crate::inference::traits::{EngineInfo, InferenceBackend, TensorBuffer, TensorSpec};
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};
use ort::session::Session;
use ort::value::Tensor;

pub struct CpuAdapter {
    state: parking_lot::RwLock<CpuState>,
}

struct CpuState {
    engine_info: Option<EngineInfo>,
    buffers_allocated: bool,
    session: Option<Session>,
}

impl CpuAdapter {
    pub fn new() -> Self {
        Self {
            state: parking_lot::RwLock::new(CpuState {
                engine_info: None,
                buffers_allocated: false,
                session: None,
            }),
        }
    }
}

impl Default for CpuAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CpuAdapter {
    fn backend_name(&self) -> &str {
        "cpu"
    }

    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        info!(backend = "cpu", engine_path = %path, "Loading model via ONNX Runtime");

        if !std::path::Path::new(path).exists() {
            return Err(MiddlewareError::EngineLoadFailed(format!(
                "Model file not found: {}",
                path
            )));
        }

        let session = Session::builder()
            .map_err(|e| {
                MiddlewareError::EngineLoadFailed(format!("Failed to create SessionBuilder: {}", e))
            })?
            .commit_from_file(path)
            .map_err(|e| {
                MiddlewareError::EngineLoadFailed(format!("Failed to load ONNX model: {}", e))
            })?;

        let input_specs = session
            .inputs()
            .iter()
            .map(|input| {
                let name = input.name().to_string();
                let shape = match input.dtype().tensor_shape() {
                    Some(s) => s
                        .iter()
                        .map(|&d| if d <= 0 { 0 } else { d as usize })
                        .collect(),
                    None => vec![],
                };
                TensorSpec {
                    name,
                    shape,
                    precision: Precision::FP32,
                }
            })
            .collect::<Vec<_>>();

        let output_specs = session
            .outputs()
            .iter()
            .map(|output| {
                let name = output.name().to_string();
                let shape = match output.dtype().tensor_shape() {
                    Some(s) => s
                        .iter()
                        .map(|&d| if d <= 0 { 0 } else { d as usize })
                        .collect(),
                    None => vec![],
                };
                TensorSpec {
                    name,
                    shape,
                    precision: Precision::FP32,
                }
            })
            .collect::<Vec<_>>();

        let info = EngineInfo {
            name: std::path::Path::new(path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "cpu_model".into()),
            inputs: input_specs,
            outputs: output_specs,
            memory_bytes: 0,
        };

        let mut state = self.state.write();
        state.engine_info = Some(info.clone());
        state.session = Some(session);
        info!(backend = "cpu", engine = %info.name, "Model loaded successfully");
        Ok(info)
    }

    fn allocate_buffers(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        state.buffers_allocated = true;
        debug!(backend = "cpu", "Buffers allocated (no-op for CPU)");
        Ok(())
    }

    /// Run inference via ONNX Runtime.
    ///
    /// **NOTE:** Unlike NVIDIA/TI/Qualcomm backends, ONNX Runtime validates tensor
    /// names. The `input.name` field MUST match the model's declared input binding
    /// (e.g. `"data"` for MobileNetV2-7). A mismatch causes a runtime error.
    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let mut state = self.state.write();
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

        let session = state
            .session
            .as_mut()
            .ok_or(MiddlewareError::EngineNotLoaded)?;

        let mut session_inputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            input.validate().map_err(|e| {
                MiddlewareError::InferenceFailed(format!("Input validation: {}", e))
            })?;

            let value = match input.precision {
                Precision::FP32 => {
                    let f32_data: &[f32] = bytemuck::cast_slice(&input.data);
                    let vec_data = f32_data.to_vec();
                    Tensor::from_array((input.shape.clone(), vec_data))
                        .map_err(|e| {
                            MiddlewareError::InferenceFailed(format!(
                                "Failed to create input tensor: {}",
                                e
                            ))
                        })?
                        .into_dyn()
                }
                _ => {
                    return Err(MiddlewareError::InferenceFailed(format!(
                        "Unsupported precision for CPU fallback: {:?}",
                        input.precision
                    )));
                }
            };
            session_inputs.push((input.name.as_str(), value));
        }

        let outputs = session
            .run(session_inputs)
            .map_err(|e| MiddlewareError::InferenceFailed(format!("Inference failed: {}", e)))?;

        let mut result_buffers = Vec::with_capacity(outputs.len());
        for (name, value) in outputs {
            let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| {
                MiddlewareError::InferenceFailed(format!("Failed to extract output tensor: {}", e))
            })?;

            let u8_data = bytemuck::cast_slice(data).to_vec();
            let shape_vec = shape.iter().map(|&d| d as usize).collect::<Vec<usize>>();

            result_buffers.push(TensorBuffer {
                name: name.to_string(),
                data: u8_data,
                shape: shape_vec,
                precision: Precision::FP32,
            });
        }

        debug!(
            backend = "cpu",
            outputs_count = result_buffers.len(),
            "Inference complete"
        );
        Ok(result_buffers)
    }

    fn release(&self) -> MiddlewareResult<()> {
        let mut state = self.state.write();
        state.engine_info = None;
        state.session = None;
        state.buffers_allocated = false;
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_adapter_basics() {
        let a = CpuAdapter::new();
        assert_eq!(a.backend_name(), "cpu");
        assert!(!a.is_ready());
    }

    #[test]
    fn cpu_adapter_default() {
        let a = CpuAdapter::default();
        assert!(a.engine_info().is_none());
    }

    #[test]
    fn cpu_load_nonexistent_fails() {
        let a = CpuAdapter::new();
        assert!(a.load_engine("/nonexistent/model.onnx").is_err());
    }
}
