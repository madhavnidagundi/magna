// =============================================================================
// Magna Middleware — Inference Backend Trait
// =============================================================================
//! Defines the core abstraction that every backend adapter must implement.
//!
//! The middleware never calls vendor-specific APIs directly.  All interaction
//! goes through [`InferenceBackend`].  This trait is object-safe so that the
//! engine manager can hold `Box<dyn InferenceBackend>`.

use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

// ---------------------------------------------------------------------------
// Backend capabilities
// ---------------------------------------------------------------------------

/// Runtime availability reported by a backend implementation.
///
/// This describes implemented behavior in the current process, not theoretical
/// vendor support on the target platform.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum BackendAvailability {
    Available,
    Unavailable { reason: String },
    Stub { reason: String },
}

/// Explicit contract exposed by a backend adapter.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend_name: String,
    pub availability: BackendAvailability,
    pub max_inputs: usize,
    pub max_outputs: usize,
    pub supported_input_precisions: Vec<Precision>,
    pub supported_output_precisions: Vec<Precision>,
    pub supports_engine_building: bool,
    pub supports_dynamic_shapes: bool,
}

impl BackendCapabilities {
    pub fn unavailable(backend_name: &str, reason: impl Into<String>) -> Self {
        Self {
            backend_name: backend_name.to_string(),
            availability: BackendAvailability::Unavailable {
                reason: reason.into(),
            },
            max_inputs: 0,
            max_outputs: 0,
            supported_input_precisions: Vec::new(),
            supported_output_precisions: Vec::new(),
            supports_engine_building: false,
            supports_dynamic_shapes: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor descriptor
// ---------------------------------------------------------------------------

/// Lightweight descriptor for a single tensor binding inside a compiled engine.
///
/// `precision` is populated by the backend adapter when it introspects the
/// loaded engine (e.g. via `getTensorDataType` in TensorRT, or QNN context
/// query). Callers must **not** assume a default — always read from the engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    /// Native element precision reported by the compiled engine.
    /// Defaults to FP32 when the backend cannot introspect (e.g. CPU/simulated).
    pub precision: Precision,
}

impl TensorSpec {
    /// Number of elements implied by this tensor spec, using checked arithmetic.
    pub fn checked_num_elements(&self) -> MiddlewareResult<usize> {
        checked_num_elements(&self.name, &self.shape)
    }

    /// Expected byte count for this tensor spec, using checked arithmetic.
    pub fn checked_byte_size(&self) -> MiddlewareResult<usize> {
        checked_byte_size(&self.name, &self.shape, self.precision)
    }
}

/// Lightweight descriptor for a contiguous, row-major tensor buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorBuffer {
    pub name: String,
    /// Raw bytes of the tensor (owned).
    pub data: Vec<u8>,
    /// Shape — e.g. `[1, 3, 224, 224]` for a single RGB image.
    pub shape: Vec<usize>,
    /// Element precision.
    pub precision: Precision,
}

impl TensorBuffer {
    /// Number of elements implied by shape.
    pub fn num_elements(&self) -> usize {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape.iter().product()
    }

    /// Size of one element in bytes for the current precision.
    pub fn element_size(&self) -> usize {
        self.precision.element_size()
    }

    /// Expected total byte count.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.element_size()
    }

    /// Number of elements implied by shape, using checked arithmetic.
    pub fn checked_num_elements(&self) -> MiddlewareResult<usize> {
        checked_num_elements(&self.name, &self.shape)
    }

    /// Expected total byte count, using checked arithmetic.
    pub fn checked_byte_size(&self) -> MiddlewareResult<usize> {
        checked_byte_size(&self.name, &self.shape, self.precision)
    }

    /// Create a new tensor buffer from f32 data.
    pub fn from_f32(name: &str, data: &[f32], shape: Vec<usize>) -> Self {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            name: name.to_string(),
            data: bytes,
            shape,
            precision: Precision::FP32,
        }
    }

    /// Try to interpret as f32 slice, returning Result instead of panicking.
    pub fn try_as_f32_slice(&self) -> MiddlewareResult<&[f32]> {
        if self.precision != Precision::FP32 {
            return Err(MiddlewareError::PostprocessingFailed(format!(
                "Expected FP32 tensor, got {:?}",
                self.precision
            )));
        }
        if !self.data.len().is_multiple_of(4) {
            return Err(MiddlewareError::PostprocessingFailed(format!(
                "Byte length {} is not a multiple of 4",
                self.data.len()
            )));
        }
        Ok(bytemuck::cast_slice(&self.data))
    }

    /// Validate that this tensor's byte count matches shape × element_size.
    pub fn validate(&self) -> MiddlewareResult<()> {
        let expected = self.checked_byte_size()?;
        if self.data.len() != expected {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor size mismatch: data has {} bytes but shape {:?} × {:?} expects {} bytes",
                self.data.len(),
                self.shape,
                self.precision,
                expected,
            )));
        }
        Ok(())
    }

    /// Validate this concrete tensor against loaded engine metadata.
    pub fn validate_against(&self, spec: &TensorSpec) -> MiddlewareResult<()> {
        validate_nonzero_concrete_shape(&self.name, &self.shape)?;

        if self.name != spec.name {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor name mismatch: got '{}', expected '{}'",
                self.name, spec.name
            )));
        }
        if self.shape != spec.shape {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor shape mismatch for '{}': got {:?}, expected {:?}",
                self.name, self.shape, spec.shape
            )));
        }
        if self.precision != spec.precision {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor precision mismatch for '{}': got {:?}, expected {:?}",
                self.name, self.precision, spec.precision
            )));
        }
        self.validate()
    }
}

pub fn validate_nonzero_concrete_shape(name: &str, shape: &[usize]) -> MiddlewareResult<()> {
    if shape.is_empty() {
        return Err(MiddlewareError::InferenceFailed(format!(
            "Tensor '{}' has empty shape; scalar tensors are not supported by this path",
            name
        )));
    }
    if let Some((idx, _)) = shape.iter().enumerate().find(|(_, dim)| **dim == 0) {
        return Err(MiddlewareError::InferenceFailed(format!(
            "Tensor '{}' has zero dimension at index {}",
            name, idx
        )));
    }
    Ok(())
}

pub fn checked_num_elements(name: &str, shape: &[usize]) -> MiddlewareResult<usize> {
    validate_nonzero_concrete_shape(name, shape)?;
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            MiddlewareError::InferenceFailed(format!(
                "Tensor '{}' element count overflows for shape {:?}",
                name, shape
            ))
        })
    })
}

pub fn checked_byte_size(
    name: &str,
    shape: &[usize],
    precision: Precision,
) -> MiddlewareResult<usize> {
    checked_num_elements(name, shape)?
        .checked_mul(precision.element_size())
        .ok_or_else(|| {
            MiddlewareError::InferenceFailed(format!(
                "Tensor '{}' byte size overflows for shape {:?} and precision {:?}",
                name, shape, precision
            ))
        })
}

// ---------------------------------------------------------------------------
// Engine metadata
// ---------------------------------------------------------------------------

/// Metadata describing a loaded engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EngineInfo {
    /// Human-readable engine name or file stem.
    pub name: String,
    /// Input descriptors.
    pub inputs: Vec<TensorSpec>,
    /// Output descriptors.
    pub outputs: Vec<TensorSpec>,
    /// Estimated GPU/accelerator memory usage in bytes.
    pub memory_bytes: u64,
}

// ---------------------------------------------------------------------------
// InferenceBackend trait
// ---------------------------------------------------------------------------

/// Core trait that every backend adapter implements.
///
/// # Lifecycle
/// 1. `load_engine` — load the precompiled artifact from disk.
/// 2. `allocate_buffers` — pre-allocate device-side I/O buffers.
/// 3. `infer` — run a single forward pass (may be called many times).
/// 4. `release` — tear down all resources.
pub trait InferenceBackend: Send + Sync {
    /// Human-readable backend identifier (e.g. `"nvidia"`, `"qualcomm"`).
    fn backend_name(&self) -> &str;

    /// Load a precompiled engine file from the given path.
    fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo>;

    /// Pre-allocate input/output buffers on the device.
    fn allocate_buffers(&self) -> MiddlewareResult<()>;

    /// Execute a single inference pass.
    ///
    /// The implementation copies `inputs` to device memory, runs the engine,
    /// and copies the `outputs` back.
    fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>>;

    /// Release all resources (engine handle, device memory, etc.).
    fn release(&self) -> MiddlewareResult<()>;

    /// Optional: Build a hardware-optimized engine from an ONNX model.
    fn build_engine(
        &self,
        _onnx_path: &str,
        _engine_path: &str,
        _precision: Precision,
        _calib_cache: Option<&str>,
    ) -> MiddlewareResult<()> {
        Err(MiddlewareError::NotSupported(
            "Building engines is not supported for this backend".into(),
        ))
    }

    /// Return metadata about the currently loaded engine, or `None`.
    fn engine_info(&self) -> Option<EngineInfo>;

    /// Whether an engine is currently loaded and buffers are allocated.
    fn is_ready(&self) -> bool;

    /// Explicit backend behavior and availability.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::unavailable(
            self.backend_name(),
            "backend-specific capabilities are not implemented",
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_buffer_from_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let buf = TensorBuffer::from_f32("input", &data, vec![1, 2, 3]);
        assert_eq!(buf.num_elements(), 6);
        assert_eq!(buf.element_size(), 4);
        assert_eq!(buf.byte_size(), 24);

        let slice = buf.try_as_f32_slice().unwrap();
        assert_eq!(slice.len(), 6);
        assert!((slice[0] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn tensor_shape_calculation() {
        let buf = TensorBuffer {
            name: "input".to_string(),
            data: vec![0u8; 3 * 224 * 224 * 4],
            shape: vec![1, 3, 224, 224],
            precision: Precision::FP32,
        };
        assert_eq!(buf.num_elements(), 150_528);
        assert_eq!(buf.byte_size(), 150_528 * 4);
    }

    #[test]
    fn tensor_validation() {
        let valid = TensorBuffer {
            name: "test".to_string(),
            data: vec![0u8; 24],
            shape: vec![1, 2, 3],
            precision: Precision::FP32,
        };
        assert!(valid.validate().is_ok());

        let invalid = TensorBuffer {
            name: "test".to_string(),
            data: vec![0u8; 10], // wrong size
            shape: vec![1, 2, 3],
            precision: Precision::FP32,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn checked_tensor_size_overflow_fails() {
        let buf = TensorBuffer {
            name: "huge".to_string(),
            data: vec![],
            shape: vec![usize::MAX, 2],
            precision: Precision::FP32,
        };
        assert!(buf.checked_byte_size().is_err());
        assert!(buf.validate().is_err());
    }

    #[test]
    fn zero_dimension_fails_validation() {
        let buf = TensorBuffer {
            name: "bad".to_string(),
            data: vec![],
            shape: vec![1, 0, 224, 224],
            precision: Precision::FP32,
        };
        assert!(buf.validate().is_err());
    }

    #[test]
    fn validate_against_spec_rejects_mismatch() {
        let spec = TensorSpec {
            name: "input".to_string(),
            shape: vec![1, 2],
            precision: Precision::FP32,
        };
        let bad = TensorBuffer {
            name: "other".to_string(),
            data: vec![0u8; 8],
            shape: vec![1, 2],
            precision: Precision::FP32,
        };
        assert!(bad.validate_against(&spec).is_err());
    }

    #[test]
    fn try_as_f32_non_fp32_fails() {
        let buf = TensorBuffer {
            name: "test".to_string(),
            data: vec![0u8; 4],
            shape: vec![1, 4],
            precision: Precision::INT8,
        };
        assert!(buf.try_as_f32_slice().is_err());
    }

    #[test]
    fn empty_shape_is_zero_elements() {
        let buf = TensorBuffer {
            name: "test".to_string(),
            data: vec![],
            shape: vec![],
            precision: Precision::FP32,
        };
        assert_eq!(buf.num_elements(), 0);
        assert_eq!(buf.byte_size(), 0);
    }
}
