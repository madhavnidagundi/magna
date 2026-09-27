// =============================================================================
// Magna Middleware — Buffer Manager
// =============================================================================
//! Manages pre-allocated host-side tensor buffers to avoid heap allocations
//! on the inference hot path.

use crate::inference::traits::{EngineInfo, TensorBuffer};
use crate::utils::errors::{MiddlewareError, MiddlewareResult};

/// Manages pre-allocated input/output buffers.
#[derive(Debug)]
pub struct BufferManager {
    input_buffer: Option<TensorBuffer>,
    output_buffer: Option<TensorBuffer>,
}

impl BufferManager {
    /// Create a new, empty buffer manager.
    pub fn new() -> Self {
        Self {
            input_buffer: None,
            output_buffer: None,
        }
    }

    /// Pre-allocate buffers based on the engine info.
    pub fn allocate(&mut self, info: &EngineInfo) -> MiddlewareResult<()> {
        let input_spec = info
            .inputs
            .first()
            .ok_or_else(|| MiddlewareError::InferenceFailed("Engine info has no inputs".into()))?;
        let output_spec = info
            .outputs
            .first()
            .ok_or_else(|| MiddlewareError::InferenceFailed("Engine info has no outputs".into()))?;

        // Read precision from the tensor spec (engine-reported) rather than
        // the top-level EngineInfo.precision, so each tensor's buffer is sized
        // correctly even when input and output precisions differ.
        let input_elems: usize = input_spec.shape.iter().product();
        let output_elems: usize = output_spec.shape.iter().product();
        let input_elem_size = input_spec.precision.element_size();
        let output_elem_size = output_spec.precision.element_size();

        self.input_buffer = Some(TensorBuffer {
            name: input_spec.name.clone(),
            data: vec![0u8; input_elems * input_elem_size],
            shape: input_spec.shape.clone(),
            precision: input_spec.precision,
        });

        self.output_buffer = Some(TensorBuffer {
            name: output_spec.name.clone(),
            data: vec![0u8; output_elems * output_elem_size],
            shape: output_spec.shape.clone(),
            precision: output_spec.precision,
        });

        Ok(())
    }

    /// Get a mutable reference to the pre-allocated input buffer.
    pub fn input_buffer_mut(&mut self) -> Option<&mut TensorBuffer> {
        self.input_buffer.as_mut()
    }

    /// Get a reference to the pre-allocated output buffer.
    pub fn output_buffer(&self) -> Option<&TensorBuffer> {
        self.output_buffer.as_ref()
    }

    /// Release all buffers.
    pub fn release(&mut self) {
        self.input_buffer = None;
        self.output_buffer = None;
    }

    /// Whether buffers have been allocated.
    pub fn is_allocated(&self) -> bool {
        self.input_buffer.is_some() && self.output_buffer.is_some()
    }
}

impl Default for BufferManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::errors::Precision;

    fn sample_engine_info() -> EngineInfo {
        EngineInfo {
            name: "resnet50".into(),
            inputs: vec![crate::inference::traits::TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 224, 224],
                precision: Precision::FP32,
            }],
            outputs: vec![crate::inference::traits::TensorSpec {
                name: "output".to_string(),
                shape: vec![1, 1000],
                precision: Precision::FP32,
            }],
            memory_bytes: 50 * 1024 * 1024,
        }
    }

    #[test]
    fn allocate_buffers() {
        let mut bm = BufferManager::new();
        assert!(!bm.is_allocated());

        bm.allocate(&sample_engine_info()).unwrap();
        assert!(bm.is_allocated());

        let input = bm.input_buffer_mut().unwrap();
        assert_eq!(input.shape, vec![1, 3, 224, 224]);
        assert_eq!(input.data.len(), 3 * 224 * 224 * 4);
    }

    #[test]
    fn release_buffers() {
        let mut bm = BufferManager::new();
        bm.allocate(&sample_engine_info()).unwrap();
        assert!(bm.is_allocated());

        bm.release();
        assert!(!bm.is_allocated());
    }

    #[test]
    fn fp16_buffer_sizes() {
        let mut bm = BufferManager::new();
        let info = EngineInfo {
            name: "model".into(),
            inputs: vec![crate::inference::traits::TensorSpec {
                name: "input".to_string(),
                shape: vec![1, 3, 224, 224],
                precision: Precision::FP16,
            }],
            outputs: vec![crate::inference::traits::TensorSpec {
                name: "output".to_string(),
                shape: vec![1, 1000],
                precision: Precision::FP16,
            }],
            memory_bytes: 0,
        };
        bm.allocate(&info).unwrap();
        let input = bm.input_buffer_mut().unwrap();
        assert_eq!(input.data.len(), 3 * 224 * 224 * 2); // FP16 = 2 bytes
    }
}
