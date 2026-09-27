//! Request validation helpers used before backend FFI calls.
//!
//! The first consumer is the Qualcomm QNN path, whose native wrapper currently
//! supports exactly one concrete input tensor and one output tensor.

use std::collections::HashSet;

use crate::inference::traits::{
    checked_byte_size, validate_nonzero_concrete_shape, BackendCapabilities, EngineInfo,
    TensorBuffer,
};
use crate::utils::errors::{MiddlewareError, MiddlewareResult};

#[derive(Debug, Clone, Copy)]
pub struct InferenceLimits {
    pub max_inputs: usize,
    pub max_rank: usize,
    pub max_dimension: usize,
    pub max_tensor_elements: usize,
    pub max_tensor_bytes: usize,
    pub max_total_input_bytes: usize,
}

impl InferenceLimits {
    pub const fn qualcomm_default() -> Self {
        Self {
            max_inputs: 1,
            max_rank: 8,
            max_dimension: 1_000_000,
            max_tensor_elements: 64_000_000,
            max_tensor_bytes: 512 * 1024 * 1024,
            max_total_input_bytes: 1024 * 1024 * 1024,
        }
    }
}

pub fn validate_qualcomm_request(
    inputs: &[TensorBuffer],
    engine: &EngineInfo,
    capabilities: &BackendCapabilities,
) -> MiddlewareResult<()> {
    validate_request(
        inputs,
        engine,
        capabilities,
        InferenceLimits::qualcomm_default(),
    )
}

pub fn validate_request(
    inputs: &[TensorBuffer],
    engine: &EngineInfo,
    capabilities: &BackendCapabilities,
    limits: InferenceLimits,
) -> MiddlewareResult<()> {
    if inputs.is_empty() {
        return Err(MiddlewareError::InferenceFailed(
            "Inference request contains no input tensors".into(),
        ));
    }
    if inputs.len() > limits.max_inputs {
        return Err(MiddlewareError::InferenceFailed(format!(
            "Inference request has {} input tensors but the configured limit is {}",
            inputs.len(),
            limits.max_inputs
        )));
    }
    if inputs.len() > capabilities.max_inputs {
        return Err(MiddlewareError::InferenceFailed(format!(
            "Backend '{}' supports at most {} input tensor(s), got {}",
            capabilities.backend_name,
            capabilities.max_inputs,
            inputs.len()
        )));
    }
    if engine.inputs.len() > capabilities.max_inputs
        || engine.outputs.len() > capabilities.max_outputs
    {
        return Err(MiddlewareError::InferenceFailed(format!(
            "Loaded engine '{}' has {} input(s)/{} output(s), but backend '{}' supports {}/{}",
            engine.name,
            engine.inputs.len(),
            engine.outputs.len(),
            capabilities.backend_name,
            capabilities.max_inputs,
            capabilities.max_outputs
        )));
    }
    if inputs.len() != engine.inputs.len() {
        return Err(MiddlewareError::InferenceFailed(format!(
            "Inference request has {} input tensor(s), expected {}",
            inputs.len(),
            engine.inputs.len()
        )));
    }

    let mut seen = HashSet::new();
    let mut total_bytes = 0usize;
    for input in inputs {
        if input.name.is_empty() {
            return Err(MiddlewareError::InferenceFailed(
                "Inference request contains an input tensor with an empty name".into(),
            ));
        }
        if !seen.insert(input.name.as_str()) {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Duplicate input tensor name '{}'",
                input.name
            )));
        }

        validate_nonzero_concrete_shape(&input.name, &input.shape)?;
        if input.shape.len() > limits.max_rank {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor '{}' rank {} exceeds limit {}",
                input.name,
                input.shape.len(),
                limits.max_rank
            )));
        }
        if let Some((idx, dim)) = input
            .shape
            .iter()
            .enumerate()
            .find(|(_, dim)| **dim > limits.max_dimension)
        {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor '{}' dimension {} value {} exceeds limit {}",
                input.name, idx, dim, limits.max_dimension
            )));
        }

        let elements = input.checked_num_elements()?;
        if elements > limits.max_tensor_elements {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor '{}' element count {} exceeds limit {}",
                input.name, elements, limits.max_tensor_elements
            )));
        }
        let bytes = input.checked_byte_size()?;
        if bytes > limits.max_tensor_bytes {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Tensor '{}' byte size {} exceeds limit {}",
                input.name, bytes, limits.max_tensor_bytes
            )));
        }
        total_bytes = total_bytes.checked_add(bytes).ok_or_else(|| {
            MiddlewareError::InferenceFailed("Total input byte count overflows".into())
        })?;
        if total_bytes > limits.max_total_input_bytes {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Total input byte size {} exceeds limit {}",
                total_bytes, limits.max_total_input_bytes
            )));
        }

        let Some(spec) = engine.inputs.iter().find(|spec| spec.name == input.name) else {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Unknown input tensor '{}' for engine '{}'",
                input.name, engine.name
            )));
        };
        input.validate_against(spec)?;
    }

    for output in &engine.outputs {
        let bytes = checked_byte_size(&output.name, &output.shape, output.precision)?;
        if output.shape.len() > limits.max_rank {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Output tensor '{}' rank {} exceeds limit {}",
                output.name,
                output.shape.len(),
                limits.max_rank
            )));
        }
        if bytes > limits.max_tensor_bytes {
            return Err(MiddlewareError::InferenceFailed(format!(
                "Output tensor '{}' byte size {} exceeds limit {}",
                output.name, bytes, limits.max_tensor_bytes
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::traits::{
        BackendAvailability, BackendCapabilities, EngineInfo, TensorSpec,
    };
    use crate::utils::errors::Precision;

    fn caps() -> BackendCapabilities {
        BackendCapabilities {
            backend_name: "qualcomm".into(),
            availability: BackendAvailability::Available,
            max_inputs: 1,
            max_outputs: 1,
            supported_input_precisions: vec![Precision::FP32, Precision::FP16, Precision::INT8],
            supported_output_precisions: vec![Precision::FP32, Precision::FP16, Precision::INT8],
            supports_engine_building: false,
            supports_dynamic_shapes: false,
        }
    }

    fn engine() -> EngineInfo {
        EngineInfo {
            name: "model".into(),
            inputs: vec![TensorSpec {
                name: "input".into(),
                shape: vec![1, 2],
                precision: Precision::FP32,
            }],
            outputs: vec![TensorSpec {
                name: "output".into(),
                shape: vec![1, 1],
                precision: Precision::FP32,
            }],
            memory_bytes: 0,
        }
    }

    #[test]
    fn valid_qualcomm_request_passes() {
        let input = TensorBuffer {
            name: "input".into(),
            data: vec![0u8; 8],
            shape: vec![1, 2],
            precision: Precision::FP32,
        };
        assert!(validate_qualcomm_request(&[input], &engine(), &caps()).is_ok());
    }

    #[test]
    fn duplicate_name_fails() {
        let input = TensorBuffer {
            name: "input".into(),
            data: vec![0u8; 8],
            shape: vec![1, 2],
            precision: Precision::FP32,
        };
        let mut engine = engine();
        engine.inputs.push(engine.inputs[0].clone());
        assert!(validate_request(
            &[input.clone(), input],
            &engine,
            &BackendCapabilities {
                max_inputs: 2,
                ..caps()
            },
            InferenceLimits {
                max_inputs: 2,
                ..InferenceLimits::qualcomm_default()
            },
        )
        .is_err());
    }

    #[test]
    fn byte_mismatch_fails() {
        let input = TensorBuffer {
            name: "input".into(),
            data: vec![0u8; 4],
            shape: vec![1, 2],
            precision: Precision::FP32,
        };
        assert!(validate_qualcomm_request(&[input], &engine(), &caps()).is_err());
    }
}
