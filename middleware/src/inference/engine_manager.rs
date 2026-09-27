// =============================================================================
// Magna Middleware — Engine Manager
// =============================================================================
//! Orchestrates engine loading and inference, selecting the right backend
//! adapter based on the HardwareProfile.
//!
//! When an ONNX file is supplied, the engine manager can automatically
//! optimize it for the target hardware before loading.

#[cfg(feature = "nvidia")]
use crate::backends::nvidia::adapter::NvidiaAdapter;
#[cfg(feature = "nvidia")]
use crate::backends::nvidia::NvidiaHardware;
use crate::inference::traits::{BackendCapabilities, EngineInfo, InferenceBackend, TensorBuffer};
use crate::inference::validation;
use crate::utils::errors::{MiddlewareError, MiddlewareResult};

#[cfg(not(any(
    feature = "cpu",
    feature = "nvidia",
    feature = "qualcomm",
    feature = "ti"
)))]
compile_error!("At least one backend feature must be enabled");

// Import all adapters conditionally, following the priority order
#[cfg(all(
    not(feature = "nvidia"),
    not(feature = "qualcomm"),
    not(feature = "ti"),
    feature = "cpu"
))]
use crate::backends::cpu::adapter::CpuAdapter;
#[cfg(all(not(feature = "nvidia"), feature = "qualcomm"))]
use crate::backends::qualcomm::adapter::QualcommAdapter;
#[cfg(all(not(feature = "nvidia"), not(feature = "qualcomm"), feature = "ti"))]
use crate::backends::ti::adapter::TiAdapter;

/// Select the appropriate backend adapter at compile time.
///
/// # Backend Selection Priority
///
/// When multiple backend features are enabled, they are evaluated in this order:
/// 1. **nvidia** — NVIDIA TensorRT (Orin/Thor)
/// 2. **qualcomm** — Qualcomm QNN
/// 3. **ti** — Texas Instruments TIDL
/// 4. **cpu** — CPU fallback (ONNX Runtime)
///
/// The first matching backend in the priority order is selected. For example, if both
/// `nvidia` and `cpu` features are enabled, NVIDIA will be used and CPU will be ignored.
///
/// This allows flexibility in feature configuration (e.g., keeping CPU as a fallback)
/// while ensuring deterministic backend selection at compile time.
fn selected_backend() -> Box<dyn InferenceBackend> {
    #[cfg(feature = "nvidia")]
    {
        // Determine NVIDIA hardware variant
        let hardware = if cfg!(feature = "thor") {
            NvidiaHardware::Thor
        } else if cfg!(feature = "orin") {
            NvidiaHardware::Orin
        } else {
            // Default to Orin when nvidia is enabled without sub-variant
            NvidiaHardware::Orin
        };
        Box::new(NvidiaAdapter::new(hardware))
    }

    #[cfg(all(not(feature = "nvidia"), feature = "qualcomm"))]
    {
        Box::new(QualcommAdapter::new())
    }

    #[cfg(all(not(feature = "nvidia"), not(feature = "qualcomm"), feature = "ti"))]
    {
        Box::new(TiAdapter::new())
    }

    #[cfg(all(
        not(feature = "nvidia"),
        not(feature = "qualcomm"),
        not(feature = "ti"),
        feature = "cpu"
    ))]
    {
        Box::new(CpuAdapter::new())
    }
}

/// The engine manager owns the selected backend and delegates all calls.
pub struct EngineManager {
    backend: Box<dyn InferenceBackend>,
    engine: Option<EngineInfo>,
}

impl EngineManager {
    /// Create an engine manager using the compile-time selected backend.
    pub fn new() -> MiddlewareResult<Self> {
        Ok(Self {
            backend: selected_backend(),
            engine: None,
        })
    }

    /// Load an engine file, allocate buffers, and transition to ready.
    pub fn load_engine(&mut self, path: &str) -> MiddlewareResult<EngineInfo> {
        let info = self.backend.load_engine(path)?;
        if let Err(e) = self.backend.allocate_buffers() {
            self.backend.release().ok();
            return Err(e);
        }
        self.engine = Some(info.clone());
        Ok(info)
    }

    /// Run inference on the given input tensors.
    pub fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        if !self.backend.is_ready() {
            return Err(MiddlewareError::EngineNotLoaded);
        }
        if self.backend.backend_name() == "qualcomm" {
            let engine = self
                .engine
                .as_ref()
                .ok_or(MiddlewareError::EngineNotLoaded)?;
            validation::validate_qualcomm_request(inputs, engine, &self.backend.capabilities())?;
        }
        self.backend.infer(inputs)
    }

    /// Release all backend resources.
    pub fn release(&mut self) -> MiddlewareResult<()> {
        self.backend.release()?;
        self.engine = None;
        Ok(())
    }

    /// Get the currently loaded engine info.
    pub fn engine_info(&self) -> Option<EngineInfo> {
        self.engine.clone()
    }

    /// Access the underlying backend adapter directly.
    pub fn get_backend(&self) -> &dyn InferenceBackend {
        self.backend.as_ref()
    }

    /// Get the backend name.
    pub fn backend_name(&self) -> &str {
        self.backend.backend_name()
    }

    /// Get the selected backend's explicit capabilities.
    pub fn capabilities(&self) -> BackendCapabilities {
        self.backend.capabilities()
    }

    /// Check if engine is loaded and ready for inference.
    pub fn is_ready(&self) -> bool {
        self.backend.is_ready()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::errors::Precision;

    #[test]
    fn load_and_infer_simulated() {
        let mut em = EngineManager::new().unwrap();

        let model_path = if let Ok(p) = std::env::var("TEST_ENGINE_PATH") {
            if std::path::Path::new(&p).exists() {
                p
            } else {
                eprintln!(
                    "SKIPPED [load_and_infer_simulated]: TEST_ENGINE_PATH set but not found at {}",
                    p
                );
                return;
            }
        } else if em.backend_name() == "cpu" {
            let possible_paths = [
                "../models/mobilenetv2.onnx",
                "models/mobilenetv2.onnx",
                "../../models/mobilenetv2.onnx",
            ];
            if let Some(p) = possible_paths
                .into_iter()
                .find(|p| std::path::Path::new(p).exists())
            {
                p.to_string()
            } else {
                eprintln!("SKIPPED [load_and_infer_simulated]: ONNX test model not found");
                return;
            }
        } else {
            let possible_paths = [
                "../models/mobilenetv4_nvidia_optimized_int8.engine",
                "models/mobilenetv4_nvidia_optimized_int8.engine",
                "../../models/mobilenetv4_nvidia_optimized_int8.engine",
            ];

            if let Some(p) = possible_paths
                .into_iter()
                .find(|p| std::path::Path::new(p).exists())
            {
                p.to_string()
            } else {
                eprintln!("SKIPPED [load_and_infer_simulated]: TensorRT test engine not found");
                return;
            }
        };

        let info = em.load_engine(&model_path).unwrap();
        assert!(!info.name.is_empty());

        let input = TensorBuffer {
            name: info.inputs[0].name.clone(),
            data: vec![0u8; 3 * 224 * 224 * 4],
            shape: vec![1, 3, 224, 224],
            precision: Precision::FP32,
        };
        let output = em.infer(&[input]).unwrap();
        assert_eq!(output[0].shape, vec![1, 1000]);
    }
}
