// =============================================================================
// Magna Middleware — Structured Error Types
// =============================================================================
//! Centralized error definitions for all middleware modules.
//!
//! Every module returns [`MiddlewareError`] so call sites get one uniform type.
//! Internal modules can create domain-specific variants (e.g. `Preprocessing`,
//! `Backend`) while keeping a single error enum at the crate boundary.

use std::fmt;
use std::str::FromStr;

// ---------------------------------------------------------------------------
// Primary error enum
// ---------------------------------------------------------------------------

/// Top-level error type for the Magna middleware.
#[derive(Debug, thiserror::Error)]
pub enum MiddlewareError {
    // -- lifecycle / state --------------------------------------------------
    #[error("Invalid state transition: cannot move from {from} to {to}")]
    InvalidStateTransition { from: String, to: String },

    #[error("Operation requires state '{required}', but current state is '{current}'")]
    InvalidState { required: String, current: String },

    // -- engine / inference -------------------------------------------------
    #[error("Engine load failed: {0}")]
    EngineLoadFailed(String),

    #[error("Inference execution failed: {0}")]
    InferenceFailed(String),

    #[error("Buffer allocation failed: {0}")]
    BufferAllocationFailed(String),

    #[error("Engine not loaded")]
    EngineNotLoaded,

    // -- hardware -----------------------------------------------------------
    #[error("Hardware detection failed: {0}")]
    HardwareDetectionFailed(String),

    #[error("Unsupported hardware: {0}")]
    UnsupportedHardware(String),

    #[error("Backend '{0}' is not available on this platform")]
    BackendUnavailable(String),

    // -- preprocessing ------------------------------------------------------
    #[error("Preprocessing failed: {0}")]
    PreprocessingFailed(String),

    #[error("Image load failed: {0}")]
    ImageLoadFailed(String),

    // -- postprocessing -----------------------------------------------------
    #[error("Postprocessing failed: {0}")]
    PostprocessingFailed(String),

    // -- model loading ------------------------------------------------------
    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    // -- config -------------------------------------------------------------
    #[error("Configuration error: {0}")]
    ConfigError(String),

    // -- I/O ----------------------------------------------------------------
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Input source exhausted or disconnected: {0}")]
    InputExhausted(String),

    // -- FFI / backend ------------------------------------------------------
    #[error("FFI error in backend '{backend}': {message}")]
    FfiError { backend: String, message: String },

    // -- optimization -------------------------------------------------------
    #[error("Model optimization failed: {0}")]
    OptimizationFailed(String),

    // -- dependency ---------------------------------------------------------
    #[error("Dependency missing: {0}")]
    DependencyMissing(String),

    // -- catch-all ----------------------------------------------------------
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not supported: {0}")]
    NotSupported(String),
}

/// Convenience alias used throughout the crate.
pub type MiddlewareResult<T> = Result<T, MiddlewareError>;

// ---------------------------------------------------------------------------
// Precision mode
// ---------------------------------------------------------------------------

/// Numeric precision modes supported by backend runtimes.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    Default,
    clap::ValueEnum,
)]
pub enum Precision {
    #[default]
    FP32,
    FP16,
    INT8,
    FP8,
}

impl Precision {
    /// Size of one element in bytes for this precision.
    pub fn element_size(self) -> usize {
        match self {
            Precision::FP32 => 4,
            Precision::FP16 => 2,
            Precision::INT8 => 1,
            Precision::FP8 => 1,
        }
    }

    /// Map TensorRT element byte-size to a Precision variant.
    pub fn from_elem_size(bytes: i32) -> Self {
        match bytes {
            1 => Precision::INT8,
            2 => Precision::FP16,
            4 => Precision::FP32,
            _ => Precision::FP32, // safe default
        }
    }

    /// Human-readable short name (lowercase) for filenames and CLI args.
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::FP32 => "fp32",
            Precision::FP16 => "fp16",
            Precision::INT8 => "int8",
            Precision::FP8 => "fp8",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::FP32 => write!(f, "FP32"),
            Precision::FP16 => write!(f, "FP16"),
            Precision::INT8 => write!(f, "INT8"),
            Precision::FP8 => write!(f, "FP8"),
        }
    }
}

impl FromStr for Precision {
    type Err = MiddlewareError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fp32" | "float32" => Ok(Precision::FP32),
            "fp16" | "float16" | "half" => Ok(Precision::FP16),
            "int8" | "i8" => Ok(Precision::INT8),
            "fp8" | "float8" => Ok(Precision::FP8),
            other => Err(MiddlewareError::ConfigError(format!(
                "Unknown precision '{}'. Expected: fp32, fp16, int8, fp8",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Middleware configuration
// ---------------------------------------------------------------------------

/// Top-level configuration passed to `initialize()`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MiddlewareConfig {
    /// Fallback precision used before an engine is loaded.
    /// Backends that CAN introspect (TensorRT) ignore this after engine load.
    #[serde(default)]
    pub fallback_precision: Precision,

    /// Number of warmup runs before timing begins.
    #[serde(default = "default_warmup")]
    pub warmup_runs: usize,

    /// Enable verbose / debug logging.
    #[serde(default)]
    pub debug: bool,

    /// Optional path to class-label file (one label per line).
    pub labels_path: Option<String>,

    /// Optional path to the ONNX / engine model file to load.
    pub model_path: Option<String>,

    /// gRPC server listen address (e.g. 127.0.0.1:50051).
    #[serde(default = "default_grpc_address")]
    pub grpc_address: String,
}

fn default_warmup() -> usize {
    3
}

fn default_grpc_address() -> String {
    "127.0.0.1:50051".into()
}

impl Default for MiddlewareConfig {
    fn default() -> Self {
        Self {
            fallback_precision: Precision::default(),
            warmup_runs: default_warmup(),
            debug: false,
            labels_path: None,
            model_path: None,
            grpc_address: default_grpc_address(),
        }
    }
}

// ---------------------------------------------------------------------------
// Model metadata (dynamic, from ONNX inspection)
// ---------------------------------------------------------------------------

/// Dynamically extracted model metadata — no hardcoded shapes or assumptions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    /// Model file name or identifier.
    pub name: String,
    /// Input tensor names and their shapes (may contain -1 for dynamic dims).
    pub inputs: Vec<TensorMeta>,
    /// Output tensor names and their shapes.
    pub outputs: Vec<TensorMeta>,
    /// Total number of model parameters (0 if unknown).
    pub num_parameters: u64,
}

/// Metadata for a single tensor (input or output).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorMeta {
    /// Tensor name from the ONNX graph.
    pub name: String,
    /// Shape dimensions. -1 means dynamic.
    pub shape: Vec<i64>,
    /// Element data type string (e.g. "float32", "int64").
    pub dtype: String,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "unknown".into(),
            inputs: vec![],
            outputs: vec![],
            num_parameters: 0,
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
    fn error_display_messages() {
        let e = MiddlewareError::InvalidStateTransition {
            from: "Uninitialized".into(),
            to: "Ready".into(),
        };
        assert!(e.to_string().contains("Uninitialized"));
        assert!(e.to_string().contains("Ready"));
    }

    #[test]
    fn precision_display() {
        assert_eq!(Precision::FP32.to_string(), "FP32");
        assert_eq!(Precision::FP8.to_string(), "FP8");
    }

    #[test]
    fn precision_element_size() {
        assert_eq!(Precision::FP32.element_size(), 4);
        assert_eq!(Precision::FP16.element_size(), 2);
        assert_eq!(Precision::INT8.element_size(), 1);
        assert_eq!(Precision::FP8.element_size(), 1);
    }

    #[test]
    fn precision_from_str() {
        assert_eq!("fp32".parse::<Precision>().unwrap(), Precision::FP32);
        assert_eq!("FP16".parse::<Precision>().unwrap(), Precision::FP16);
        assert_eq!("int8".parse::<Precision>().unwrap(), Precision::INT8);
        assert_eq!("half".parse::<Precision>().unwrap(), Precision::FP16);
        assert!("banana".parse::<Precision>().is_err());
    }

    #[test]
    fn config_default_values() {
        let cfg = MiddlewareConfig::default();
        assert_eq!(cfg.fallback_precision, Precision::FP32);
        assert_eq!(cfg.warmup_runs, 3);
        assert!(!cfg.debug);
    }

    #[test]
    fn config_deserialize_from_toml() {
        let toml_str = r#"
            fallback_precision = "FP16"
            warmup_runs = 5
            debug = true
            labels_path = "/data/labels.txt"
        "#;
        let cfg: MiddlewareConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.fallback_precision, Precision::FP16);
        assert_eq!(cfg.warmup_runs, 5);
        assert!(cfg.debug);
        assert_eq!(cfg.labels_path, Some("/data/labels.txt".into()));
    }

    #[test]
    fn model_metadata_default() {
        let meta = ModelMetadata::default();
        assert_eq!(meta.name, "unknown");
        assert!(meta.inputs.is_empty());
        assert!(meta.outputs.is_empty());
    }
}
