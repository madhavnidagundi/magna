// =============================================================================
// Magna Middleware — Library Root
// =============================================================================
//! # Magna Middleware
//!
//! A modular, production-grade middleware for hardware-agnostic AOT
//! (Ahead-of-Time) compiled inference execution on embedded edge AI hardware.
//!
//! ## Supported Hardware
//! - **NVIDIA** Orin / Thor (via TensorRT `.engine` files)
//! - **Qualcomm** Snapdragon (via SNPE / QNN)
//! - **Texas Instruments** TDA4 / Jacinto (via TIDL)
//! - **CPU** fallback (via ONNX Runtime, for development/testing)
//!
//! ## Key Features
//! - **Hardware-agnostic**: Single API works across all platforms
//! - **Dataset-agnostic**: No ImageNet or classification assumptions
//! - **Task-agnostic**: Raw tensor output, optional classification
//! - **gRPC interface**: Network-accessible inference service
//! - **AOT compilation**: `create_middleware` CLI for deployment
//!
//! ## Quick Start
//! ```rust,no_run
//! use magna_middleware::api::public_api::Middleware;
//! use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
//!
//! let mw = Middleware::new();
//!
//! mw.initialize(MiddlewareConfig {
//!     fallback_precision: Precision::FP32,
//!     ..Default::default()
//! }).unwrap();
//!
//! mw.load_engine("model.engine").unwrap();
//!
//! // Option A: raw tensor inference
//! // let output = mw.infer(&input_tensor).unwrap();
//!
//! // Option B: end-to-end image inference
//! // let result = mw.infer_from_image("cat.jpg").unwrap();
//!
//! mw.shutdown().unwrap();
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod api;
pub mod backends;
pub mod hardware;
pub mod inference;
pub mod lifecycle;
pub mod metrics;
pub mod postprocess;
pub mod preprocess;
#[cfg(feature = "opentelemetry")]
pub mod telemetry;
pub mod utils;

// Re-export the most commonly used types at crate root for convenience.
pub use api::public_api::Middleware;
pub use utils::errors::{MiddlewareConfig, MiddlewareError, MiddlewareResult, Precision};
