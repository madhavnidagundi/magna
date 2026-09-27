// =============================================================================
// Magna Middleware — Public API
// =============================================================================
//! Thread-safe, state-aware public API for the middleware.
//!
//! All methods are guarded by a mutex and check the lifecycle state before
//! performing any operation.  This is the **only** entry point that external
//! code should use.

use crate::inference::engine_manager::EngineManager;
use crate::inference::traits::{BackendCapabilities, EngineInfo, TensorBuffer};
use crate::lifecycle::state_manager::{State, StateManager};
#[cfg(feature = "metrics")]
use crate::metrics::{MetricsCollector, MetricsSnapshot};
use crate::postprocess::classifier::{Classifier, InferenceResult};
use crate::preprocess::imagenet;
#[cfg(test)]
use crate::utils::errors::Precision;
use crate::utils::errors::{MiddlewareConfig, MiddlewareError, MiddlewareResult};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use tracing::{error, info};
// `debug!` only used in metrics-gated timing logs, so conditionally imported
#[cfg(feature = "metrics")]
use tracing::debug;

// ---------------------------------------------------------------------------
// Middleware (inner state)
// ---------------------------------------------------------------------------

struct Inner {
    state: StateManager,
    config: MiddlewareConfig,
    engine_manager: Option<EngineManager>,
    classifier: Classifier,
    #[cfg(feature = "metrics")]
    metrics: Mutex<MetricsCollector>,
    #[cfg(feature = "opentelemetry")]
    telemetry: Option<crate::telemetry::TelemetryProvider>,
    last_prediction: Mutex<Option<InferenceResult>>,
}

// ---------------------------------------------------------------------------
// Middleware (public handle)
// ---------------------------------------------------------------------------

/// Thread-safe handle to the Magna middleware.
///
/// Clone this handle freely — all clones share the same state via `Arc<Mutex>`.
#[derive(Clone)]
pub struct Middleware {
    inner: Arc<RwLock<Inner>>,
}

impl Middleware {
    // == Lifecycle ===========================================================

    /// Create a new middleware instance (starts in `Uninitialized`).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                state: StateManager::new(),
                config: MiddlewareConfig::default(),
                engine_manager: None,
                classifier: Classifier::new(false),
                #[cfg(feature = "metrics")]
                metrics: Mutex::new(MetricsCollector::new()),
                #[cfg(feature = "opentelemetry")]
                telemetry: None,
                last_prediction: Mutex::new(None),
            })),
        }
    }

    #[cfg(feature = "opentelemetry")]
    /// Attach a telemetry provider to this middleware's shared inner state.
    ///
    /// All `Middleware` clones share the same `Arc<RwLock<Inner>>`, so calling
    /// this method updates telemetry for every clone that references the same
    /// inner state.
    pub fn with_telemetry(self, telemetry: crate::telemetry::TelemetryProvider) -> Self {
        self.inner.write().telemetry = Some(telemetry);
        self
    }

    /// Initialize the middleware with the given configuration.
    pub fn initialize(&self, config: MiddlewareConfig) -> MiddlewareResult<()> {
        if config.debug {
            info!(debug = true, "Debug logging enabled");
        }

        // Create engine manager using compile-time backend.
        let engine_manager = EngineManager::new()?;

        // Load labels if provided.
        let classifier = if let Some(ref labels_path) = config.labels_path {
            Classifier::load_labels_from_file(labels_path, true)?
        } else {
            Classifier::new(true)
        };

        let mut inner = self.inner.write();
        inner.state.transition_to(State::Initialized)?;

        // Update internal state.
        #[cfg(feature = "metrics")]
        {
            inner.metrics = Mutex::new(MetricsCollector::new());
        }
        inner.engine_manager = Some(engine_manager);
        inner.classifier = classifier;
        inner.config = config;

        let backend = inner
            .engine_manager
            .as_ref()
            .map(|em| em.backend_name().to_string())
            .unwrap_or_else(|| "none".into());
        info!(backend = %backend, "Middleware initialized");
        Ok(())
    }

    /// Build a hardware-optimized engine from an ONNX model.
    pub fn build_engine(
        &self,
        onnx_path: &str,
        engine_path: &str,
        precision: crate::utils::errors::Precision,
        calib_cache: Option<&str>,
    ) -> MiddlewareResult<()> {
        let inner = self.inner.read();
        inner.state.require_at_least(State::Initialized)?;

        let em = inner
            .engine_manager
            .as_ref()
            .ok_or_else(|| MiddlewareError::Internal("Engine manager not initialized".into()))?;

        em.get_backend()
            .build_engine(onnx_path, engine_path, precision, calib_cache)
    }

    /// Load a precompiled engine artifact.
    pub fn load_engine(&self, path: &str) -> MiddlewareResult<EngineInfo> {
        let mut inner = self.inner.write();

        inner.state.require_at_least(State::Initialized)?;

        let em = inner
            .engine_manager
            .as_mut()
            .ok_or_else(|| MiddlewareError::Internal("Engine manager not initialized".into()))?;

        let backend_name = em.backend_name().to_string();
        let info = em.load_engine(path).map_err(|e| {
            error!(engine_path = %path, error = %e, "Engine load failed");
            if backend_name != "qualcomm" {
                inner.state.set_error();
            }
            e
        })?;

        // Update metrics with model memory usage
        #[cfg(feature = "metrics")]
        inner.metrics.lock().set_memory_usage(info.memory_bytes);

        // Transition through EngineLoaded → Ready.
        if inner.state.current() == State::Initialized || inner.state.current() == State::Ready {
            inner.state.transition_to(State::EngineLoaded)?;
        }
        inner.state.transition_to(State::Ready)?;

        info!(
            engine  = %info.name,
            inputs  = info.inputs.len(),
            outputs = info.outputs.len(),
            "Engine loaded"
        );
        Ok(info)
    }

    // == Inference ===========================================================

    /// Run inference on generic input tensors.
    pub fn infer(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<Vec<TensorBuffer>> {
        let inner = self.inner.read();
        inner.state.require(State::Ready)?;

        #[cfg(feature = "opentelemetry")]
        let otel_start = inner.telemetry.as_ref().map(|_| std::time::Instant::now());

        #[cfg(feature = "metrics")]
        inner.metrics.lock().start_timing();

        let em = inner
            .engine_manager
            .as_ref()
            .ok_or(MiddlewareError::EngineNotLoaded)?;
        let output_res = em.infer(inputs);

        #[cfg(feature = "opentelemetry")]
        let telemetry_clone = inner.telemetry.clone();

        #[cfg(feature = "opentelemetry")]
        let backend_name = em.backend_name().to_string();

        #[cfg(feature = "metrics")]
        {
            let duration = inner.metrics.lock().stop_timing();
            if output_res.is_ok() {
                debug!(
                    latency_ms = duration.as_secs_f64() * 1000.0,
                    "Inference completed"
                );
            }
        }

        drop(inner);

        match output_res {
            Ok(output) => {
                #[cfg(feature = "opentelemetry")]
                if let (Some(tel), Some(start)) = (telemetry_clone, otel_start) {
                    tel.latency_histogram
                        .record(start.elapsed().as_secs_f64() * 1000.0, &[]);
                    tel.request_counter.add(1, &[]);

                    if backend_name == "ti" {
                        tel.tidl_latency_histogram
                            .record(start.elapsed().as_secs_f64() * 1000.0, &[]);
                        tel.tidl_request_counter.add(1, &[]);
                    }
                }

                Ok(output)
            }
            Err(e) => {
                error!(error = %e, "Inference failed");
                Err(e)
            }
        }
    }

    /// End-to-end: load image → preprocess → infer → postprocess.
    ///
    /// # Engine-Driven Precision
    /// The preprocessing precision is derived from the **loaded engine's input
    /// tensor spec** rather than `MiddlewareConfig::precision`. This eliminates
    /// the manual synchronisation bug where a user-configured `FP32` precision
    /// would mismatch an `INT8` or `FP16` engine, causing runtime size errors.
    ///
    /// Precision fallback order:
    /// 1. `engine_info.inputs[0].precision` (preferred — engine is source of truth)
    /// 2. `config.fallback_precision` (fallback when engine is not yet loaded)
    pub fn infer_from_image(&self, image_path: &str) -> MiddlewareResult<InferenceResult> {
        // Determine the expected input precision from the engine spec.
        // This is the core of Phase 2 (Engine-Driven Auto-Detection).
        let precision = {
            let inner = self.inner.read();
            inner
                .engine_manager
                .as_ref()
                .and_then(|em| em.engine_info())
                .and_then(|info| info.inputs.first().map(|spec| spec.precision))
                .unwrap_or(inner.config.fallback_precision) // fallback: use config if no engine loaded
        };

        tracing::debug!(
            precision = %precision,
            source = {
                let has_engine = {
                    let inner = self.inner.read();
                    inner.engine_manager.as_ref()
                        .and_then(|em| em.engine_info())
                        .and_then(|info| info.inputs.first().map(|_| true))
                        .unwrap_or(false)
                };
                if has_engine { "engine" } else { "config" }
            },
            "Preprocessing precision resolved"
        );

        let mut input_tensor = imagenet::preprocess_image_file(image_path, precision)?;

        // Use the actual input tensor name from the loaded engine.
        let input_name = {
            let inner = self.inner.read();
            inner
                .engine_manager
                .as_ref()
                .and_then(|em| em.engine_info())
                .and_then(|info| info.inputs.first().map(|spec| spec.name.clone()))
                .unwrap_or_else(|| "input".into())
        };
        input_tensor.name = input_name;

        let output = self.infer(&[input_tensor])?;

        let inner = self.inner.read();
        let result = InferenceResult::from_outputs(&output, Some(&inner.classifier))?;

        info!(result = %result, "Classification complete");
        *inner.last_prediction.lock() = Some(result.clone());

        Ok(result)
    }

    /// Run inference and return a generic result (raw tensor + optional classification).
    pub fn infer_generic(&self, inputs: &[TensorBuffer]) -> MiddlewareResult<InferenceResult> {
        let output = self.infer(inputs)?;

        let inner = self.inner.read();
        let result = InferenceResult::from_outputs(&output, Some(&inner.classifier))?;
        *inner.last_prediction.lock() = Some(result.clone());

        Ok(result)
    }

    // == Queries =============================================================

    /// Get the result of the last inference run.
    pub fn get_last_prediction(&self) -> MiddlewareResult<InferenceResult> {
        let inner = self.inner.read();
        let x = inner.last_prediction.lock().clone();
        x.ok_or(MiddlewareError::Internal("No prediction available".into()))
    }

    /// Get a snapshot of collected metrics.
    ///
    /// Returns an empty (all-zero) snapshot when the `metrics` feature is
    /// disabled.
    #[cfg(feature = "metrics")]
    pub fn get_metrics(&self) -> MetricsSnapshot {
        let inner = self.inner.read();
        let engine_precision = inner
            .engine_manager
            .as_ref()
            .and_then(|em| em.engine_info())
            .and_then(|info| info.inputs.first().map(|s| s.precision));
        let snapshot = inner.metrics.lock().snapshot(engine_precision);
        snapshot
    }

    /// Stub returned when `metrics` feature is disabled.
    #[cfg(not(feature = "metrics"))]
    pub fn get_metrics(&self) -> crate::metrics::MetricsSnapshot {
        crate::metrics::MetricsSnapshot {
            inference_count: 0,
            last_latency_ms: 0.0,
            avg_latency_ms: 0.0,
            min_latency_ms: 0.0,
            max_latency_ms: 0.0,
            throughput_ips: 0.0,
            precision: None,
            memory_bytes: 0,
        }
    }

    /// Get the current lifecycle state.
    pub fn state(&self) -> State {
        let inner = self.inner.read();
        inner.state.current()
    }

    /// Get info about the currently loaded engine, if any.
    pub fn engine_info(&self) -> Option<EngineInfo> {
        let inner = self.inner.read();
        inner
            .engine_manager
            .as_ref()
            .and_then(|em| em.engine_info())
    }

    /// Get explicit capabilities for the selected backend.
    pub fn capabilities(&self) -> Option<BackendCapabilities> {
        let inner = self.inner.read();
        inner.engine_manager.as_ref().map(|em| em.capabilities())
    }

    /// Get the current configuration.
    pub fn config(&self) -> MiddlewareConfig {
        let inner = self.inner.read();
        inner.config.clone()
    }

    /// Get the backend name.
    pub fn backend_name(&self) -> String {
        let inner = self.inner.read();
        inner
            .engine_manager
            .as_ref()
            .map(|em| em.backend_name().to_string())
            .unwrap_or_else(|| "none".into())
    }

    /// Check if an engine is loaded and ready for inference.
    pub fn is_ready(&self) -> bool {
        let inner = self.inner.read();
        inner
            .engine_manager
            .as_ref()
            .map(|em| em.is_ready())
            .unwrap_or(false)
    }

    // == Shutdown ============================================================

    /// Gracefully shut down the middleware, releasing all resources.
    pub fn shutdown(&self) -> MiddlewareResult<()> {
        let mut inner = self.inner.write();

        if let Some(ref mut em) = inner.engine_manager {
            em.release().ok(); // Best-effort release.
        }
        inner.engine_manager = None;
        *inner.last_prediction.lock() = None;
        #[cfg(feature = "metrics")]
        inner.metrics.lock().reset();

        inner.state.transition_to(State::Uninitialized)?;
        info!("Middleware shut down");
        Ok(())
    }
}

impl Default for Middleware {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Middleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        let backend = inner
            .engine_manager
            .as_ref()
            .map(|em| em.backend_name().to_string())
            .unwrap_or_else(|| "none".into());
        f.debug_struct("Middleware")
            .field("state", &inner.state.current())
            .field("backend", &backend)
            .field("fallback_precision", &inner.config.fallback_precision)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_middleware_is_uninitialized() {
        let mw = Middleware::new();
        assert_eq!(mw.state(), State::Uninitialized);
    }

    #[test]
    fn initialize_with_simulated_backend() {
        let mw = Middleware::new();
        let config = MiddlewareConfig {
            fallback_precision: Precision::FP32,
            ..Default::default()
        };
        mw.initialize(config).unwrap();
        assert_eq!(mw.state(), State::Initialized);
    }

    #[test]
    fn full_lifecycle() {
        let mw = Middleware::new();
        mw.initialize(MiddlewareConfig {
            fallback_precision: Precision::FP32,
            ..Default::default()
        })
        .unwrap();

        println!("DEBUG: backend_name = {}", mw.backend_name());

        let path;
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

        if let Ok(p) = std::env::var("TEST_ENGINE_PATH") {
            let pb = std::path::PathBuf::from(&p);
            if pb.exists() {
                path = p;
            } else {
                eprintln!(
                    "SKIPPED [full_lifecycle]: TEST_ENGINE_PATH set but not found at {}",
                    p
                );
                return;
            }
        } else if mw.backend_name() == "cpu" {
            let p = std::path::Path::new(&manifest_dir).join("../models/mobilenetv2.onnx");
            if p.exists() {
                path = p.to_string_lossy().to_string();
            } else {
                eprintln!(
                    "SKIPPED [full_lifecycle]: ONNX test model not found at {}",
                    p.display()
                );
                return;
            }
        } else {
            let p = std::path::Path::new(&manifest_dir)
                .join("../models/mobilenetv4_nvidia_optimized_int8.engine");
            if p.exists() {
                path = p.to_string_lossy().to_string();
            } else {
                eprintln!(
                    "SKIPPED [full_lifecycle]: TensorRT test engine not found at {}",
                    p.display()
                );
                return;
            }
        }

        let info = mw.load_engine(&path).unwrap();
        assert_eq!(mw.state(), State::Ready);

        let input = TensorBuffer {
            name: info.inputs[0].name.clone(),
            data: vec![0u8; 3 * 224 * 224 * 4],
            shape: vec![1, 3, 224, 224],
            precision: Precision::FP32,
        };
        let output = mw.infer(&[input]).unwrap();
        assert_eq!(output[0].shape, vec![1, 1000]);

        #[cfg(feature = "metrics")]
        {
            let metrics = mw.get_metrics();
            assert_eq!(metrics.inference_count, 1);
        }

        mw.shutdown().unwrap();
        assert_eq!(mw.state(), State::Uninitialized);
    }

    #[test]
    fn infer_before_ready_fails() {
        let mw = Middleware::new();
        mw.initialize(MiddlewareConfig {
            ..Default::default()
        })
        .unwrap();

        let input = TensorBuffer::from_f32("input", &[0.0; 6], vec![1, 2, 3]);
        assert!(mw.infer(&[input]).is_err());
    }

    #[test]
    fn thread_safe_clone() {
        let mw = Middleware::new();
        let mw2 = mw.clone();

        let handle = std::thread::spawn(move || {
            mw2.initialize(MiddlewareConfig {
                ..Default::default()
            })
            .unwrap();
        });

        handle.join().unwrap();
        assert_eq!(mw.state(), State::Initialized);
    }
}
