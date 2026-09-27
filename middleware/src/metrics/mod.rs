// =============================================================================
// Magna Middleware - Metrics Module
// =============================================================================
//! Inference performance metrics, conditionally compiled via the `metrics`
//! Cargo feature flag.
//!
//! # Usage
//!
//! Enable at build time:
//! ```bash
//! cargo build --features metrics
//! cargo build --features "nvidia,metrics"
//! ```
//!
//! When the feature is **disabled** (the default), every call to
//! [`MetricsCollector`] compiles to nothing - zero mutex locks, zero timing
//! calls, zero binary overhead.
//!
//! When the feature is **enabled**, full latency tracking (min/max/avg/count),
//! throughput calculation, and memory reporting are available via
//! [`MetricsCollector::snapshot`].

// ---------------------------------------------------------------------------
// Full implementation (feature = "metrics")
// ---------------------------------------------------------------------------

#[cfg(feature = "metrics")]
mod inner {
    use std::time::{Duration, Instant};

    use serde::{Deserialize, Serialize};
    use tracing::debug;

    use crate::utils::errors::Precision;

    // == MetricsSnapshot =========================================================================

    /// Point-in-time snapshot of all collected metrics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MetricsSnapshot {
        /// Total number of inference runs recorded.
        pub inference_count: u64,
        /// Latest inference latency in milliseconds.
        pub last_latency_ms: f64,
        /// Average inference latency (over all recorded runs).
        pub avg_latency_ms: f64,
        /// Minimum inference latency.
        pub min_latency_ms: f64,
        /// Maximum inference latency.
        pub max_latency_ms: f64,
        /// Throughput estimate (inferences per second).
        pub throughput_ips: f64,
        /// Currently active precision mode, derived lazily from the engine.
        pub precision: Option<Precision>,
        /// Estimated model memory usage in bytes (0 if unavailable).
        pub memory_bytes: u64,
    }

    impl std::fmt::Display for MetricsSnapshot {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "== Inference Metrics ==================")?;
            writeln!(f, "  Count      : {}", self.inference_count)?;
            writeln!(f, "  Last       : {:.3} ms", self.last_latency_ms)?;
            writeln!(f, "  Average    : {:.3} ms", self.avg_latency_ms)?;
            writeln!(f, "  Min        : {:.3} ms", self.min_latency_ms)?;
            writeln!(f, "  Max        : {:.3} ms", self.max_latency_ms)?;
            writeln!(f, "  Throughput : {:.1} inf/s", self.throughput_ips)?;
            if let Some(p) = self.precision {
                writeln!(f, "  Precision  : {}", p)?;
            } else {
                writeln!(f, "  Precision  : Unknown (No engine)")?;
            }
            writeln!(f, "  Memory     : {} bytes", self.memory_bytes)?;
            write!(f, "=======================================")
        }
    }

    // == MetricsCollector ========================================================================

    /// Collects and aggregates inference performance metrics.
    ///
    /// Not thread-safe on its own - the public API layer wraps it in a
    /// `Mutex`.
    #[derive(Debug)]
    pub struct MetricsCollector {
        total_latency: Duration,
        min_latency: Duration,
        max_latency: Duration,
        last_latency: Duration,
        count: u64,
        memory_bytes: u64,
        timer: Option<Instant>,
    }

    impl Default for MetricsCollector {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MetricsCollector {
        /// Create a new, empty collector.
        pub fn new() -> Self {
            Self {
                total_latency: Duration::ZERO,
                min_latency: Duration::MAX,
                max_latency: Duration::ZERO,
                last_latency: Duration::ZERO,
                count: 0,
                memory_bytes: 0,
                timer: None,
            }
        }

        /// Start the per-inference timer.
        #[inline]
        pub fn start_timing(&mut self) {
            self.timer = Some(Instant::now());
        }

        /// Stop the timer and record the elapsed duration.
        ///
        /// Returns the elapsed [`Duration`], or [`Duration::ZERO`] if the
        /// timer was not started.
        #[inline]
        pub fn stop_timing(&mut self) -> Duration {
            let elapsed = match self.timer.take() {
                Some(start) => start.elapsed(),
                None => {
                    debug!("MetricsCollector::stop_timing called without start_timing");
                    Duration::ZERO
                }
            };
            self.record_latency(elapsed);
            elapsed
        }

        /// Manually record a latency measurement (e.g. measured outside the
        /// collector).
        pub fn record_latency(&mut self, duration: Duration) {
            self.count += 1;
            self.total_latency += duration;
            self.last_latency = duration;
            if duration < self.min_latency {
                self.min_latency = duration;
            }
            if duration > self.max_latency {
                self.max_latency = duration;
            }
        }

        /// Update the reported model memory usage.
        pub fn set_memory_usage(&mut self, bytes: u64) {
            self.memory_bytes = bytes;
        }

        /// Produce a [`MetricsSnapshot`] from the current state.
        pub fn snapshot(&self, precision: Option<Precision>) -> MetricsSnapshot {
            let avg_ms = if self.count > 0 {
                self.total_latency.as_secs_f64() * 1000.0 / self.count as f64
            } else {
                0.0
            };
            let throughput = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };

            MetricsSnapshot {
                inference_count: self.count,
                last_latency_ms: self.last_latency.as_secs_f64() * 1000.0,
                avg_latency_ms: avg_ms,
                min_latency_ms: if self.count > 0 {
                    self.min_latency.as_secs_f64() * 1000.0
                } else {
                    0.0
                },
                max_latency_ms: self.max_latency.as_secs_f64() * 1000.0,
                throughput_ips: throughput,
                precision,
                memory_bytes: self.memory_bytes,
            }
        }

        /// Reset all accumulated statistics.
        pub fn reset(&mut self) {
            self.total_latency = Duration::ZERO;
            self.min_latency = Duration::MAX;
            self.max_latency = Duration::ZERO;
            self.last_latency = Duration::ZERO;
            self.count = 0;
            self.timer = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-cost no-op stubs (feature = "metrics" disabled)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "metrics"))]
mod inner {
    use crate::utils::errors::Precision;

    /// Empty snapshot returned when metrics are disabled.
    ///
    /// All fields are zero / default. No computation is performed.
    #[derive(Debug, Clone, Default)]
    pub struct MetricsSnapshot {
        pub inference_count: u64,
        pub last_latency_ms: f64,
        pub avg_latency_ms: f64,
        pub min_latency_ms: f64,
        pub max_latency_ms: f64,
        pub throughput_ips: f64,
        pub precision: Option<Precision>,
        pub memory_bytes: u64,
    }

    /// No-op metrics collector - every method is a compile-time zero.
    ///
    /// When the `metrics` feature is disabled this struct contains no fields,
    /// occupies zero bytes (ZST after optimisation), and all method calls are
    /// completely elided by the compiler.
    #[derive(Debug, Default)]

    pub struct MetricsCollector {}

    impl MetricsCollector {
        #[inline(always)]
        pub fn new() -> Self {
            Self {}
        }

        #[inline(always)]
        pub fn start_timing(&mut self) {}

        #[inline(always)]
        pub fn stop_timing(&mut self) -> std::time::Duration {
            std::time::Duration::ZERO
        }

        #[inline(always)]
        pub fn record_latency(&mut self, _duration: std::time::Duration) {}

        #[inline(always)]
        pub fn set_memory_usage(&mut self, _bytes: u64) {}

        pub fn snapshot(&self, _precision: Option<Precision>) -> super::MetricsSnapshot {
            super::MetricsSnapshot {
                inference_count: 0,
                last_latency_ms: 0.0,
                avg_latency_ms: 0.0,
                min_latency_ms: 0.0,
                max_latency_ms: 0.0,
                throughput_ips: 0.0,
                precision: _precision,
                memory_bytes: 0,
            }
        }

        #[inline(always)]
        pub fn reset(&mut self) {}
    }
}

// ---------------------------------------------------------------------------
// Re-exports (same names regardless of feature flag)
// ---------------------------------------------------------------------------

pub use inner::{MetricsCollector, MetricsSnapshot};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use super::*;
    use crate::utils::errors::Precision;
    use std::time::Duration;

    #[test]
    fn empty_snapshot() {
        let mc = MetricsCollector::new();
        let snap = mc.snapshot(Some(Precision::FP32));
        assert_eq!(snap.inference_count, 0);
        assert_eq!(snap.avg_latency_ms, 0.0);
        assert_eq!(snap.throughput_ips, 0.0);
    }

    #[test]
    fn record_latency() {
        let mut mc = MetricsCollector::new();
        mc.record_latency(Duration::from_millis(10));
        mc.record_latency(Duration::from_millis(20));
        mc.record_latency(Duration::from_millis(30));

        let snap = mc.snapshot(Some(Precision::FP16));
        assert_eq!(snap.inference_count, 3);
        assert!((snap.avg_latency_ms - 20.0).abs() < 0.1);
        assert!((snap.min_latency_ms - 10.0).abs() < 0.1);
        assert!((snap.max_latency_ms - 30.0).abs() < 0.1);
        assert!(snap.throughput_ips > 0.0);
    }

    #[test]
    fn start_stop_timing() {
        let mut mc = MetricsCollector::new();
        mc.start_timing();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = mc.stop_timing();
        assert!(elapsed.as_millis() >= 4);
        assert_eq!(mc.snapshot(Some(Precision::FP32)).inference_count, 1);
    }

    #[test]
    fn reset_clears_state() {
        let mut mc = MetricsCollector::new();
        mc.record_latency(Duration::from_millis(50));
        mc.reset();
        let snap = mc.snapshot(Some(Precision::INT8));
        assert_eq!(snap.inference_count, 0);
    }

    #[test]
    fn display_format() {
        let mut mc = MetricsCollector::new();
        mc.record_latency(Duration::from_millis(12));
        let s = format!("{}", mc.snapshot(Some(Precision::FP32)));
        assert!(s.contains("Inference Metrics"));
        assert!(s.contains("12."));
    }

    #[test]
    fn memory_usage() {
        let mut mc = MetricsCollector::new();
        mc.set_memory_usage(1024 * 1024);
        assert_eq!(mc.snapshot(None).memory_bytes, 1024 * 1024);
    }
}

#[cfg(all(test, not(feature = "metrics")))]
mod noop_tests {
    use super::*;

    #[test]
    fn noop_collector_compiles_and_runs() {
        let mut mc = MetricsCollector::new();
        mc.start_timing();
        mc.stop_timing();
        mc.set_memory_usage(1024);
        mc.reset();
        let snap = mc.snapshot(None);
        // All fields are zero/default - nothing was recorded.
        assert_eq!(snap.inference_count, 0);
        assert_eq!(snap.avg_latency_ms, 0.0);
    }
}
