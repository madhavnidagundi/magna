// =============================================================================
// Magna Middleware — Centralized Logging Configuration
// =============================================================================
//! Async-safe, structured logging initialisation for all Magna binaries.
//!
//! Backed by the [`tracing`] ecosystem, which is fully compatible with Tokio's
//! async runtime.  All log calls throughout the library use `tracing` macros
//! (`tracing::info!`, `tracing::debug!`, etc.) which emit structured events
//! rather than plain strings.
//!
//! # Usage (in binary `main`)
//! ```rust,no_run
//! use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};
//!
//! init_logger(LogConfig {
//!     format: LogFormat::Pretty,
//!     default_level: "info".into(),
//! });
//! ```
//!
//! # Runtime control
//! The `RUST_LOG` environment variable overrides `default_level` and supports
//! per-module filtering:
//! ```bash
//! RUST_LOG=info,magna_middleware::backends::nvidia=debug ./magna_server ...
//! ```

use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Output format for log records.
#[derive(Debug, Clone, Default)]
pub enum LogFormat {
    /// Human-readable, coloured output for development.
    #[default]
    Pretty,
    /// Newline-delimited JSON — suitable for ELK / Splunk / CloudWatch.
    Json,
    /// Single-line compact output for CI / embedded deployments.
    Compact,
}

/// Configuration passed to [`init_logger`].
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Output format. Defaults to [`LogFormat::Pretty`].
    pub format: LogFormat,
    /// Minimum log level when `RUST_LOG` is not set.
    /// Accepts `"trace"`, `"debug"`, `"info"`, `"warn"`, `"error"`.
    pub default_level: String,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            format: LogFormat::Pretty,
            default_level: "info".into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Initialiser
// ---------------------------------------------------------------------------

/// Initialise the global tracing subscriber.
///
/// Call **once** at the top of every binary `main` function.  Subsequent
/// calls are silently ignored (the subscriber is already set).
///
/// The `RUST_LOG` environment variable always takes precedence over
/// `config.default_level`.
pub fn init_logger(config: LogConfig) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.default_level));

    match config.format {
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().json().with_current_span(true))
                .try_init()
                .ok();
        }
        LogFormat::Compact => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().compact())
                .try_init()
                .ok();
        }
        LogFormat::Pretty => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer())
                .try_init()
                .ok();
        }
    }
}

/// Convenience initialiser for unit tests.
///
/// Uses `with_test_writer()` so log output appears in `cargo test` output
/// only for failing tests.  Safe to call multiple times across tests.
pub fn init_test_logger() {
    tracing_subscriber::fmt()
        .with_test_writer()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .try_init()
        .ok();
}
