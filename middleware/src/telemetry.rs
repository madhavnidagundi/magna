use axum::{routing::get, Router};
use opentelemetry::{
    metrics::{Counter, Histogram, Meter, MeterProvider},
    KeyValue,
};
use opentelemetry_sdk::{metrics::SdkMeterProvider, Resource};
use prometheus::{Encoder, TextEncoder};
use std::net::SocketAddr;
use tokio::sync::oneshot;
use tracing::{error, info};

use std::sync::Arc;

#[derive(Clone)]
pub struct TelemetryProvider {
    pub meter: Meter,
    pub request_counter: Counter<u64>,
    pub latency_histogram: Histogram<f64>,
    /// TIDL-specific request counter. Registered on the same [`Meter`] as
    /// the generic `request_counter` — no second exporter/provider.
    pub tidl_request_counter: Counter<u64>,
    /// TIDL-specific latency histogram. Registered on the same [`Meter`] as
    /// the generic `latency_histogram` — no second exporter/provider.
    pub tidl_latency_histogram: Histogram<f64>,
    pub(crate) _provider: Arc<SdkMeterProvider>,
}

impl TelemetryProvider {
    /// Flush and shut down the underlying meter provider.
    ///
    /// Call this on shutdown when no `TelemetryServer` is available (e.g. the
    /// stdout-fallback path).
    pub fn shutdown(&self) {
        if let Err(e) = self._provider.force_flush() {
            error!(error = %e, "Failed to flush meter provider");
        }
        if let Err(e) = self._provider.shutdown() {
            error!(error = %e, "Failed to shut down meter provider");
        }
    }
}

pub struct TelemetryServer {
    shutdown_tx: Option<oneshot::Sender<()>>,
    provider: Arc<SdkMeterProvider>,
}

impl TelemetryServer {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Err(e) = self.provider.shutdown() {
            error!(error = %e, "Failed to shut down meter provider");
        }
        info!("Telemetry server shut down gracefully.");
    }
}

pub fn init_telemetry(
    device: &str,
    service_name: &str,
    version: &str,
) -> (TelemetryProvider, Option<TelemetryServer>) {
    let use_prometheus =
        std::env::var("OTEL_PROMETHEUS").unwrap_or_else(|_| "true".to_string()) == "true";
    let host = std::env::var("OTEL_METRICS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = match std::env::var("OTEL_METRICS_PORT") {
        Ok(v) => v.parse::<u16>().unwrap_or_else(|_| {
            tracing::warn!("Invalid OTEL_METRICS_PORT \"{}\", defaulting to 9090", v);
            9090
        }),
        Err(_) => 9090,
    };

    init_telemetry_impl(device, service_name, version, &host, port, use_prometheus)
}

pub fn init_telemetry_impl(
    device: &str,
    service_name: &str,
    version: &str,
    host: &str,
    port: u16,
    use_prometheus: bool,
) -> (TelemetryProvider, Option<TelemetryServer>) {
    let resource = Resource::builder()
        .with_attributes(vec![
            KeyValue::new("service.name", service_name.to_string()),
            KeyValue::new("service.version", version.to_string()),
            KeyValue::new("device", device.to_string()),
        ])
        .build();

    if use_prometheus {
        let registry = prometheus::Registry::new();
        let exporter = match opentelemetry_prometheus::exporter()
            .with_registry(registry.clone())
            .build()
        {
            Ok(e) => e,
            Err(err) => {
                tracing::error!("Failed to build Prometheus exporter: {}", err);
                return init_telemetry_impl(device, service_name, version, host, port, false);
            }
        };

        let provider = Arc::new(
            SdkMeterProvider::builder()
                .with_resource(resource)
                .with_reader(exporter)
                .build(),
        );

        let meter = provider.meter("magna-middleware");

        let request_counter = meter
            .u64_counter("inference_requests_total")
            .with_description("Total number of inference requests")
            .build();

        let latency_histogram = meter
            .f64_histogram("inference_latency_milliseconds")
            .with_description("Latency of inference requests in milliseconds")
            .build();

        let tidl_request_counter = meter
            .u64_counter("tidl_inference_requests_total")
            .with_description("Total number of TIDL inference requests")
            .build();

        let tidl_latency_histogram = meter
            .f64_histogram("tidl_inference_latency_milliseconds")
            .with_description("Latency of TIDL inference requests in milliseconds")
            .build();

        let provider_struct = TelemetryProvider {
            meter,
            request_counter,
            latency_histogram,
            tidl_request_counter,
            tidl_latency_histogram,
            _provider: provider.clone(),
        };

        // Start Prometheus HTTP server
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let app = Router::new().route(
            "/metrics",
            get({
                let registry = registry.clone();
                move || async move {
                    let mut buffer = vec![];
                    let encoder = TextEncoder::new();
                    let metric_families = registry.gather();
                    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
                        error!("Failed to encode metrics: {}", e);
                        return Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
                    }
                    Ok(String::from_utf8(buffer).unwrap_or_default())
                }
            }),
        );

        let addr_str = format!("{}:{}", host, port);
        let std_listener = match std::net::TcpListener::bind(&addr_str) {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, address = %addr_str, "Failed to bind Prometheus metrics server port");
                return (provider_struct, None);
            }
        };

        let addr = match std_listener.local_addr() {
            Ok(a) => a,
            Err(_) => SocketAddr::from(([127, 0, 0, 1], port)),
        };

        info!("Starting Prometheus metrics server on http://{}", addr);

        let server_started = (|| -> Result<(), Box<dyn std::error::Error>> {
            std_listener.set_nonblocking(true)?;
            let listener = tokio::net::TcpListener::from_std(std_listener)?;
            tokio::spawn(async move {
                let server = axum::serve(listener, app);
                let _ = server
                    .with_graceful_shutdown(async {
                        shutdown_rx.await.ok();
                    })
                    .await;
            });
            Ok(())
        })();

        match server_started {
            Ok(()) => (
                provider_struct,
                Some(TelemetryServer {
                    shutdown_tx: Some(shutdown_tx),
                    provider: provider.clone(),
                }),
            ),
            Err(e) => {
                error!(error = %e, "Failed to start Prometheus metrics server");
                (provider_struct, None)
            }
        }
    } else {
        // Stdout fallback
        let exporter = opentelemetry_stdout::MetricExporter::default();
        let reader = opentelemetry_sdk::metrics::PeriodicReader::builder(exporter).build();

        let provider = Arc::new(
            SdkMeterProvider::builder()
                .with_resource(resource)
                .with_reader(reader)
                .build(),
        );

        let meter = provider.meter("magna-middleware");

        let request_counter = meter
            .u64_counter("inference_requests_total")
            .with_description("Total number of inference requests")
            .build();

        let latency_histogram = meter
            .f64_histogram("inference_latency_milliseconds")
            .with_description("Latency of inference requests in milliseconds")
            .build();

        let tidl_request_counter = meter
            .u64_counter("tidl_inference_requests_total")
            .with_description("Total number of TIDL inference requests")
            .build();

        let tidl_latency_histogram = meter
            .f64_histogram("tidl_inference_latency_milliseconds")
            .with_description("Latency of TIDL inference requests in milliseconds")
            .build();

        let provider_struct = TelemetryProvider {
            meter,
            request_counter,
            latency_histogram,
            tidl_request_counter,
            tidl_latency_histogram,
            _provider: provider,
        };

        (provider_struct, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_init_telemetry_all() {
        // 1. Test stdout fallback
        let (provider, server) =
            init_telemetry_impl("cpu", "test-service", "1.0.0", "127.0.0.1", 0, false);
        assert!(server.is_none());
        provider.request_counter.add(1, &[]);
        provider.latency_histogram.record(12.34, &[]);

        // 2. Test prometheus
        let (provider, server) =
            init_telemetry_impl("cpu", "test-service", "1.0.0", "127.0.0.1", 0, true);
        assert!(server.is_some());
        provider.request_counter.add(1, &[]);
        provider.latency_histogram.record(12.34, &[]);

        if let Some(srv) = server {
            srv.shutdown().await;
        }
    }
}
