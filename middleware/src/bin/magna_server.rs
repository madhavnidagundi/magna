// =============================================================================
// Magna Middleware ΓÇö gRPC Inference Server
// =============================================================================
//! Production-grade gRPC server that wraps the Middleware public API.
//!
//! Features:
//! - Handles RGB, BGR, and JPEG image formats
//! - Returns raw output tensor + optional classification
//! - Measures and reports inference latency
//! - Health check endpoint
//! - Graceful shutdown on SIGTERM/Ctrl+C
//! - Configurable max request size

use clap::Parser;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, error, info};

use magna_middleware::api::grpc::magna_grpc::inference_service_server::{
    InferenceService, InferenceServiceServer,
};
use magna_middleware::api::grpc::magna_grpc::{
    BackendCapabilityInfo, GetModelInfoRequest, GetModelInfoResponse, HealthRequest,
    HealthResponse, InferenceRequest, InferenceResponse, TensorSpecInfo,
};
use magna_middleware::api::public_api::Middleware;
use magna_middleware::inference::traits::{
    BackendAvailability, BackendCapabilities, EngineInfo, TensorBuffer,
};
use magna_middleware::inference::validation;
use magna_middleware::utils::errors::{MiddlewareConfig, MiddlewareError, Precision};
use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "magna_server",
    about = "Magna Middleware gRPC Inference Server"
)]
struct Cli {
    /// Path to the compiled engine / model file.
    #[arg(short, long)]
    model: String,

    /// Precision: fp32, fp16, int8, fp8.
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// gRPC listen address.
    #[arg(short, long, default_value = "127.0.0.1:50051")]
    address: String,

    /// Path to labels file (optional, enables classification output).
    #[arg(short, long)]
    labels: Option<String>,

    /// Maximum request size in bytes (default 10 MB).
    #[arg(long, default_value = "10485760")]
    max_request_size: usize,

    /// Maximum total size of a streamed model upload (default 64 GiB).
    #[arg(long, default_value = "68719476736")]
    max_upload_size: u64,

    /// Enable debug logging.
    #[arg(short, long)]
    debug: bool,

    /// Log output format: pretty, json, compact.
    #[arg(long, default_value = "pretty")]
    log_format: String,
}

// ---------------------------------------------------------------------------
// Request correlation ID counter
// ---------------------------------------------------------------------------

/// Lock-free monotonic counter for per-request correlation IDs.
static REQUEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[inline]
fn next_request_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

struct MagnaInferenceService {
    middleware: Middleware,
    max_upload_size: u64,
}

fn is_simple_filename(filename: &str) -> bool {
    !filename.is_empty()
        && filename != "."
        && filename != ".."
        && Path::new(filename)
            .file_name()
            .and_then(|name| name.to_str())
            == Some(filename)
}

fn is_allowed_upload_dir(target_dir: &str) -> bool {
    matches!(target_dir, "models" | "contexts" | "eval_data")
}

// Keep large tonic statuses off the helpers' success-path stack. The service
// converts back to tonic's required Status type at the RPC boundary.
fn parse_grpc_precision(datatype: &str, strict: bool) -> Result<Precision, Box<Status>> {
    if strict {
        datatype.parse::<Precision>().map_err(|_| {
            Status::invalid_argument(format!(
                "Unknown datatype '{}'; expected FP32, FP16, INT8, or FP8",
                datatype
            ))
            .into()
        })
    } else {
        Ok(match datatype {
            "FP32" => Precision::FP32,
            "FP16" => Precision::FP16,
            "INT8" => Precision::INT8,
            "FP8" => Precision::FP8,
            _ => Precision::FP32,
        })
    }
}

fn convert_inference_request(
    req: &InferenceRequest,
    backend_name: &str,
    engine: Option<&EngineInfo>,
    capabilities: Option<&BackendCapabilities>,
) -> Result<Vec<TensorBuffer>, Box<Status>> {
    let strict_qualcomm = backend_name == "qualcomm";
    if req.inputs.is_empty() {
        return Err(Status::invalid_argument("inputs is empty").into());
    }

    let mut buffers = Vec::with_capacity(req.inputs.len());
    for grpc_in in &req.inputs {
        let precision = parse_grpc_precision(&grpc_in.datatype, strict_qualcomm)?;
        if strict_qualcomm && grpc_in.name.is_empty() {
            return Err(Status::invalid_argument("input tensor name is empty").into());
        }

        let mut shape = Vec::with_capacity(grpc_in.shape.len());
        for (idx, &dim) in grpc_in.shape.iter().enumerate() {
            if strict_qualcomm && dim <= 0 {
                return Err(Status::invalid_argument(format!(
                    "input tensor '{}' has non-positive dimension {} at index {}",
                    grpc_in.name, dim, idx
                ))
                .into());
            }
            let dim = usize::try_from(dim).map_err(|_| {
                Status::invalid_argument(format!(
                    "input tensor '{}' dimension {} cannot be represented as usize",
                    grpc_in.name, dim
                ))
            })?;
            shape.push(dim);
        }

        buffers.push(TensorBuffer {
            name: grpc_in.name.clone(),
            data: grpc_in.raw_data.clone(),
            shape,
            precision,
        });
    }

    if strict_qualcomm {
        let engine = engine.ok_or_else(|| Status::failed_precondition("no model is loaded"))?;
        let capabilities = capabilities
            .ok_or_else(|| Status::failed_precondition("backend capabilities unavailable"))?;
        validation::validate_qualcomm_request(&buffers, engine, capabilities)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
    }

    Ok(buffers)
}

fn map_inference_error(e: MiddlewareError) -> Status {
    match e {
        MiddlewareError::EngineNotLoaded | MiddlewareError::InvalidState { .. } => {
            Status::failed_precondition(e.to_string())
        }
        MiddlewareError::InferenceFailed(_) => Status::invalid_argument(e.to_string()),
        MiddlewareError::BackendUnavailable(_) => Status::failed_precondition(e.to_string()),
        _ => Status::internal(e.to_string()),
    }
}

fn precision_to_proto(precision: Precision) -> String {
    precision.to_string()
}

fn availability_to_proto(availability: &BackendAvailability) -> String {
    match availability {
        BackendAvailability::Available => "Available".into(),
        BackendAvailability::Unavailable { reason } => format!("Unavailable: {reason}"),
        BackendAvailability::Stub { reason } => format!("Stub: {reason}"),
    }
}

fn spec_to_proto(spec: &magna_middleware::inference::traits::TensorSpec) -> TensorSpecInfo {
    TensorSpecInfo {
        name: spec.name.clone(),
        datatype: precision_to_proto(spec.precision),
        shape: spec.shape.iter().map(|&s| s as i64).collect(),
    }
}

#[tonic::async_trait]
impl InferenceService for MagnaInferenceService {
    async fn infer(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let req = request.into_inner();

        let req_id = next_request_id();
        debug!(
            request_id = req_id,
            input_count = req.inputs.len(),
            "Inference request received"
        );

        let backend_name = self.middleware.backend_name();
        let engine_info = self.middleware.engine_info();
        let capabilities = self.middleware.capabilities();
        let buffers = convert_inference_request(
            &req,
            &backend_name,
            engine_info.as_ref(),
            capabilities.as_ref(),
        )
        .map_err(|status| *status)?;

        // Run inference
        let start = Instant::now();
        let result = self.middleware.infer_generic(&buffers).map_err(|e| {
            error!(request_id = req_id, error = %e, "Inference failed");
            map_inference_error(e)
        })?;
        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Build response
        let mut resp_outputs = Vec::new();
        for tb in &result.outputs {
            let datatype = match tb.precision {
                Precision::FP32 => "FP32",
                Precision::FP16 => "FP16",
                Precision::INT8 => "INT8",
                Precision::FP8 => "FP8",
            }
            .to_string();

            resp_outputs.push(magna_middleware::api::grpc::magna_grpc::InferOutputTensor {
                name: tb.name.clone(),
                datatype,
                shape: tb.shape.iter().map(|&s| s as i64).collect(),
                raw_data: tb.data.clone(),
            });
        }

        let resp = InferenceResponse {
            outputs: resp_outputs,
            inference_time_ms: elapsed_ms,
        };

        debug!(
            request_id = req_id,
            output_count = resp.outputs.len(),
            latency_ms = elapsed_ms,
            "Inference complete"
        );
        Ok(Response::new(resp))
    }

    async fn health_check(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let state = self.middleware.state();
        let engine = self.middleware.engine_info();

        // inference_count is 0 when metrics feature is disabled
        #[cfg(feature = "metrics")]
        let inference_count = self.middleware.get_metrics().inference_count;
        #[cfg(not(feature = "metrics"))]
        let inference_count: u64 = 0;

        Ok(Response::new(HealthResponse {
            ready: state == magna_middleware::lifecycle::state_manager::State::Ready,
            state: format!("{:?}", state),
            backend: self.middleware.backend_name(),
            model_name: engine.map(|e| e.name).unwrap_or_default(),
            inference_count,
        }))
    }

    async fn upload_model(
        &self,
        request: Request<tonic::Streaming<magna_middleware::api::grpc::magna_grpc::ModelChunk>>,
    ) -> Result<Response<magna_middleware::api::grpc::magna_grpc::UploadModelResponse>, Status>
    {
        let mut stream = request.into_inner();
        let mut file = None;
        let mut bytes_received = 0u64;
        let mut expected_size = None;
        let mut remote_path = String::new();

        while let Some(chunk) = stream.message().await? {
            if file.is_none() {
                let filename = chunk.filename;
                let target_dir = if chunk.target_dir.is_empty() {
                    "models"
                } else {
                    chunk.target_dir.as_str()
                };

                if !is_simple_filename(&filename) {
                    return Err(Status::invalid_argument("Invalid filename"));
                }

                if !is_allowed_upload_dir(target_dir) {
                    return Err(Status::invalid_argument(
                        "target_dir must be models, contexts, or eval_data",
                    ));
                }

                if chunk.total_size > self.max_upload_size {
                    return Err(Status::resource_exhausted(format!(
                        "Upload exceeds the {} byte limit",
                        self.max_upload_size
                    )));
                }
                expected_size = (chunk.total_size != 0).then_some(chunk.total_size);

                let dir = std::path::PathBuf::from(target_dir);
                if let Err(e) = tokio::fs::create_dir_all(&dir).await {
                    return Err(Status::internal(format!("Failed to create dir: {}", e)));
                }

                if tokio::fs::symlink_metadata(&dir)
                    .await
                    .map(|metadata| metadata.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    return Err(Status::invalid_argument("target_dir cannot be a symlink"));
                }

                let path = dir.join(&filename);
                remote_path = path.to_string_lossy().to_string();

                if tokio::fs::symlink_metadata(&path)
                    .await
                    .map(|metadata| metadata.file_type().is_symlink())
                    .unwrap_or(false)
                {
                    return Err(Status::invalid_argument(
                        "Upload destination cannot be a symlink",
                    ));
                }

                match tokio::fs::File::create(&path).await {
                    Ok(f) => file = Some(f),
                    Err(e) => {
                        return Err(Status::internal(format!("Failed to create file: {}", e)))
                    }
                }
            }

            bytes_received = bytes_received
                .checked_add(chunk.data.len() as u64)
                .ok_or_else(|| Status::resource_exhausted("Upload size overflow"))?;
            if bytes_received > self.max_upload_size
                || expected_size.is_some_and(|size| bytes_received > size)
            {
                return Err(Status::resource_exhausted(
                    "Upload exceeds its declared limit",
                ));
            }

            if let Some(ref mut f) = file {
                if let Err(e) = f.write_all(&chunk.data).await {
                    return Err(Status::internal(format!("Failed to write chunk: {}", e)));
                }
            }
        }

        if file.is_none() {
            return Err(Status::invalid_argument("Upload stream is empty"));
        }
        if expected_size.is_some_and(|size| bytes_received != size) {
            return Err(Status::invalid_argument(
                "Received size does not match declared total_size",
            ));
        }

        if let Some(mut f) = file {
            if let Err(e) = f.flush().await {
                return Err(Status::internal(format!("Failed to flush file: {}", e)));
            }
        }

        Ok(Response::new(
            magna_middleware::api::grpc::magna_grpc::UploadModelResponse {
                success: true,
                message: "Model uploaded successfully".to_string(),
                remote_path,
                bytes_received,
            },
        ))
    }

    async fn prepare_context(
        &self,
        request: Request<magna_middleware::api::grpc::magna_grpc::PrepareContextRequest>,
    ) -> Result<Response<magna_middleware::api::grpc::magna_grpc::PrepareContextResponse>, Status>
    {
        let req = request.into_inner();

        #[cfg(not(feature = "qualcomm"))]
        {
            let _ = req;
            return Err(Status::unimplemented(
                "Context preparation is only supported on Qualcomm backend",
            ));
        }

        #[cfg(feature = "qualcomm")]
        {
            if !is_simple_filename(&req.dlc_filename) || !req.dlc_filename.ends_with(".dlc") {
                return Err(Status::invalid_argument(
                    "dlc_filename must be a plain .dlc filename",
                ));
            }

            let dlc_path = std::path::PathBuf::from("models").join(&req.dlc_filename);
            if !dlc_path.exists() {
                return Err(Status::not_found("DLC file not found in ./models"));
            }

            let basename = req
                .dlc_filename
                .strip_suffix(".dlc")
                .unwrap_or(&req.dlc_filename);
            let ctx_dir = std::path::PathBuf::from("generated_ctx");
            if let Err(e) = tokio::fs::create_dir_all(&ctx_dir).await {
                return Err(Status::internal(format!(
                    "Failed to create generated_ctx: {}",
                    e
                )));
            }

            let output_bin = ctx_dir.join(format!("{}_native.bin", basename));
            if output_bin.exists() && !req.force {
                return Ok(Response::new(
                    magna_middleware::api::grpc::magna_grpc::PrepareContextResponse {
                        success: true,
                        message: "Context already exists".to_string(),
                        context_path: output_bin.to_string_lossy().to_string(),
                    },
                ));
            }

            let output = tokio::process::Command::new("./bin/qnn-context-binary-generator")
                .arg("--model")
                .arg("./lib/libQnnModelDlc.so")
                .arg("--backend")
                .arg("./lib/libQnnHtp.so")
                .arg("--dlc_path")
                .arg(&dlc_path)
                .arg("--binary_file")
                .arg(format!("{}_native", basename))
                .arg("--output_dir")
                .arg(&ctx_dir)
                .output()
                .await
                .map_err(|e| Status::internal(format!("Failed to execute QNN generator: {}", e)))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(Status::internal(format!(
                    "QNN generator failed: {}",
                    stderr
                )));
            }

            Ok(Response::new(
                magna_middleware::api::grpc::magna_grpc::PrepareContextResponse {
                    success: true,
                    message: "Context generated successfully".to_string(),
                    context_path: output_bin.to_string_lossy().to_string(),
                },
            ))
        }
    }

    async fn list_models(
        &self,
        _request: Request<magna_middleware::api::grpc::magna_grpc::ListModelsRequest>,
    ) -> Result<Response<magna_middleware::api::grpc::magna_grpc::ListModelsResponse>, Status> {
        let mut models = Vec::new();

        for dir in &["models", "generated_ctx"] {
            if let Ok(mut entries) = tokio::fs::read_dir(dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }

                    if let Ok(metadata) = entry.metadata().await {
                        let filename = path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        let model_type = if filename.ends_with(".dlc") {
                            "dlc"
                        } else if filename.ends_with(".bin") {
                            "bin"
                        } else if filename.ends_with(".onnx") {
                            "onnx"
                        } else if filename.ends_with(".engine") {
                            "tensorrt"
                        } else {
                            "unknown"
                        };

                        models.push(magna_middleware::api::grpc::magna_grpc::ModelInfo {
                            filename,
                            path: path.to_string_lossy().to_string(),
                            size_bytes: metadata.len(),
                            model_type: model_type.to_string(),
                        });
                    }
                }
            }
        }

        Ok(Response::new(
            magna_middleware::api::grpc::magna_grpc::ListModelsResponse { models },
        ))
    }

    async fn get_model_info(
        &self,
        _request: Request<GetModelInfoRequest>,
    ) -> Result<Response<GetModelInfoResponse>, Status> {
        let engine = self
            .middleware
            .engine_info()
            .ok_or_else(|| Status::failed_precondition("no model is loaded"))?;
        let capabilities = self
            .middleware
            .capabilities()
            .ok_or_else(|| Status::failed_precondition("backend capabilities unavailable"))?;

        Ok(Response::new(GetModelInfoResponse {
            model_name: engine.name,
            backend: capabilities.backend_name.clone(),
            backend_availability: availability_to_proto(&capabilities.availability),
            inputs: engine.inputs.iter().map(spec_to_proto).collect(),
            outputs: engine.outputs.iter().map(spec_to_proto).collect(),
            capabilities: Some(BackendCapabilityInfo {
                max_inputs: capabilities.max_inputs as u64,
                max_outputs: capabilities.max_outputs as u64,
                supported_input_precisions: capabilities
                    .supported_input_precisions
                    .iter()
                    .copied()
                    .map(precision_to_proto)
                    .collect(),
                supported_output_precisions: capabilities
                    .supported_output_precisions
                    .iter()
                    .copied()
                    .map(precision_to_proto)
                    .collect(),
                supports_engine_building: capabilities.supports_engine_building,
                supports_dynamic_shapes: capabilities.supports_dynamic_shapes,
            }),
        }))
    }
}

// (Removed decode_image_data entirely due to new Generic grpc interface)

// ---------------------------------------------------------------------------
// Server startup
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let format = match cli.log_format.as_str() {
        "json" => LogFormat::Json,
        "compact" => LogFormat::Compact,
        _ => LogFormat::Pretty,
    };
    let default_level = if cli.debug { "debug" } else { "info" };
    init_logger(LogConfig {
        format,
        default_level: default_level.into(),
    });

    info!(
        precision = %cli.precision,
        model     = %cli.model,
        address   = %cli.address,
        "Magna gRPC Inference Server starting"
    );

    let precision: Precision = cli
        .precision
        .parse()
        .map_err(|e| format!("Invalid precision: {}", e))?;

    #[allow(unused_mut)]
    let mut mw = Middleware::new();

    let config = MiddlewareConfig {
        fallback_precision: precision,
        warmup_runs: 1,
        debug: cli.debug,
        labels_path: cli.labels.clone(),
        model_path: Some(cli.model.clone()),
        grpc_address: cli.address.clone(),
    };

    mw.initialize(config)?;

    #[cfg(feature = "opentelemetry")]
    let (_provider, telemetry_server) = {
        let device_name = mw.backend_name();
        let (provider, server) = magna_middleware::telemetry::init_telemetry(
            &device_name,
            "magna-server",
            env!("CARGO_PKG_VERSION"),
        );
        mw = mw.with_telemetry(provider.clone());
        (provider, server)
    };
    let engine_info = mw.load_engine(&cli.model)?;
    info!(
        engine  = %engine_info.name,
        inputs  = engine_info.inputs.len(),
        outputs = engine_info.outputs.len(),
        "Engine loaded"
    );

    let addr: SocketAddr = cli.address.parse()?;
    let service = MagnaInferenceService {
        middleware: mw.clone(),
        max_upload_size: cli.max_upload_size,
    };

    // Configure server with request size limit
    let server = Server::builder().add_service(
        InferenceServiceServer::new(service).max_decoding_message_size(cli.max_request_size),
    );

    info!(address = %addr, "Listening for gRPC connections");

    // Graceful shutdown on Ctrl+C / SIGTERM
    let shutdown_mw = mw.clone();
    server
        .serve_with_shutdown(addr, async move {
            tokio::signal::ctrl_c().await.ok();
            info!("Shutdown signal received — cleaning up");
            if let Err(e) = shutdown_mw.shutdown() {
                error!(error = %e, "Shutdown error");
            }
            #[cfg(feature = "opentelemetry")]
            {
                if let Some(ts) = telemetry_server {
                    ts.shutdown().await;
                } else {
                    _provider.shutdown();
                }
            }
            info!("Shutdown complete");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{convert_inference_request, is_allowed_upload_dir, is_simple_filename};
    use magna_middleware::inference::traits::{
        BackendAvailability, BackendCapabilities, EngineInfo, TensorSpec,
    };
    use magna_middleware::utils::errors::Precision;

    #[test]
    fn model_management_paths_are_restricted() {
        assert!(is_simple_filename("model.dlc"));
        assert!(!is_simple_filename("../model.dlc"));
        assert!(!is_simple_filename("models/model.dlc"));
        assert!(!is_simple_filename(".."));

        assert!(is_allowed_upload_dir("models"));
        assert!(is_allowed_upload_dir("contexts"));
        assert!(is_allowed_upload_dir("eval_data"));
        assert!(!is_allowed_upload_dir("middleware/src"));
        assert!(!is_allowed_upload_dir("../models"));
    }

    #[test]
    fn qualcomm_request_conversion_rejects_bad_datatype() {
        let req = magna_middleware::api::grpc::magna_grpc::InferenceRequest {
            inputs: vec![magna_middleware::api::grpc::magna_grpc::InferInputTensor {
                name: "input".into(),
                datatype: "UNKNOWN".into(),
                shape: vec![1, 2],
                raw_data: vec![0u8; 8],
            }],
            model_name: String::new(),
        };
        let engine = EngineInfo {
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
        };
        let caps = BackendCapabilities {
            backend_name: "qualcomm".into(),
            availability: BackendAvailability::Available,
            max_inputs: 1,
            max_outputs: 1,
            supported_input_precisions: vec![Precision::FP32],
            supported_output_precisions: vec![Precision::FP32],
            supports_engine_building: false,
            supports_dynamic_shapes: false,
        };
        assert!(convert_inference_request(&req, "qualcomm", Some(&engine), Some(&caps)).is_err());
    }
}
