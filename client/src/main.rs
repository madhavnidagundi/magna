// =============================================================================
// Magna Middleware — gRPC Client (Camera Edge Process)
// =============================================================================
//! Captures frames from a camera and sends them to the gRPC server for
//! inference using the generic tensor proto.

use clap::Parser;
use std::time::Duration;
#[cfg(feature = "camera")]
use std::time::Instant;
use tonic::Request;
use tracing::{error, info, warn};

use magna_middleware::api::grpc::magna_grpc::inference_service_client::InferenceServiceClient;
use magna_middleware::api::grpc::magna_grpc::{
    GetModelInfoRequest, InferInputTensor, InferenceRequest,
};
use magna_middleware::preprocess::imagenet;
use magna_middleware::utils::errors::Precision;

pub mod input;

#[derive(Parser, Debug)]
#[command(
    name = "magna_client",
    about = "Edge Camera gRPC Client for Magna Inference"
)]
struct Cli {
    /// Camera device path (e.g. /dev/video0) or image file path.
    #[arg(short = 'c', long, default_value = "/dev/video0")]
    camera: String,

    /// gRPC server address.
    #[arg(short, long, default_value = "http://127.0.0.1:50051")]
    server: String,

    /// Maximum reconnection attempts before giving up (0 = infinite).
    #[arg(long, default_value = "0")]
    max_reconnects: usize,

    /// Delay between reconnection attempts in seconds.
    #[arg(long, default_value = "3")]
    reconnect_delay: u64,

    /// Enable debug logging.
    #[arg(short, long)]
    debug: bool,

    /// Inference precision (fp32, fp16, int8)
    #[arg(short, long, default_value = "fp32")]
    precision: Precision,

    /// Tensor memory layout (nchw, nhwc)
    #[arg(short = 'l', long, default_value = "nchw")]
    layout: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let default_level = if cli.debug { "debug" } else { "info" };
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .try_init()
        .ok();

    info!(server = %cli.server, camera = %cli.camera, "Magna Edge Camera Client starting");

    let mut reconnect_count = 0;

    // Reconnection loop
    loop {
        match run_inference_loop(&cli).await {
            Ok(()) => {
                info!("[Client] Inference loop completed normally");
                break;
            }
            Err(e) => {
                error!(error = %e, "Connection lost");
                reconnect_count += 1;

                if cli.max_reconnects > 0 && reconnect_count >= cli.max_reconnects {
                    error!(
                        max_reconnects = cli.max_reconnects,
                        "Max reconnection attempts reached — exiting"
                    );
                    return Err(e);
                }

                warn!(
                    delay_secs = cli.reconnect_delay,
                    attempt = reconnect_count,
                    "Reconnecting..."
                );
                tokio::time::sleep(Duration::from_secs(cli.reconnect_delay)).await;
            }
        }
    }

    Ok(())
}

async fn run_inference_loop(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    info!("[Client] Connecting to {}...", cli.server);
    let mut client = InferenceServiceClient::connect(cli.server.clone()).await?;
    info!("[Client] Connected!");

    let model_info = client
        .get_model_info(Request::new(GetModelInfoRequest {}))
        .await?
        .into_inner();
    let input_info = model_info
        .inputs
        .first()
        .ok_or("server returned model metadata without inputs")?
        .clone();
    let input_name = input_info.name;
    let input_precision: Precision = input_info.datatype.parse()?;
    info!(
        backend = %model_info.backend,
        model = %model_info.model_name,
        input = %input_name,
        precision = %input_precision,
        "Discovered loaded model metadata"
    );

    // Determine input source
    let is_file = std::path::Path::new(&cli.camera).extension().is_some();

    if is_file {
        // Single file mode
        info!("[Client] File mode: sending '{}'", cli.camera);

        let img = image::open(&cli.camera)?;
        let rgb = img.to_rgb8();

        let hwc = cli.layout.to_lowercase() == "nhwc";
        let config = magna_middleware::preprocess::imagenet::PreprocessConfig {
            hwc_layout: hwc,
            ..Default::default()
        };

        let tensor = imagenet::preprocess_raw_rgb_with_config(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            input_precision,
            &config,
        )?;

        let req = Request::new(InferenceRequest {
            inputs: vec![InferInputTensor {
                name: input_name.clone(),
                datatype: input_precision.to_string(),
                shape: tensor.shape.iter().map(|&s| s as i64).collect(),
                raw_data: tensor.data,
            }],
            model_name: String::new(),
        });

        let response = client.infer(req).await?;
        let inner = response.into_inner();

        info!(
            "[Client] Received {} generic outputs in {:.1} ms",
            inner.outputs.len(),
            inner.inference_time_ms
        );
        for (i, out) in inner.outputs.iter().enumerate() {
            info!(
                "   Output {}: '{}' ({}) shape={:?} elements={}",
                i,
                out.name,
                out.datatype,
                out.shape,
                (out.raw_data.len() / 4)
            );
        }
    } else {
        // Camera streaming mode
        info!("[Client] Camera mode: streaming from '{}'", cli.camera);

        // Use the camera provider if available
        #[cfg(feature = "camera")]
        {
            use crate::input::camera::CameraInputProvider;
            use crate::input::provider::InputProvider;

            let mut camera = CameraInputProvider::new(&cli.camera)?;
            let mut frame_count = 0u64;
            let start_time = Instant::now();

            loop {
                match camera.next_frame() {
                    Ok(Some(frame)) => {
                        let hwc = cli.layout.to_lowercase() == "nhwc";
                        let config = magna_middleware::preprocess::imagenet::PreprocessConfig {
                            hwc_layout: hwc,
                            ..Default::default()
                        };

                        let tensor = imagenet::preprocess_raw_rgb_with_config(
                            &frame.rgb_data,
                            frame.width,
                            frame.height,
                            input_precision,
                            &config,
                        )?;

                        let req = Request::new(InferenceRequest {
                            inputs: vec![InferInputTensor {
                                name: input_name.clone(),
                                datatype: input_precision.to_string(),
                                shape: tensor.shape.iter().map(|&s| s as i64).collect(),
                                raw_data: tensor.data,
                            }],
                            model_name: String::new(),
                        });

                        match client.infer(req).await {
                            Ok(response) => {
                                let inner = response.into_inner();
                                info!(
                                    "[Client] Frame {}: {} generic outputs {:.1}ms",
                                    frame_count,
                                    inner.outputs.len(),
                                    inner.inference_time_ms
                                );
                            }
                            Err(e) => {
                                error!("[Client] gRPC error: {}", e);
                                return Err(e.into());
                            }
                        }

                        frame_count += 1;
                        if frame_count.is_multiple_of(30) {
                            let fps = frame_count as f64 / start_time.elapsed().as_secs_f64();
                            info!(
                                "[Client] Throughput: {:.1} FPS ({} frames)",
                                fps, frame_count
                            );
                        }
                    }
                    Ok(None) => {
                        info!("[Client] Camera stream ended");
                        break;
                    }
                    Err(e) => {
                        error!("[Client] Camera error: {}", e);
                        return Err(e.into());
                    }
                }
            }
        }

        #[cfg(not(feature = "camera"))]
        {
            error!("[Client] Camera support requires --features camera");
            return Err("Camera feature not enabled".into());
        }
    }

    Ok(())
}
