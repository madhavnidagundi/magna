//! Remote inference command via gRPC.
//!
//! Sends inference requests to the Magna server running on the device.
//! The server handles model loading and QNN execution locally.
//!
//! Configuration:
//! - `MAGNA_SERVER` env var or `--server` flag (default: `http://127.0.0.1:50051`)

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tracing::info;

use magna_middleware::api::grpc::magna_grpc::inference_service_client::InferenceServiceClient;
use magna_middleware::api::grpc::magna_grpc::{
    GetModelInfoRequest, InferInputTensor, InferenceRequest, ListModelsRequest,
};

#[derive(Parser, Debug)]
pub struct InferArgs {
    /// Input data file (.raw binary tensor) to send for inference.
    #[arg()]
    pub input: PathBuf,

    /// Model name to use for inference (optional — server uses loaded model by default).
    #[arg(long)]
    pub model: Option<String>,

    /// gRPC server address (default: $MAGNA_SERVER or http://127.0.0.1:50051).
    #[arg(short, long)]
    pub server: Option<String>,

    /// Input tensor shape (comma-separated, e.g. "1,3,224,224"). Required —
    /// there is no default shape; specify the exact dimensions for your model.
    #[arg(long)]
    pub shape: String,

    /// Input tensor precision: FP32, FP16, INT8. Ignored (forced to INT8) if
    /// --quant-scale/--quant-offset are given.
    #[arg(long, default_value = "FP32")]
    pub precision: String,

    /// Quantization scale for converting a float32 .raw input to uint8 before
    /// sending (q = round(x / scale) - offset, clamped to [0, 255]). Query
    /// this from your model's actual input encoding (e.g. via
    /// `qairt-dlc-info`) — there is no default; every model's calibration is
    /// different. Must be given together with --quant-offset.
    #[arg(long)]
    pub quant_scale: Option<f32>,

    /// Quantization offset — see --quant-scale. Must be given together with
    /// --quant-scale.
    #[arg(long)]
    pub quant_offset: Option<f32>,

    /// Save output tensor to file.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// List available models on the server instead of running inference.
    #[arg(long)]
    pub list_models: bool,

    /// Dry run: print the RPC that would be sent without executing it.
    #[arg(long)]
    pub dry_run: bool,
}

fn resolve_server(args: &InferArgs) -> String {
    args.server
        .clone()
        .or_else(|| std::env::var("MAGNA_SERVER").ok())
        .unwrap_or_else(|| "http://127.0.0.1:50051".into())
}

fn parse_shape(shape_str: &str) -> Result<Vec<i64>> {
    shape_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<i64>()
                .context(format!("Invalid shape dimension: '{}'", s))
        })
        .collect()
}

/// Execute the infer command.
pub fn run_infer(args: &InferArgs) -> Result<()> {
    let server_addr = resolve_server(args);

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;

    // List models mode
    if args.list_models {
        return rt.block_on(async {
            let mut client = InferenceServiceClient::connect(server_addr.clone())
                .await
                .context(format!("Failed to connect to server at {}", server_addr))?;

            let response = client
                .list_models(tonic::Request::new(ListModelsRequest {}))
                .await
                .context("ListModels RPC failed")?;

            let models = response.into_inner().models;
            println!("============================================================");
            println!("  Models available on server ({}):", server_addr);
            println!("============================================================");
            if models.is_empty() {
                println!("  (none)");
            }
            for m in &models {
                println!(
                    "  {} ({}) — {:.2} MB",
                    m.filename,
                    m.model_type,
                    m.size_bytes as f64 / 1_048_576.0
                );
            }
            println!("============================================================");
            Ok(())
        });
    }

    // Inference mode
    if !args.input.exists() {
        bail!("Input file does not exist: {}", args.input.display());
    }

    let cli_shape = parse_shape(&args.shape)?;
    let mut input_data =
        std::fs::read(&args.input).context(format!("Failed to read {}", args.input.display()))?;

    let precision = match (args.quant_scale, args.quant_offset) {
        (Some(scale), Some(offset)) => {
            if input_data.len() % 4 != 0 {
                bail!(
                    "Input file size ({} bytes) is not a multiple of 4 — expected \
                     float32 data for --quant-scale/--quant-offset conversion",
                    input_data.len()
                );
            }
            let quantized: Vec<u8> = input_data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| {
                    let x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    let q = (x / scale).round() - offset;
                    q.clamp(0.0, 255.0) as u8
                })
                .collect();
            info!(
                input_bytes = input_data.len(),
                output_bytes = quantized.len(),
                scale,
                offset,
                "Quantized float32 input to uint8"
            );
            input_data = quantized;
            "INT8".to_string()
        }
        (None, None) => args.precision.clone(),
        _ => bail!("--quant-scale and --quant-offset must be given together"),
    };

    println!("============================================================");
    println!("  [Magna Infer] Running inference via gRPC                  ");
    println!("============================================================");
    println!("  Server      : {}", server_addr);
    println!("  Input file  : {}", args.input.display());
    println!("  Shape       : {:?}", cli_shape);
    println!("  Precision   : {}", precision);
    println!(
        "  Input size  : {:.2} MB",
        input_data.len() as f64 / 1_048_576.0
    );
    println!("============================================================");

    if args.dry_run {
        info!(
            "DRY RUN: Would send {} bytes to server for inference",
            input_data.len()
        );
        return Ok(());
    }

    let response = rt.block_on(async {
        let mut client = InferenceServiceClient::connect(server_addr.clone())
            .await
            .context(format!("Failed to connect to server at {}", server_addr))?;

        let model_info = client
            .get_model_info(tonic::Request::new(GetModelInfoRequest {}))
            .await
            .context("GetModelInfo RPC failed")?
            .into_inner();
        let input_info = model_info
            .inputs
            .first()
            .context("server returned model metadata without inputs")?;
        let input_name = input_info.name.clone();
        let request_shape = if input_info.shape.is_empty() {
            cli_shape.clone()
        } else {
            input_info.shape.clone()
        };
        let request_precision = match (args.quant_scale, args.quant_offset) {
            (Some(_), Some(_)) => precision.clone(),
            _ => input_info.datatype.clone(),
        };
        info!(
            backend = %model_info.backend,
            model = %model_info.model_name,
            input = %input_name,
            precision = %request_precision,
            "Discovered loaded model metadata"
        );

        let request = tonic::Request::new(InferenceRequest {
            inputs: vec![InferInputTensor {
                name: input_name,
                datatype: request_precision,
                shape: request_shape,
                raw_data: input_data,
            }],
            model_name: args.model.clone().unwrap_or_default(),
        });

        let response = client
            .infer(request)
            .await
            .context("Inference RPC failed")?;

        Ok::<_, anyhow::Error>(response.into_inner())
    })?;

    println!("============================================================");
    println!(
        "  Inference complete! ({:.2} ms)",
        response.inference_time_ms
    );
    println!("  Outputs: {}", response.outputs.len());
    for (i, out) in response.outputs.iter().enumerate() {
        println!(
            "    [{i}] '{}' ({}) shape={:?} — {} bytes",
            out.name,
            out.datatype,
            out.shape,
            out.raw_data.len()
        );
    }
    println!("============================================================");

    // Save output if requested
    if let Some(ref output_path) = args.output {
        if let Some(first_output) = response.outputs.first() {
            std::fs::write(output_path, &first_output.raw_data).context(format!(
                "Failed to write output to {}",
                output_path.display()
            ))?;
            info!("Output saved to: {}", output_path.display());
        }
    }

    Ok(())
}
