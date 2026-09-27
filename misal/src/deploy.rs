//! Deploy command — upload model artifacts to the Magna server via gRPC.
//!
//! Uses the `UploadModel` streaming RPC to transfer model files to the
//! device running `magna_server`, eliminating the need for SSH/SCP.
//!
//! Configuration:
//! - `MAGNA_SERVER` env var or `--server` flag (default: `http://127.0.0.1:50051`)

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::info;

use magna_middleware::api::grpc::magna_grpc::inference_service_client::InferenceServiceClient;
use magna_middleware::api::grpc::magna_grpc::ModelChunk;

/// Upload chunk size: 1 MB
const CHUNK_SIZE: usize = 1024 * 1024;

#[derive(Parser, Debug)]
pub struct DeployArgs {
    /// Path to the model artifact to deploy (.dlc or .bin).
    #[arg()]
    pub model: PathBuf,

    /// Target subdirectory on the device: "models" or "contexts".
    /// Default: auto-detected from file extension (.bin → contexts, .dlc → models).
    #[arg(long)]
    pub target_dir: Option<String>,

    /// gRPC server address (default: $MAGNA_SERVER or http://127.0.0.1:50051).
    #[arg(short, long)]
    pub server: Option<String>,

    /// Dry run: print what would be uploaded without doing it.
    #[arg(long)]
    pub dry_run: bool,
}

fn resolve_server(args: &DeployArgs) -> String {
    args.server
        .clone()
        .or_else(|| std::env::var("MAGNA_SERVER").ok())
        .unwrap_or_else(|| "http://127.0.0.1:50051".into())
}

fn detect_target_dir(path: &Path) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("bin") => "contexts".into(),
        _ => "models".into(),
    }
}

/// Execute the deploy command.
pub fn run_deploy(args: &DeployArgs) -> Result<()> {
    if !args.model.exists() {
        bail!("Model file does not exist: {}", args.model.display());
    }

    let server_addr = resolve_server(args);
    let target_dir = args
        .target_dir
        .clone()
        .unwrap_or_else(|| detect_target_dir(&args.model));
    let filename = args
        .model
        .file_name()
        .context("Cannot determine model filename")?
        .to_string_lossy()
        .into_owned();
    let file_data = std::fs::read(&args.model).context(format!(
        "Failed to read model file: {}",
        args.model.display()
    ))?;
    let total_size = file_data.len() as u64;

    println!("============================================================");
    println!("  [Magna Deploy] Uploading model to server                  ");
    println!("============================================================");
    println!("  Local file  : {}", args.model.display());
    println!("  Server      : {}", server_addr);
    println!("  Target dir  : {}", target_dir);
    println!("  File size   : {:.2} MB", total_size as f64 / 1_048_576.0);
    println!("============================================================");

    if args.dry_run {
        info!(
            "DRY RUN: Would upload {} ({} bytes) to {}/{}",
            filename, total_size, target_dir, filename
        );
        return Ok(());
    }

    // Build the stream of chunks
    let chunks: Vec<ModelChunk> = file_data
        .chunks(CHUNK_SIZE)
        .enumerate()
        .map(|(i, chunk)| ModelChunk {
            filename: if i == 0 {
                filename.clone()
            } else {
                String::new()
            },
            target_dir: if i == 0 {
                target_dir.clone()
            } else {
                String::new()
            },
            data: chunk.to_vec(),
            total_size: if i == 0 { total_size } else { 0 },
        })
        .collect();

    let num_chunks = chunks.len();
    info!("Uploading {} in {} chunks...", filename, num_chunks);

    // Run the async upload in a blocking context
    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let response = rt.block_on(async {
        let mut client = InferenceServiceClient::connect(server_addr.clone())
            .await
            .context(format!("Failed to connect to server at {}", server_addr))?;

        let stream = tokio_stream::iter(chunks);
        let response = client
            .upload_model(stream)
            .await
            .context("UploadModel RPC failed")?;

        Ok::<_, anyhow::Error>(response.into_inner())
    })?;

    if response.success {
        println!("============================================================");
        println!("  Deploy successful!");
        println!("  Remote path : {}", response.remote_path);
        println!("  Bytes sent  : {}", response.bytes_received);
        println!("============================================================");
    } else {
        bail!("Deploy failed: {}", response.message);
    }

    Ok(())
}
