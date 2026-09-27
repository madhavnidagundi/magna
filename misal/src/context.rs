//! Context binary generation command via gRPC.
//!
//! Sends a `PrepareContext` RPC to the Magna server running on the device,
//! which triggers on-device `qnn-context-binary-generator` execution.
//!
//! Configuration:
//! - `MAGNA_SERVER` env var or `--server` flag (default: `http://127.0.0.1:50051`)

use anyhow::{bail, Context, Result};
use clap::Parser;
use tracing::info;

use magna_middleware::api::grpc::magna_grpc::inference_service_client::InferenceServiceClient;
use magna_middleware::api::grpc::magna_grpc::PrepareContextRequest;

#[derive(Parser, Debug)]
pub struct PrepareContextArgs {
    /// Name of the DLC model already deployed to the device (e.g. mobilenetv2_int8.dlc).
    #[arg()]
    pub model: String,

    /// gRPC server address (default: $MAGNA_SERVER or http://127.0.0.1:50051).
    #[arg(short, long)]
    pub server: Option<String>,

    /// Force regeneration even if context binary already exists.
    #[arg(long)]
    pub force: bool,

    /// Dry run: print the RPC that would be sent without executing it.
    #[arg(long)]
    pub dry_run: bool,
}

fn resolve_server(args: &PrepareContextArgs) -> String {
    args.server
        .clone()
        .or_else(|| std::env::var("MAGNA_SERVER").ok())
        .unwrap_or_else(|| "http://127.0.0.1:50051".into())
}

/// Execute the prepare-context command.
pub fn run_prepare_context(args: &PrepareContextArgs) -> Result<()> {
    let server_addr = resolve_server(args);
    let dlc_filename = std::path::Path::new(&args.model)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| args.model.clone());

    println!("============================================================");
    println!("  [Magna Context] Generating QNN context binary             ");
    println!("============================================================");
    println!("  Server      : {}", server_addr);
    println!("  DLC model   : {}", dlc_filename);
    println!("  Force regen : {}", args.force);
    println!("============================================================");

    if args.dry_run {
        info!(
            "DRY RUN: Would send PrepareContext RPC for '{}' (force={})",
            dlc_filename, args.force
        );
        return Ok(());
    }

    let rt = tokio::runtime::Runtime::new().context("Failed to create tokio runtime")?;
    let response = rt.block_on(async {
        let mut client = InferenceServiceClient::connect(server_addr.clone())
            .await
            .context(format!("Failed to connect to server at {}", server_addr))?;

        let request = tonic::Request::new(PrepareContextRequest {
            dlc_filename: dlc_filename.clone(),
            force: args.force,
        });

        let response = client
            .prepare_context(request)
            .await
            .context("PrepareContext RPC failed")?;

        Ok::<_, anyhow::Error>(response.into_inner())
    })?;

    if response.success {
        println!("============================================================");
        println!("  Context binary ready!");
        println!("  Device path : {}", response.context_path);
        println!("  {}", response.message);
        println!("============================================================");
    } else {
        bail!("Context generation failed: {}", response.message);
    }

    Ok(())
}
