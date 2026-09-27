mod context;
mod deploy;
mod hardware_target;
mod infer;
mod optimize;

use clap::{Parser, Subcommand};
use std::process::ExitCode;
use tracing::error;

#[derive(Parser, Debug)]
#[command(
    name = "misal",
    about = "Magna Edge AI CLI — Hardware-agnostic model optimization and deployment tools",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Optimize an ONNX model for a specific hardware target
    Optimize(optimize::OptimizeArgs),

    /// Deploy a model artifact to the Radxa board via SCP
    Deploy(deploy::DeployArgs),

    /// Generate a QNN context binary on the Radxa board
    PrepareContext(context::PrepareContextArgs),

    /// Run QNN inference on the Radxa board via SSH
    Infer(infer::InferArgs),
}

fn main() -> ExitCode {
    // Basic tracing setup
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Optimize(args) => match optimize::run_optimize(args) {
            Ok(output) => {
                tracing::info!(
                    "Optimization complete! Output artifact: {}",
                    output.display()
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!("Optimization failed: {:#}", e);
                ExitCode::FAILURE
            }
        },
        Commands::Deploy(args) => match deploy::run_deploy(args) {
            Ok(()) => {
                tracing::info!("Deployment complete!");
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!("Deployment failed: {:#}", e);
                ExitCode::FAILURE
            }
        },
        Commands::PrepareContext(args) => match context::run_prepare_context(args) {
            Ok(()) => {
                tracing::info!("Context preparation complete!");
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!("Context preparation failed: {:#}", e);
                ExitCode::FAILURE
            }
        },
        Commands::Infer(args) => match infer::run_infer(args) {
            Ok(()) => {
                tracing::info!("Inference complete!");
                ExitCode::SUCCESS
            }
            Err(e) => {
                error!("Inference failed: {:#}", e);
                ExitCode::FAILURE
            }
        },
    }
}
