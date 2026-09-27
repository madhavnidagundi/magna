// =============================================================================
// Magna Middleware ΓÇö CLI Entry Point
// =============================================================================
//! Binary entry point for direct model inference from the command line.
//!
//! Usage:
//! ```bash
//! magna --model model.engine --image cat.jpg
//! magna --model model.engine --image cat.jpg --labels labels.txt
//! magna --model model.engine --image cat.jpg --backend simulated
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::manual_is_multiple_of)]

use clap::Parser;
use std::time::Instant;
use tracing::{error, info};

use magna_middleware::api::public_api::Middleware;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "magna", about = "Magna Edge AI Middleware ΓÇö Inference CLI")]
struct Cli {
    /// Path to the compiled engine / model file.
    #[arg(short, long)]
    model: Option<String>,

    /// Path to an image file for inference.
    #[arg(short, long)]
    image: Option<String>,

    /// Path to a directory of images for batch inference.
    #[arg(long)]
    image_dir: Option<String>,

    /// Path to a JSON file mapping synsets to class IDs (for accuracy).
    #[arg(long)]
    synset_mapping: Option<String>,

    /// Path to an ONNX model to build into a hardware-optimized engine.
    #[arg(long)]
    build: Option<String>,

    /// Output path for the built engine (used with --build).
    #[arg(short, long)]
    output: Option<String>,

    /// Path to an INT8 calibration cache (optional, used with --build).
    #[arg(long)]
    calib: Option<String>,

    /// Precision: fp32, fp16, int8, fp8.
    #[arg(short, long, default_value = "fp32")]
    precision: String,

    /// Path to labels file (one class per line).
    #[arg(short, long)]
    labels: Option<String>,

    /// Enable debug logging.
    #[arg(short, long)]
    debug: bool,

    /// Log output format: pretty, json, compact.
    #[arg(long, default_value = "pretty")]
    log_format: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
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

    let precision: Precision = match cli.precision.parse() {
        Ok(p) => p,
        Err(e) => {
            error!(precision = %cli.precision, error = %e, "Invalid precision");
            std::process::exit(1);
        }
    };

    let mw = Middleware::new();

    // Initialize
    let config = MiddlewareConfig {
        fallback_precision: precision,
        warmup_runs: 0,
        debug: cli.debug,
        labels_path: cli.labels.clone(),
        model_path: cli.model.clone(),
        grpc_address: String::new(),
    };

    if let Err(e) = mw.initialize(config) {
        error!(error = %e, "Failed to initialize middleware");
        std::process::exit(1);
    }

    // Handle Engine Building
    if let Some(onnx_path) = cli.build {
        let engine_path = cli
            .output
            .unwrap_or_else(|| onnx_path.replace(".onnx", ".engine"));

        info!(onnx = %onnx_path, engine = %engine_path, "Starting AOT engine build...");
        if let Err(e) = mw.build_engine(&onnx_path, &engine_path, precision, cli.calib.as_deref()) {
            error!(error = %e, "Engine build failed");
            std::process::exit(1);
        }
        info!("Engine build successful!");
        std::process::exit(0);
    }

    // Load engine
    let model_path = match cli.model.as_ref() {
        Some(p) => p,
        None => {
            error!("--model is required unless --build is used");
            std::process::exit(1);
        }
    };
    match mw.load_engine(model_path) {
        Ok(info) => {
            info!(
                engine       = %info.name,
                inputs       = info.inputs.len(),
                outputs      = info.outputs.len(),
                memory_bytes = info.memory_bytes,
                "Engine loaded"
            );
        }
        Err(e) => {
            error!(engine_path = %model_path, error = %e, "Failed to load engine");
            std::process::exit(1);
        }
    }

    // -----------------------------------------------------------------------
    // Image Selection
    // -----------------------------------------------------------------------
    let mut image_paths = Vec::new();

    if let Some(img) = &cli.image {
        image_paths.push(img.clone());
    } else if let Some(dir) = &cli.image_dir {
        println!(">>> Scanning directory: {}", dir);
        for entry in walkdir::WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy().to_lowercase();
                if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                    image_paths.push(path.to_string_lossy().into_owned());
                }
            }
        }
        println!(">>> Found {} images", image_paths.len());
    } else {
        error!("Either --image or --image-dir must be specified");
        std::process::exit(1);
    }

    if image_paths.is_empty() {
        error!("No valid images found for inference");
        std::process::exit(1);
    }

    // Accuracy tracking
    let synset_map: Option<std::collections::HashMap<String, usize>> =
        cli.synset_mapping.as_ref().map(|path| {
            let content =
                std::fs::read_to_string(path).expect("Failed to read synset mapping file");
            serde_json::from_str(&content).expect("Failed to parse synset mapping JSON")
        });

    let start = Instant::now();
    let processed = AtomicUsize::new(0);
    let top1_correct = AtomicUsize::new(0);
    let had_error = AtomicBool::new(false);

    let total_images = image_paths.len();

    // Parallel inference iteration
    image_paths
        .into_par_iter()
        .enumerate()
        .for_each(|(i, img_path_str)| {
            let img_path = std::path::Path::new(&img_path_str);

            match mw.infer_from_image(&img_path_str) {
                Ok(result) => {
                    let p = processed.fetch_add(1, Ordering::SeqCst) + 1;

                    // Accuracy calculation
                    if let (Some(map), Some(cls)) = (&synset_map, &result.classification) {
                        if let Some(parent) = img_path.parent().and_then(|p| p.file_name()) {
                            let synset = parent.to_string_lossy().to_string();
                            if let Some(&gt_id) = map.get(&synset) {
                                if !cls.top5.is_empty() && cls.top5[0].2 == gt_id {
                                    top1_correct.fetch_add(1, Ordering::SeqCst);
                                }
                            }
                        }
                    }

                    if total_images == 1
                        || i == total_images - 1
                        || (p % 100 == 0 && cli.image_dir.is_some())
                    {
                        if cli.image_dir.is_some() {
                            let acc =
                                (top1_correct.load(Ordering::SeqCst) as f32 / p as f32) * 100.0;
                            println!(
                                "    {}/{} images processed | Top-1 Accuracy: {:.2}%...",
                                p, total_images, acc
                            );
                        } else {
                            println!("\n=== Inference Result ===");
                            println!("{}", result);

                            if let Some(cls) = &result.classification {
                                println!("Top-1: {} ({:.4})", cls.top1_label, cls.top1_score);
                                for (label, score, idx) in &cls.top5 {
                                    println!("  {} ({:.4}) [{}]", label, score, idx);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!(image = %img_path_str, error = %e, "Inference failed");
                    had_error.store(true, Ordering::SeqCst);
                }
            }
        });

    let elapsed = start.elapsed();
    let total_processed = processed.load(Ordering::SeqCst);
    let total_correct = top1_correct.load(Ordering::SeqCst);

    if had_error.load(Ordering::SeqCst) && cli.image_dir.is_none() {
        std::process::exit(1);
    }

    if total_processed > 1 {
        let accuracy = (total_correct as f64 / total_processed as f64) * 100.0;
        println!("\n=== Final Results ===");
        println!("  Images Processed: {}", total_processed);
        if synset_map.is_some() {
            println!("  Top-1 Accuracy:   {:.2}%", accuracy);
        }
        println!("  Total time: {:.1}ms", elapsed.as_secs_f64() * 1000.0);
    } else {
        println!("\nInference time: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    }

    // Metrics (only available when compiled with --features metrics)
    #[cfg(feature = "metrics")]
    {
        let metrics = mw.get_metrics();
        println!("\n=== Metrics ===");
        println!("  Total inferences: {}", metrics.inference_count);
        println!("  Avg latency:      {:.3}ms", metrics.avg_latency_ms);
        println!("  Min latency:      {:.3}ms", metrics.min_latency_ms);
        println!("  Max latency:      {:.3}ms", metrics.max_latency_ms);
    }

    mw.shutdown().ok();
}
