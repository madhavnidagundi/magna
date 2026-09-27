// =============================================================================
// Magna Middleware — Labeled Dataset Accuracy Evaluation (in-process, OpenTelemetry)
// =============================================================================
//! Runs a labeled evaluation set through the middleware directly in-process
//! (no gRPC hop, no Python) and reports accuracy as custom OpenTelemetry
//! metrics on the same Prometheus endpoint `magna_server` already uses,
//! alongside the generic `inference_requests_total` /
//! `inference_latency_milliseconds` every backend gets automatically.
//!
//! Works with any model and any labeled single-label classification dataset:
//! input shape, input/output tensor names, and quantization parameters are
//! all read from the loaded engine or supplied on the command line — nothing
//! is hardcoded to a specific model or dataset.
//!
//! `magna_server` must NOT be running against the same model at the same
//! time on backends that only allow one process to hold a device session at
//! once (e.g. the Qualcomm HTP backend — see RADXA_NEXT_STEPS.md's "wedged
//! DSP session" note).
//!
//! Usage:
//! ```bash
//! eval_50k --model generated_ctx/mobilenetv4_medium_native.bin \
//!     --raw-dir eval_data/raw --labels eval_data/labels.txt \
//!     --shape 1,3,224,224 --quant-scale 0.018658448011 --quant-offset -114.0
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use clap::Parser;
use std::time::Instant;
use tracing::{error, info, warn};

use magna_middleware::api::public_api::Middleware;
use magna_middleware::inference::traits::TensorBuffer;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};
use magna_middleware::utils::logging::{init_logger, LogConfig, LogFormat};

#[derive(Parser, Debug)]
#[command(
    name = "eval_50k",
    about = "Magna labeled-dataset accuracy evaluation — in-process, OpenTelemetry-instrumented"
)]
struct Cli {
    /// Path to the compiled engine (context .bin, .dlc, .onnx, .engine, etc.
    /// — whatever the target backend expects).
    #[arg(short, long)]
    model: String,

    /// Directory containing eval_XXXXX.raw files.
    #[arg(long, default_value = "eval_data/raw")]
    raw_dir: String,

    /// Path to labels.txt (one ground-truth class index per line).
    #[arg(long, default_value = "eval_data/labels.txt")]
    labels: String,

    /// Number of images to evaluate (0 = all in labels.txt).
    #[arg(long, default_value_t = 0)]
    num_images: usize,

    /// Zero-padding width for eval_XXXXX.raw filenames.
    #[arg(long, default_value_t = 5)]
    index_width: usize,

    /// Starting index (for partial/resumed runs).
    #[arg(long, default_value_t = 0)]
    start_index: usize,

    /// Log progress every N images.
    #[arg(long, default_value_t = 500)]
    log_interval: usize,

    /// Keep the process (and its Prometheus /metrics endpoint) alive after
    /// the evaluation loop finishes, until Ctrl-C.
    #[arg(long, default_value_t = true)]
    keep_alive: bool,

    /// Input tensor shape (comma-separated, e.g. "1,3,224,224"). If omitted,
    /// the shape reported by the loaded engine's first input is used.
    #[arg(long)]
    shape: Option<String>,

    /// Precision of the raw input files on disk, before any
    /// --quant-scale/--quant-offset conversion. Ignored (forced to INT8) if
    /// --quant-scale/--quant-offset are given.
    #[arg(long, value_enum, default_value_t = Precision::FP32)]
    precision: Precision,

    /// Quantization scale for converting a float32 .raw input to uint8
    /// before inference (q = round(x / scale) - offset, clamped to
    /// [0, 255]). Query this from your model's actual input encoding (e.g.
    /// via `qairt-dlc-info`) — there is no default; every model's
    /// calibration is different. Must be given together with
    /// --quant-offset.
    #[arg(long)]
    quant_scale: Option<f32>,

    /// Quantization offset — see --quant-scale. Must be given together with
    /// --quant-scale.
    #[arg(long)]
    quant_offset: Option<f32>,

    /// Free-text label identifying the backend/device under test, attached
    /// to telemetry as the `device` resource attribute (e.g. "qualcomm",
    /// "nvidia", "cpu").
    #[arg(long, default_value = "middleware")]
    device_label: String,
}

fn parse_shape(shape_str: &str) -> Result<Vec<usize>, String> {
    shape_str
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid shape dimension: '{}'", s))
        })
        .collect()
}

/// float32 NCHW raw bytes -> uint8 quantized bytes: q = round(x / scale) - offset, clipped to [0, 255].
fn quantize(float_bytes: &[u8], scale: f32, offset: f32) -> Vec<u8> {
    float_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            let x = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            let q = (x / scale).round() - offset;
            q.clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Ranks the output tensor and checks whether `gt_class` is the top prediction /
/// within the top 5. Decodes FP32 and FP16 outputs to real float scores; for
/// INT8/FP8 outputs, ranks on the raw quantized bytes directly — valid because
/// dequantization is a positive-scale affine transform applied uniformly
/// across all classes, so it doesn't change the ranking.
fn top_k_correct(output: &TensorBuffer, gt_class: usize) -> (bool, bool) {
    let scores: Vec<f32> = match output.precision {
        Precision::FP32 => output
            .try_as_f32_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default(),
        Precision::FP16 => output
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Precision::INT8 | Precision::FP8 => output.data.iter().map(|&b| b as f32).collect(),
    };

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top1 = indexed.first().map(|(i, _)| *i) == Some(gt_class);
    let top5 = indexed.iter().take(5).any(|(i, _)| *i == gt_class);
    (top1, top5)
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    init_logger(LogConfig {
        format: LogFormat::Pretty,
        default_level: "info".into(),
    });

    let labels_content = std::fs::read_to_string(&cli.labels).unwrap_or_else(|e| {
        error!(path = %cli.labels, error = %e, "Failed to read labels file");
        std::process::exit(1);
    });
    let gt_labels: Vec<usize> = labels_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap_or(0))
        .collect();

    let num_images = if cli.num_images == 0 {
        gt_labels.len()
    } else {
        cli.num_images.min(gt_labels.len())
    };

    let mw = Middleware::new();
    let config = MiddlewareConfig {
        fallback_precision: Precision::INT8,
        model_path: Some(cli.model.clone()),
        ..Default::default()
    };
    if let Err(e) = mw.initialize(config) {
        error!(error = %e, "Failed to initialize middleware");
        std::process::exit(1);
    }

    #[cfg(feature = "opentelemetry")]
    let (
        mw,
        telemetry_provider,
        telemetry_server,
        images_total,
        top1_correct_total,
        top5_correct_total,
        top1_ratio_gauge,
        top5_ratio_gauge,
    ) = {
        let (provider, server) = magna_middleware::telemetry::init_telemetry(
            &cli.device_label,
            "magna-eval-50k",
            env!("CARGO_PKG_VERSION"),
        );
        let images_total = provider
            .meter
            .u64_counter("eval_images_total")
            .with_description("Total images evaluated in the accuracy run")
            .build();
        let top1_correct_total = provider
            .meter
            .u64_counter("eval_top1_correct_total")
            .with_description("Images with a correct top-1 prediction")
            .build();
        let top5_correct_total = provider
            .meter
            .u64_counter("eval_top5_correct_total")
            .with_description("Images with a correct top-5 prediction")
            .build();
        let top1_ratio_gauge = provider
            .meter
            .f64_gauge("eval_top1_accuracy_ratio")
            .with_description("Running top-1 accuracy ratio (0.0-1.0)")
            .build();
        let top5_ratio_gauge = provider
            .meter
            .f64_gauge("eval_top5_accuracy_ratio")
            .with_description("Running top-5 accuracy ratio (0.0-1.0)")
            .build();
        let mw = mw.with_telemetry(provider.clone());
        (
            mw,
            provider,
            server,
            images_total,
            top1_correct_total,
            top5_correct_total,
            top1_ratio_gauge,
            top5_ratio_gauge,
        )
    };

    let info = match mw.load_engine(&cli.model) {
        Ok(info) => {
            info!(
                engine = %info.name,
                inputs = info.inputs.len(),
                outputs = info.outputs.len(),
                "Engine loaded"
            );
            info
        }
        Err(e) => {
            error!(error = %e, "Failed to load engine");
            std::process::exit(1);
        }
    };

    let input_name = info
        .inputs
        .first()
        .map(|t| t.name.clone())
        .unwrap_or_else(|| "input".to_string());

    let shape = match cli.shape.as_deref() {
        Some(s) => match parse_shape(s) {
            Ok(shape) => shape,
            Err(e) => {
                error!(error = %e, "Invalid --shape");
                std::process::exit(1);
            }
        },
        None => match info.inputs.first().map(|t| t.shape.clone()) {
            Some(shape) if !shape.is_empty() => {
                info!(?shape, "Using input shape reported by loaded engine");
                shape
            }
            _ => {
                error!(
                    "Engine did not report an input shape — pass --shape explicitly (e.g. --shape 1,3,224,224)"
                );
                std::process::exit(1);
            }
        },
    };

    info!(num_images, raw_dir = %cli.raw_dir, "Starting evaluation");
    let overall_start = Instant::now();

    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut total = 0usize;

    for (i, &gt_class) in gt_labels
        .iter()
        .enumerate()
        .take(num_images)
        .skip(cli.start_index)
    {
        let raw_path = format!(
            "{}/eval_{:0width$}.raw",
            cli.raw_dir,
            i,
            width = cli.index_width
        );
        let raw_bytes = match std::fs::read(&raw_path) {
            Ok(b) => b,
            Err(e) => {
                warn!(path = %raw_path, error = %e, "Missing raw file, stopping early");
                break;
            }
        };

        let (data, input_precision) = match (cli.quant_scale, cli.quant_offset) {
            (Some(scale), Some(offset)) => {
                if raw_bytes.len() % 4 != 0 {
                    error!(
                        bytes = raw_bytes.len(),
                        "Input file size is not a multiple of 4 — expected float32 data for --quant-scale/--quant-offset conversion"
                    );
                    std::process::exit(1);
                }
                (quantize(&raw_bytes, scale, offset), Precision::INT8)
            }
            (None, None) => (raw_bytes, cli.precision),
            _ => {
                error!("--quant-scale and --quant-offset must be given together");
                std::process::exit(1);
            }
        };

        let input = TensorBuffer {
            name: input_name.clone(),
            data,
            shape: shape.clone(),
            precision: input_precision,
        };

        let output = match mw.infer(&[input]) {
            Ok(o) => o,
            Err(e) => {
                warn!(index = i, error = %e, "Inference failed, skipping image");
                continue;
            }
        };
        if output.is_empty() {
            continue;
        }

        let (is_top1, is_top5) = top_k_correct(&output[0], gt_class);

        total += 1;
        if is_top1 {
            top1 += 1;
        }
        if is_top5 {
            top5 += 1;
        }

        #[cfg(feature = "opentelemetry")]
        {
            images_total.add(1, &[]);
            if is_top1 {
                top1_correct_total.add(1, &[]);
            }
            if is_top5 {
                top5_correct_total.add(1, &[]);
            }
            top1_ratio_gauge.record(top1 as f64 / total as f64, &[]);
            top5_ratio_gauge.record(top5 as f64 / total as f64, &[]);
        }

        if cli.log_interval != 0 && total.is_multiple_of(cli.log_interval) {
            info!(
                progress = total,
                top1_accuracy = format!("{:.2}%", 100.0 * top1 as f64 / total as f64),
                top5_accuracy = format!("{:.2}%", 100.0 * top5 as f64 / total as f64),
                "Evaluation progress"
            );
        }
    }

    let elapsed = overall_start.elapsed();
    let total_f = total.max(1) as f64;
    info!(
        total,
        top1_accuracy = format!("{:.2}%", 100.0 * top1 as f64 / total_f),
        top5_accuracy = format!("{:.2}%", 100.0 * top5 as f64 / total_f),
        elapsed_secs = elapsed.as_secs_f64(),
        "Evaluation complete"
    );

    println!("\n=== Evaluation Result ===");
    println!("Total images: {}", total);
    println!("Top-1 accuracy: {:.2}%", 100.0 * top1 as f64 / total_f);
    println!("Top-5 accuracy: {:.2}%", 100.0 * top5 as f64 / total_f);
    println!(
        "Elapsed: {:.1}s ({:.2} img/s)",
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64().max(0.001)
    );

    #[cfg(feature = "opentelemetry")]
    {
        if cli.keep_alive {
            info!(
                "Metrics remain available at http://<host>:9090/metrics \
                 (eval_images_total, eval_top1_correct_total, eval_top5_correct_total, \
                 eval_top1_accuracy_ratio, eval_top5_accuracy_ratio) until Ctrl-C."
            );
            tokio::signal::ctrl_c().await.ok();
        }
        if let Some(server) = telemetry_server {
            server.shutdown().await;
        } else {
            telemetry_provider.shutdown();
        }
    }

    mw.shutdown().ok();
}
