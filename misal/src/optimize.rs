//! Model optimization CLI module for Magna Edge AI.
//!
//! Handles cross-compilation and optimization of ONNX models to target-specific
//! formats like TensorRT engines, QNN/SNPE bins, and RKNN models.

use anyhow::{bail, Context, Result};
use clap::Parser;
#[cfg(feature = "qualcomm")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(any(
    feature = "nvidia",
    feature = "qualcomm",
    feature = "ti",
    feature = "rockchip"
))]
use std::process::Command;
use tracing::info;

use crate::hardware_target::HardwareTarget;
#[cfg(feature = "nvidia")]
use magna_middleware::backends::nvidia::NvidiaHardware;
use magna_middleware::utils::errors::Precision;

#[derive(Parser, Debug)]
pub struct OptimizeArgs {
    /// Path to the input ONNX model.
    #[arg(short = 'm', long)]
    pub model: PathBuf,

    /// Target hardware: ORIN, THOR, SNAPDRAGON, RADXA-Q6A, TDA4, CPU.
    #[arg(long = "hw")]
    pub hardware: HardwareTarget,

    /// Output path for the hardware-optimized model.
    #[arg(short = 'o', long)]
    pub output: PathBuf,

    /// Precision for model conversion: fp32, fp16, int8.
    #[arg(short, long, default_value = "fp16")]
    pub precision: Precision,

    /// Optional path to calibration data directory (required for INT8 on some platforms).
    #[arg(long)]
    pub calibration_data: Option<PathBuf>,

    /// Input tensor name and shape, e.g. --input-dim input:1,3,224,224.
    /// Repeat for models with multiple inputs. Required for Qualcomm targets —
    /// there is no default shape; specify the exact dimensions for your
    /// model/dataset.
    #[arg(long = "input-dim", value_name = "NAME:DIMS")]
    pub input_dim: Vec<String>,

    /// Number of calibration images to use (Qualcomm INT8 only).
    #[arg(long)]
    pub num_images: Option<u32>,

    /// Optional path to calibration cache file (used by TensorRT).
    #[arg(long)]
    pub calibration_cache: Option<PathBuf>,

    /// Dry run: print the build command without executing it.
    #[arg(long)]
    pub dry_run: bool,

    /// Force execution even if cross-architecture portable check fails.
    #[arg(long)]
    pub force: bool,

    /// Enable verbose compiler logging.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Executes the optimization process for a specific hardware target.
///
/// This function verifies input parameters, performs safety checks for architecture
/// portability, and delegates compilation to the correct SDK toolchain.
///
/// # Errors
/// Returns an error if the model path is invalid, if required SDKs are missing,
/// or if the compilation process fails.
pub fn run_optimize(args: &OptimizeArgs) -> Result<PathBuf> {
    let target = args.hardware;
    let precision = args.precision;

    if !args.model.exists() {
        bail!("Input model file does not exist: {}", args.model.display());
    }

    if let Some(parent) = args.output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).context("Failed to create output directory")?;
        }
    }

    // Warn if trying to build a non-portable engine on the wrong machine
    if !target.is_cross_portable() {
        // Check if we appear to be on the right hardware
        if cfg!(target_arch = "x86_64") {
            tracing::warn!(
                "{} engines are not cross-architecture portable.\n\
                 You must run `misal optimize --hw {:?}` on the actual {:?} hardware.\n\
                 Use --dry-run to preview the command.",
                target.build_tool(),
                target,
                target
            );
            if !args.force && !args.dry_run {
                bail!("Run on target hardware or use --force to override");
            }
        }
    }

    match target {
        HardwareTarget::Orin => {
            #[cfg(feature = "nvidia")]
            {
                build_nvidia(args, precision, NvidiaHardware::Orin)
            }
            #[cfg(not(feature = "nvidia"))]
            {
                bail!("Nvidia support not compiled.")
            }
        }
        HardwareTarget::Thor => {
            #[cfg(feature = "nvidia")]
            {
                build_nvidia(args, precision, NvidiaHardware::Thor)
            }
            #[cfg(not(feature = "nvidia"))]
            {
                bail!("Nvidia support not compiled.")
            }
        }
        HardwareTarget::Snapdragon | HardwareTarget::Sa8295 | HardwareTarget::RadxaQ6a => {
            #[cfg(feature = "qualcomm")]
            {
                build_qualcomm(args, precision)
            }
            #[cfg(not(feature = "qualcomm"))]
            {
                bail!("Qualcomm support not compiled.")
            }
        }
        HardwareTarget::Rk3588 => {
            #[cfg(feature = "rockchip")]
            {
                build_rockchip(args, precision)
            }
            #[cfg(not(feature = "rockchip"))]
            {
                bail!("Rockchip support not compiled.")
            }
        }
        HardwareTarget::Tda4 => {
            #[cfg(feature = "ti")]
            {
                build_ti(args, precision)
            }
            #[cfg(not(feature = "ti"))]
            {
                bail!("TI support not compiled.")
            }
        }
        HardwareTarget::Cpu => build_cpu(args, precision),
    }
}

#[cfg(feature = "nvidia")]
fn build_nvidia(args: &OptimizeArgs, precision: Precision, hw: NvidiaHardware) -> Result<PathBuf> {
    info!("Running TensorRT engine compilation via trtexec...");

    let precision_flag = match precision {
        Precision::FP16 => Some("--fp16"),
        Precision::INT8 => Some("--int8"),
        Precision::FP8 => Some("--fp8"),
        Precision::FP32 => None, // explicit no-op
    };

    let mut cmd = Command::new("trtexec");
    cmd.arg(format!("--onnx={}", args.model.display()))
        .arg(format!("--saveEngine={}", args.output.display()))
        .arg(format!(
            "--memPoolSize=workspace:{}M",
            hw.default_workspace_mb()
        ))
        .arg(format!("--tacticSources={}", hw.tactic_sources()));

    if let Some(flag) = precision_flag {
        cmd.arg(flag);
    }

    if precision == Precision::INT8 {
        if let Some(ref cache) = args.calibration_cache {
            cmd.arg(format!("--calib={}", cache.display()));
        } else {
            bail!("INT8 precision for NVIDIA requires --calibration-cache");
        }
    }

    if args.verbose {
        cmd.arg("--verbose");
    }

    if args.dry_run {
        tracing::info!("DRY RUN: {:?}", cmd);
        return Ok(args.output.clone());
    }

    match cmd.status() {
        Ok(status) if status.success() => {
            info!("TensorRT compilation succeeded");
            Ok(args.output.clone())
        }
        Ok(status) => bail!("trtexec exited with error: {:?}", status),
        Err(e) => Err(e).context("Failed to run trtexec"),
    }
}

// ---------------------------------------------------------------------------
// QAIRT SDK discovery
// ---------------------------------------------------------------------------

/// Discovers the QAIRT SDK installation and version from environment variables.
///
/// Returns `(sdk_root_path, version_string)`.
///
/// Resolution order:
/// 1. `QAIRT_SDK_ROOT` (preferred)
/// 2. `QNN_SDK_ROOT` (legacy fallback)
///
/// Version resolution:
/// 1. `QAIRT_VERSION` env var (explicit override)
/// 2. Last path component of the SDK root (auto-detect)
#[cfg(feature = "qualcomm")]
fn discover_qairt_sdk() -> Result<(PathBuf, String)> {
    let sdk_root = std::env::var("QAIRT_SDK_ROOT")
        .or_else(|_| std::env::var("QNN_SDK_ROOT"))
        .context(
            "QAIRT_SDK_ROOT is not set.\n\
             Point it to your QAIRT SDK installation, e.g.:\n\
             export QAIRT_SDK_ROOT=/opt/qairt_workspace/qairt/<version>",
        )?;

    let sdk_path = PathBuf::from(&sdk_root);
    if !sdk_path.exists() {
        bail!("QAIRT_SDK_ROOT={} does not exist", sdk_root);
    }

    // Validate that it looks like an SDK directory
    let bin_dir = sdk_path.join("bin");
    if !bin_dir.exists() {
        bail!(
            "QAIRT_SDK_ROOT={} does not contain a bin/ directory.\n\
             Ensure it points to the SDK root (e.g. .../qairt/<version>)",
            sdk_root
        );
    }

    // Version: explicit env var > dirname auto-detect
    let version = std::env::var("QAIRT_VERSION").unwrap_or_else(|_| {
        sdk_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".into())
    });

    info!(
        sdk_root = %sdk_path.display(),
        version = %version,
        "QAIRT SDK discovered"
    );

    Ok((sdk_path, version))
}

#[cfg(feature = "qualcomm")]
fn build_qualcomm(args: &OptimizeArgs, precision: Precision) -> Result<PathBuf> {
    info!("Converting ONNX model for Qualcomm QNN via Dockerized QAIRT...");

    // The `qualcomm-optimizer` image's entrypoint (build_qnn_int8.py) always runs
    // qairt-converter followed by qairt-quantizer — there is no FP32/FP16-only mode.
    if precision != Precision::INT8 {
        bail!(
            "The Dockerized Qualcomm quantizer only supports --precision int8 \
             right now (it always runs qairt-converter + qairt-quantizer, with \
             no FP-only mode). Requested precision: {}",
            precision.as_str()
        );
    }

    if args.input_dim.is_empty() {
        bail!(
            "Qualcomm targets require at least one --input-dim NAME:DIMS \
             (e.g. --input-dim input:1,3,224,224). There is no default input \
             shape — specify the exact dimensions for your model/dataset."
        );
    }
    let input_dims: Vec<(String, String)> = args
        .input_dim
        .iter()
        .map(|entry| {
            entry
                .split_once(':')
                .map(|(name, dims)| (name.to_string(), dims.to_string()))
                .with_context(|| {
                    format!(
                        "Invalid --input-dim '{}': expected NAME:DIMS (e.g. input:1,3,224,224)",
                        entry
                    )
                })
        })
        .collect::<Result<_>>()?;

    let calib_data = args.calibration_data.as_ref().context(
        "INT8 quantization requires --calibration-data (path to a calibration image/.raw directory)",
    )?;

    let (sdk_path, version) = discover_qairt_sdk()?;
    let current_dir = std::env::current_dir().context("Failed to get current directory")?;

    // Resolve model, output, and calibration-data paths relative to /workspace
    // (the container's mount point for the project directory).
    let to_container_path = |p: &Path| -> PathBuf {
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            current_dir.join(p)
        };
        Path::new("/workspace").join(abs.strip_prefix(&current_dir).unwrap_or(&abs))
    };

    let final_output = args.output.with_extension("dlc");
    let container_model = to_container_path(&args.model);
    let container_output = to_container_path(&final_output);
    let container_dataset = to_container_path(calib_data);

    // Run the container as the host user, not root — otherwise files it
    // writes into the bind-mounted /workspace end up root-owned and unreadable.
    let uid = Command::new("id")
        .arg("-u")
        .output()
        .context("Failed to run `id -u`")?;
    let gid = Command::new("id")
        .arg("-g")
        .output()
        .context("Failed to run `id -g`")?;
    let user_flag = format!(
        "{}:{}",
        String::from_utf8_lossy(&uid.stdout).trim(),
        String::from_utf8_lossy(&gid.stdout).trim()
    );

    // Build the docker command — the image's ENTRYPOINT is `python3
    // build_qnn_int8.py`, so args are passed directly (no shell wrapper).
    let mut cmd = Command::new("docker");
    cmd.args([
        "run",
        "--rm",
        "-v",
        &format!("{}:/workspace", current_dir.display()),
        "-v",
        &format!("{}:/opt/qairt", sdk_path.display()),
        "-w",
        "/workspace",
        "--user",
        &user_flag,
        "qualcomm-optimizer",
        "--onnx",
    ])
    .arg(container_model.as_os_str())
    .args(["--dataset"])
    .arg(container_dataset.as_os_str())
    .args(["--output_dlc"])
    .arg(container_output.as_os_str());

    if let Some(n) = args.num_images {
        cmd.args(["--num_images", &n.to_string()]);
    }
    for (name, dims) in &input_dims {
        cmd.args(["--input_dim", name, dims]);
    }

    println!("============================================================");
    println!("  [Magna AOT] Compiling ONNX to QNN (QAIRT)                 ");
    println!("============================================================");
    println!("  Input ONNX  : {}", args.model.display());
    println!("  Precision   : {}", precision.as_str());
    println!("  Output      : {}", final_output.display());
    println!("  SDK Root    : {}", sdk_path.display());
    println!("  SDK Version : {}", version);
    println!("============================================================");

    if args.dry_run {
        tracing::info!("DRY RUN (Docker QAIRT): {:?}", cmd);
        return Ok(final_output);
    }

    let status = cmd
        .status()
        .context("Failed to run Dockerized QAIRT conversion. Is Docker running?")?;
    if !status.success() {
        bail!(
            "Dockerized QAIRT conversion exited with error: {:?}",
            status
        );
    }

    info!("QNN model conversion succeeded (DLC generated)");
    Ok(final_output)
}

#[cfg(feature = "ti")]
fn build_ti(args: &OptimizeArgs, _precision: Precision) -> Result<PathBuf> {
    info!("Converting ONNX model for TI TIDL...");

    if let Ok(tidl_path) = std::env::var("TIDL_TOOLS_PATH") {
        let compiler = format!("{}/tidl_model_import.out", tidl_path);

        let mut cmd = Command::new(&compiler);
        cmd.arg(&args.model).arg(&args.output);

        if args.dry_run {
            tracing::info!("DRY RUN: {:?}", cmd);
            return Ok(args.output.clone());
        }

        return match cmd.status() {
            Ok(status) if status.success() => {
                info!("TIDL model compilation succeeded");
                Ok(args.output.clone())
            }
            Ok(status) => bail!("TIDL compiler exited with error: {:?}", status),
            Err(e) => Err(e).context("Failed to run TIDL compiler"),
        };
    }

    bail!("TIDL_TOOLS_PATH not set. Cannot convert model for TDA4.");
}

#[cfg(feature = "rockchip")]
fn build_rockchip(args: &OptimizeArgs, precision: Precision) -> Result<PathBuf> {
    info!("Converting ONNX model for Rockchip RKNN...");
    // rknn_convert is a Python CLI from rknn-toolkit2
    let rknn_convert = std::env::var("RKNN_TOOLKIT_PATH")
        .map(|p| format!("{}/rknn_convert", p))
        .unwrap_or_else(|_| "rknn_convert".into());

    let output = args.output.with_extension("rknn");

    let mut cmd = Command::new(&rknn_convert);
    cmd.arg("--model")
        .arg(&args.model)
        .arg("--output")
        .arg(&output)
        .arg("--target")
        .arg("RK3588");

    // INT8 quantization requires calibration data
    if precision == Precision::INT8 {
        if let Some(ref calib) = args.calibration_data {
            cmd.arg("--quantization")
                .arg("i8")
                .arg("--dataset")
                .arg(calib);
        } else {
            bail!("INT8 quantization for RK3588 requires --calibration-data");
        }
    }

    if args.dry_run {
        tracing::info!("DRY RUN: {:?}", cmd);
        return Ok(output);
    }

    match cmd.status() {
        Ok(status) if status.success() => {
            info!("RKNN model conversion succeeded");
            Ok(output)
        }
        Ok(status) => bail!("rknn_convert exited with error: {:?}", status),
        Err(e) => Err(e).context("Failed to run rknn_convert"),
    }
}

fn build_cpu(args: &OptimizeArgs, _precision: Precision) -> Result<PathBuf> {
    info!("CPU mode — copying ONNX model as-is (no conversion needed)");

    if args.dry_run {
        tracing::info!(
            "DRY RUN: std::fs::copy({:?}, {:?})",
            args.model,
            args.output
        );
        return Ok(args.output.clone());
    }

    match std::fs::copy(&args.model, &args.output) {
        Ok(_) => Ok(args.output.clone()),
        Err(e) => Err(e).context("Failed to copy model"),
    }
}
