// =============================================================================
// Magna Middleware — Dependency Manager
// =============================================================================
//! Checks that required platform-specific dependencies are present before
//! building or running the middleware.
//!
//! Reads requirements from `requirements/<vendor>.txt` and validates each
//! entry against the system.

use std::process::Command;
use tracing::{error, info, warn};

use crate::hardware::profile::Vendor;
use crate::utils::errors::{MiddlewareError, MiddlewareResult};

/// A single dependency check result.
#[derive(Debug, Clone)]
pub struct DependencyStatus {
    pub name: String,
    pub required: bool,
    pub found: bool,
    pub version: Option<String>,
    pub detail: String,
}

impl std::fmt::Display for DependencyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let icon = if self.found {
            "✓"
        } else if self.required {
            "✗"
        } else {
            "○"
        };
        let ver = self.version.as_deref().unwrap_or("—");
        write!(
            f,
            "  [{}] {} (version: {}) — {}",
            icon, self.name, ver, self.detail
        )
    }
}

/// Check all dependencies for the given vendor.
/// Returns a list of statuses and an error if any required dependency is missing.
pub fn check_dependencies(vendor: Vendor) -> MiddlewareResult<Vec<DependencyStatus>> {
    info!(vendor = %vendor, "Checking dependencies");

    let mut statuses = vec![];

    // Always check generic dependencies
    statuses.extend(check_generic_deps());

    // Platform-specific checks
    match vendor {
        Vendor::Nvidia => statuses.extend(check_nvidia_deps()),
        Vendor::Qualcomm => statuses.extend(check_qualcomm_deps()),
        Vendor::TexasInstruments => statuses.extend(check_ti_deps()),
        Vendor::Rockchip => { /* No additional deps */ }
        Vendor::Cpu => { /* No additional deps */ }
        Vendor::Unknown => { /* No additional deps */ }
    }

    // Report
    info!("Dependency check report");
    for s in &statuses {
        if s.found {
            info!(dep = %s.name, version = ?s.version, detail = %s.detail, "OK");
        } else if s.required {
            error!(dep = %s.name, detail = %s.detail, "Missing required dependency");
        } else {
            warn!(dep = %s.name, detail = %s.detail, "Optional dependency not found");
        }
    }

    // Fail if any required is missing
    let missing: Vec<_> = statuses.iter().filter(|s| s.required && !s.found).collect();

    if !missing.is_empty() {
        let names: Vec<_> = missing.iter().map(|s| s.name.as_str()).collect();
        return Err(MiddlewareError::DependencyMissing(format!(
            "Missing required dependencies: {}",
            names.join(", ")
        )));
    }

    Ok(statuses)
}

// ---------------------------------------------------------------------------
// Generic dependencies
// ---------------------------------------------------------------------------

fn check_generic_deps() -> Vec<DependencyStatus> {
    vec![
        // Rust toolchain
        check_command("rustc", &["--version"], "Rust compiler", true),
        // protoc
        check_command("protoc", &["--version"], "Protocol Buffers compiler", false),
    ]
}

// ---------------------------------------------------------------------------
// NVIDIA dependencies
// ---------------------------------------------------------------------------

fn check_nvidia_deps() -> Vec<DependencyStatus> {
    vec![
        check_command("nvidia-smi", &[], "NVIDIA GPU driver", true),
        check_command("nvcc", &["--version"], "CUDA Toolkit", true),
        check_command("trtexec", &["--help"], "TensorRT CLI", false),
        check_library("cudart", "CUDA Runtime Library", true),
        check_library("nvinfer", "TensorRT Inference Library", true),
        check_library("nvinfer_plugin", "TensorRT Plugin Library", false),
    ]
}

// ---------------------------------------------------------------------------
// Qualcomm dependencies
// ---------------------------------------------------------------------------

fn check_qualcomm_deps() -> Vec<DependencyStatus> {
    let checks = vec![
        check_env_var("SNPE_ROOT", "SNPE SDK root", false),
        check_env_var("QNN_SDK_ROOT", "QNN SDK root", false),
        check_library("QnnHtp", "QNN HTP Backend", false),
        check_library("QnnCpu", "QNN CPU Backend", false),
        check_library("SNPE", "SNPE Runtime", false),
        check_library("rpcmem", "RPCMem (Hexagon zero-copy)", false),
    ];

    let mut result = checks;
    // At least one of SNPE or QNN should be available
    result.push(DependencyStatus {
        name: "SNPE or QNN (at least one)".into(),
        required: true,
        found: result
            .iter()
            .any(|c| c.found && (c.name.contains("QNN") || c.name.contains("SNPE"))),
        version: None,
        detail: "Either SNPE or QNN SDK is needed for Qualcomm inference".into(),
    });

    result
}

// ---------------------------------------------------------------------------
// TI dependencies
// ---------------------------------------------------------------------------

fn check_ti_deps() -> Vec<DependencyStatus> {
    vec![
        check_env_var("TIDL_TOOLS_PATH", "TIDL Tools", false),
        check_env_var("EDGEAI_SDK_PATH", "TI Edge AI SDK", false),
        check_library("tidl_rt", "TIDL Runtime", true),
        check_library("cmem", "CMEM Allocator", false),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_command(cmd: &str, args: &[&str], desc: &str, required: bool) -> DependencyStatus {
    match Command::new(cmd).args(args).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()
                .map(|l| l.trim().to_string());
            DependencyStatus {
                name: desc.into(),
                required,
                found: true,
                version,
                detail: format!("`{}` found in PATH", cmd),
            }
        }
        Ok(_) => DependencyStatus {
            name: desc.into(),
            required,
            found: false,
            version: None,
            detail: format!("`{}` found but returned error", cmd),
        },
        Err(_) => DependencyStatus {
            name: desc.into(),
            required,
            found: false,
            version: None,
            detail: format!("`{}` not found in PATH", cmd),
        },
    }
}

fn check_library(name: &str, desc: &str, required: bool) -> DependencyStatus {
    let found = crate::hardware::detector::probe_library_public(name);
    DependencyStatus {
        name: desc.into(),
        required,
        found,
        version: None,
        detail: if found {
            format!("lib{} found on system", name)
        } else {
            format!("lib{} NOT found on system", name)
        },
    }
}

fn check_env_var(var: &str, desc: &str, required: bool) -> DependencyStatus {
    match std::env::var(var) {
        Ok(val) => DependencyStatus {
            name: desc.into(),
            required,
            found: true,
            version: None,
            detail: format!("${} = {}", var, val),
        },
        Err(_) => DependencyStatus {
            name: desc.into(),
            required,
            found: false,
            version: None,
            detail: format!("${} not set", var),
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_deps_always_work() {
        // CPU vendor should not fail (no required platform-specific deps)
        let result = check_dependencies(Vendor::Cpu);
        assert!(result.is_ok());
    }
}
