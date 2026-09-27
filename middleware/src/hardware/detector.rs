// =============================================================================
// Magna Middleware — Hardware Profile Helpers
// =============================================================================
//! Compile-time hardware selection utilities.
//!
//! # Architecture
//!
//! This middleware uses **compile-time feature flags** for backend selection,
//! not runtime detection. The backend is chosen once at build time via Cargo
//! features:
//!
//! ```text
//! cargo build --features nvidia    # NVIDIA TensorRT backend
//! cargo build --features qualcomm  # Qualcomm QNN/SNPE backend
//! cargo build --features ti        # Texas Instruments TIDL backend
//! cargo build --features cpu       # CPU fallback (default)
//! ```
//!
//! This produces a lean, platform-specific binary with no runtime detection
//! overhead, consistent with the AOT (Ahead-of-Time) deployment model.
//!
//! # What This Module Provides
//!
//! - [`cpu_fallback_profile`] — a `HardwareProfile` matching the `cpu` feature
//! - [`simulated_profile`]   — a synthetic profile for test environments
//! - [`probe_library_public`] — shared library probe used by `dependency.rs`
//!
//! The old runtime `detect_hardware()` API has been removed. Backend selection
//! is performed entirely in `inference/engine_manager.rs` via `#[cfg(feature)]`
//! attributes.

use tracing::debug;

use crate::hardware::profile::{Accelerator, HardwareProfile, Vendor};

// ---------------------------------------------------------------------------
// Compile-time profiles
// ---------------------------------------------------------------------------

/// Returns a [`HardwareProfile`] representing the current CPU.
///
/// Used by the `cpu` backend and as a fallback in tests.
pub fn cpu_fallback_profile() -> HardwareProfile {
    let cpu_name = detect_cpu_name().unwrap_or_else(|| "Generic CPU".into());

    HardwareProfile {
        vendor: Vendor::Cpu,
        device_name: cpu_name.clone(),
        supports_fp32: true,
        supports_fp16: false,
        supports_int8: false,
        supports_fp8: false,
        accelerators: vec![Accelerator {
            name: cpu_name,
            device_index: 0,
            memory_bytes: 0,
        }],
    }
}

/// Returns a synthetic [`HardwareProfile`] for CI / dev-machine testing.
///
/// This profile simulates an NVIDIA GPU so that tests relying on
/// `has_accelerator()` pass on machines without real hardware.
pub fn simulated_profile() -> HardwareProfile {
    HardwareProfile {
        vendor: Vendor::Nvidia,
        device_name: "Simulated Accelerator".into(),
        supports_fp32: true,
        supports_fp16: true,
        supports_int8: true,
        supports_fp8: false,
        accelerators: vec![Accelerator {
            name: "Simulated Accelerator".into(),
            device_index: 0,
            memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB
        }],
    }
}

// ---------------------------------------------------------------------------
// Library probe (used by dependency.rs)
// ---------------------------------------------------------------------------

/// Check whether a shared library (`lib<name>.so` / `<name>.dll`) is
/// discoverable on the current system.
///
/// This is intentionally a **passive** probe — it never spawns processes and
/// only inspects the file system and `ldconfig` cache. It is used by
/// [`crate::hardware::dependency`] to verify that the SDK libraries required
/// by the *already-selected* compile-time backend are actually present at
/// runtime.
pub fn probe_library_public(name: &str) -> bool {
    probe_library(name)
}

fn probe_library(name: &str) -> bool {
    // Linux: scan common library directories
    #[cfg(target_os = "linux")]
    {
        let arch = if cfg!(target_arch = "aarch64") {
            "aarch64-linux-gnu"
        } else {
            "x86_64-linux-gnu"
        };

        let mut paths = vec![
            format!("/usr/lib/{}", arch),
            "/usr/lib".to_string(),
            "/usr/local/lib".to_string(),
            "/opt/qcom/lib".to_string(),
        ];

        if let Ok(cuda_root) = std::env::var("CUDA_ROOT").or_else(|_| std::env::var("CUDA_PATH")) {
            paths.push(format!("{}/lib64", cuda_root));
        } else {
            paths.push("/usr/local/cuda/lib64".to_string());
        }

        if let Ok(trt_lib) = std::env::var("TENSORRT_LIB") {
            paths.push(trt_lib);
        }

        let target = format!("lib{}", name);
        for path in &paths {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let fname = entry.file_name().to_string_lossy().to_string();
                    if fname.starts_with(&target)
                        && (fname.ends_with(".so") || fname.contains(".so."))
                    {
                        debug!(
                            module = "dep-check",
                            lib = %fname,
                            path = ?entry.path(),
                            "Found library"
                        );
                        return true;
                    }
                }
            }
        }

        // Also check ldconfig cache
        if let Ok(out) = std::process::Command::new("ldconfig").arg("-p").output() {
            if String::from_utf8_lossy(&out.stdout).contains(name) {
                debug!(lib = name, via = "ldconfig", "Found library");
                return true;
            }
        }
    }

    // Windows: scan PATH for DLLs
    #[cfg(target_os = "windows")]
    {
        let target = format!("{}.dll", name);
        if let Ok(path_var) = std::env::var("PATH") {
            for dir in path_var.split(';') {
                let dll = std::path::Path::new(dir).join(&target);
                if dll.exists() {
                    debug!(lib = %target, path = ?dll, "Found library");
                    return true;
                }
            }
        }
    }

    debug!(lib = name, "Library not found");
    false
}

// ---------------------------------------------------------------------------
// CPU name helper (private)
// ---------------------------------------------------------------------------

fn detect_cpu_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in info.lines() {
                if line.starts_with("model name") {
                    if let Some((_k, v)) = line.split_once(':') {
                        return Some(v.trim().to_string());
                    }
                }
            }
        }
        std::process::Command::new("uname")
            .arg("-m")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("wmic")
            .args(["cpu", "get", "name"])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .nth(1)
                    .map(|l| l.trim().to_string())
            })
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_profile_is_valid() {
        let profile = cpu_fallback_profile();
        assert_eq!(profile.vendor, Vendor::Cpu);
        assert!(!profile.device_name.is_empty());
        assert!(profile.has_accelerator());
        assert!(profile.supports_fp32);
    }

    #[test]
    fn simulated_profile_has_accelerator() {
        let profile = simulated_profile();
        assert!(profile.has_accelerator());
        assert_eq!(profile.accelerators[0].name, "Simulated Accelerator");
        assert!(profile.supports_fp16);
    }
}
