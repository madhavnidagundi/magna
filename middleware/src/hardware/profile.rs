// =============================================================================
// Magna Middleware — Hardware Profile
// =============================================================================
//! Data types describing the capabilities of the detected hardware.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Vendor enum
// ---------------------------------------------------------------------------

/// Known hardware vendors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Vendor {
    Nvidia,
    Qualcomm,
    TexasInstruments,
    Rockchip,
    Cpu,
    Unknown,
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Vendor::Nvidia => write!(f, "NVIDIA"),
            Vendor::Qualcomm => write!(f, "Qualcomm"),
            Vendor::TexasInstruments => write!(f, "Texas Instruments"),
            Vendor::Rockchip => write!(f, "Rockchip"),
            Vendor::Cpu => write!(f, "CPU"),
            Vendor::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// Accelerator info
// ---------------------------------------------------------------------------

/// Describes a single compute accelerator (GPU, DSP, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Accelerator {
    /// Human-readable name, e.g. "Orin GPU", "Hexagon DSP".
    pub name: String,
    /// Device index (useful when multiple devices present).
    pub device_index: u32,
    /// Total device memory in bytes (0 if unknown).
    pub memory_bytes: u64,
}

// ---------------------------------------------------------------------------
// Hardware profile
// ---------------------------------------------------------------------------

/// Aggregated hardware profile produced by the detection module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub vendor: Vendor,
    pub device_name: String,
    pub supports_fp32: bool,
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub supports_fp8: bool,
    pub accelerators: Vec<Accelerator>,
}

impl HardwareProfile {
    /// Create a profile that indicates *no* usable hardware was found.
    pub fn none() -> Self {
        Self {
            vendor: Vendor::Unknown,
            device_name: "None".into(),
            supports_fp32: false,
            supports_fp16: false,
            supports_int8: false,
            supports_fp8: false,
            accelerators: Vec::new(),
        }
    }

    /// Whether *any* accelerator was detected.
    pub fn has_accelerator(&self) -> bool {
        !self.accelerators.is_empty()
    }

    /// Quick helper: does this profile support the given Precision?
    pub fn supports_precision(&self, p: crate::utils::errors::Precision) -> bool {
        match p {
            crate::utils::errors::Precision::FP32 => self.supports_fp32,
            crate::utils::errors::Precision::FP16 => self.supports_fp16,
            crate::utils::errors::Precision::INT8 => self.supports_int8,
            crate::utils::errors::Precision::FP8 => self.supports_fp8,
        }
    }

    /// Returns the best precision mode supported by this hardware.
    pub fn best_precision(&self) -> crate::utils::errors::Precision {
        if self.supports_fp16 {
            crate::utils::errors::Precision::FP16
        } else {
            crate::utils::errors::Precision::FP32
        }
    }
}

impl fmt::Display for HardwareProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HardwareProfile {{ vendor: {}, device: {}, accelerators: {}, fp32: {}, fp16: {}, int8: {}, fp8: {} }}",
            self.vendor,
            self.device_name,
            self.accelerators.len(),
            self.supports_fp32,
            self.supports_fp16,
            self.supports_int8,
            self.supports_fp8,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::errors::Precision;

    #[test]
    fn none_profile_has_no_accelerators() {
        let p = HardwareProfile::none();
        assert!(!p.has_accelerator());
        assert_eq!(p.vendor, Vendor::Unknown);
    }

    #[test]
    fn precision_support_check() {
        let p = HardwareProfile {
            vendor: Vendor::Nvidia,
            device_name: "Orin GPU".into(),
            supports_fp32: true,
            supports_fp16: true,
            supports_int8: true,
            supports_fp8: false,
            accelerators: vec![Accelerator {
                name: "Orin GPU".into(),
                device_index: 0,
                memory_bytes: 8 * 1024 * 1024 * 1024,
            }],
        };
        assert!(p.supports_precision(Precision::FP32));
        assert!(p.supports_precision(Precision::FP16));
        assert!(p.supports_precision(Precision::INT8));
        assert!(!p.supports_precision(Precision::FP8));
        assert!(p.has_accelerator());
    }

    #[test]
    fn display_format() {
        let p = HardwareProfile::none();
        let s = p.to_string();
        assert!(s.contains("Unknown"));
    }

    #[test]
    fn cpu_vendor() {
        let p = HardwareProfile {
            vendor: Vendor::Cpu,
            device_name: "x86_64".into(),
            supports_fp32: true,
            supports_fp16: false,
            supports_int8: false,
            supports_fp8: false,
            accelerators: vec![Accelerator {
                name: "x86_64".into(),
                device_index: 0,
                memory_bytes: 0,
            }],
        };
        assert_eq!(p.vendor, Vendor::Cpu);
        assert_eq!(p.best_precision(), Precision::FP32);
    }
}
