//! Pure data and constants describing NVIDIA hardware capabilities.
//! This module is **always compiled** — no feature flags required.

use crate::utils::errors::Precision;

/// NVIDIA hardware generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvidiaHardware {
    /// Orin (Ampere) — supports up to FP16 precision
    Orin,
    /// Thor (next-gen) — supports up to FP8 precision
    Thor,
}

impl NvidiaHardware {
    pub const fn default_workspace_mb(self) -> u64 {
        match self {
            Self::Orin => 4096, // 4GB — optimal for Orin iGPU
            Self::Thor => 8192, // 8GB — Thor has more HBM
        }
    }

    pub const fn tactic_sources(self) -> &'static str {
        match self {
            Self::Orin => "+CUBLAS,+CUBLAS_LT,+CUDNN",
            // Thor adds edge mask convolutions for sparsity acceleration
            Self::Thor => "+CUBLAS,+CUBLAS_LT,+CUDNN,+EDGE_MASK_CONVOLUTIONS",
        }
    }

    pub const fn compute_capability(self) -> &'static str {
        match self {
            Self::Orin => "8.7",
            Self::Thor => "9.0",
        }
    }

    pub const fn supports_dla(self) -> bool {
        // Only Orin has DLA cores (2 × DLA v3.0)
        match self {
            Self::Orin => true,
            Self::Thor => false,
        }
    }

    pub const fn max_precision(self) -> Precision {
        match self {
            Self::Orin => Precision::INT8, // INT8 with calibration
            Self::Thor => Precision::FP8,  // Thor supports FP8 natively
        }
    }

    pub const fn recommended_batch_size(self) -> u32 {
        match self {
            Self::Orin => 8,
            Self::Thor => 16,
        }
    }
}
