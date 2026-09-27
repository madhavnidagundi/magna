//! Pure data and constants describing Qualcomm hardware capabilities.
//! This module is **always compiled** — no feature flags required.

use crate::utils::errors::Precision;

/// Qualcomm hardware generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualcommHardware {
    /// Snapdragon (SM8650, SM8550)
    Snapdragon,
    /// Automotive (SA8295 / Radxa Q6A)
    Sa8295,
}

impl QualcommHardware {
    pub const fn compute_capability(self) -> &'static str {
        match self {
            Self::Snapdragon => "Hexagon",
            Self::Sa8295 => "Hexagon Automotive",
        }
    }

    pub const fn default_perf_profile(self) -> &'static str {
        "burst"
    }

    pub const fn htp_precision(self) -> &'static str {
        "int8"
    }

    pub const fn max_precision(self) -> Precision {
        Precision::INT8
    }

    pub const fn recommended_batch_size(self) -> u32 {
        1
    }
}
