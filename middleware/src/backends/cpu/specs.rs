//! Pure data and constants describing CPU hardware capabilities.
//! This module is **always compiled** — no feature flags required.

/// CPU hardware fallback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuHardware {
    /// Generic CPU
    Generic,
}

impl CpuHardware {
    pub const fn compute_capability(self) -> &'static str {
        match self {
            Self::Generic => "x86_64 / aarch64",
        }
    }
}
