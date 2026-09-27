//! Pure data and constants describing Texas Instruments hardware capabilities.
//! This module is **always compiled** — no feature flags required.

/// TI hardware generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiHardware {
    /// TDA4VM / J721E
    Tda4,
}

impl TiHardware {
    pub const fn compute_capability(self) -> &'static str {
        match self {
            Self::Tda4 => "C7x DSP / MMA",
        }
    }
}
