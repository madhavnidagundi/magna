use clap::ValueEnum;

/// All hardware targets supported by `misal optimize`.
/// Parsed directly from CLI: `--hw ORIN`, `--hw RADXA-Q6A`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum HardwareTarget {
    // NVIDIA family
    Orin, // AGX Orin / Orin NX / Orin Nano (sm_87)
    Thor, // NVIDIA Thor (sm_9x)

    // Qualcomm family
    Snapdragon, // Covers SM8650, SM8550, SA8295 (automotive)
    Sa8295,     // Qualcomm automotive variant (explicit)
    RadxaQ6a,   // Radxa Dragon Q6A (Qualcomm QCS8550 — HTP)

    // Rockchip family
    Rk3588, // Generic RK3588

    // Texas Instruments
    Tda4, // TDA4VM / J721E

    // Fallback
    Cpu,
}

impl HardwareTarget {
    /// Whether engine files built on an x86 machine are portable to this target.
    pub fn is_cross_portable(self) -> bool {
        match self {
            // SM architecture locked at trtexec time — must build on device
            Self::Orin | Self::Thor => false,
            // Software compiler — portable
            Self::Snapdragon
            | Self::Sa8295
            | Self::RadxaQ6a
            | Self::Rk3588
            | Self::Tda4
            | Self::Cpu => true,
        }
    }

    /// The tool used to build the model artifact.
    pub fn build_tool(self) -> &'static str {
        match self {
            Self::Orin | Self::Thor => "trtexec",
            Self::Snapdragon | Self::Sa8295 | Self::RadxaQ6a => "qairt-converter / qairt-quantizer",
            Self::Rk3588 => "rknn_convert",
            Self::Tda4 => "tidl_model_import",
            Self::Cpu => "none (copy ONNX)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_portability() {
        assert!(!HardwareTarget::Orin.is_cross_portable());
        assert!(!HardwareTarget::Thor.is_cross_portable());
        assert!(HardwareTarget::Cpu.is_cross_portable());
        assert!(HardwareTarget::RadxaQ6a.is_cross_portable());
    }
}
