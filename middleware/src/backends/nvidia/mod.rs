pub mod specs;
pub use specs::NvidiaHardware;

#[cfg(feature = "nvidia")]
pub mod adapter;
#[cfg(feature = "nvidia")]
pub use adapter::NvidiaAdapter;
