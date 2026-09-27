pub mod specs;

#[cfg(all(feature = "cpu", not(feature = "cpu-system-ort")))]
pub mod adapter;
#[cfg(all(feature = "cpu", feature = "cpu-system-ort"))]
#[path = "c_api_adapter.rs"]
pub mod adapter;
#[cfg(feature = "cpu")]
pub use adapter::CpuAdapter;
