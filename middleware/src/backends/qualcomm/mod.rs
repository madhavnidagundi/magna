#[cfg(feature = "qualcomm")]
pub mod adapter;
pub mod specs;
#[cfg(feature = "qualcomm")]
pub use adapter::QualcommAdapter;
