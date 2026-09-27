pub mod specs;

#[cfg(feature = "ti")]
pub mod adapter;
#[cfg(feature = "ti")]
pub use adapter::TiAdapter;
