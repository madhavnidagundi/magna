// =============================================================================
// Magna Middleware — Input Provider Trait
// =============================================================================
//! Defines the [`InputProvider`] trait for pluggable input sources.

use magna_middleware::utils::errors::{MiddlewareError, MiddlewareResult};

// ---------------------------------------------------------------------------
// InputProvider trait
// ---------------------------------------------------------------------------

/// Abstraction over input sources (file, camera, gRPC stream, etc.).
pub trait InputProvider: Send {
    /// Human-readable name of the input source.
    fn source_name(&self) -> &str;

    /// Fetch the next frame. Returns `None` when exhausted.
    fn next_frame(&mut self) -> MiddlewareResult<Option<FrameData>>;

    /// Whether this provider streams indefinitely (e.g. live camera).
    fn is_streaming(&self) -> bool;
}

// ---------------------------------------------------------------------------
// FrameData
// ---------------------------------------------------------------------------

/// A single frame of input data.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Raw RGB bytes (3 bytes per pixel, row-major).
    pub rgb_data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

// ---------------------------------------------------------------------------
// FileInputProvider
// ---------------------------------------------------------------------------

/// Reads a single image file from disk.
/// After the first successful call to `next_frame`, returns `None`.
#[derive(Debug)]
pub struct FileInputProvider {
    path: String,
    consumed: bool,
}

impl FileInputProvider {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            consumed: false,
        }
    }
}

impl InputProvider for FileInputProvider {
    fn source_name(&self) -> &str {
        &self.path
    }

    fn next_frame(&mut self) -> MiddlewareResult<Option<FrameData>> {
        if self.consumed {
            return Ok(None);
        }
        // Do NOT set consumed before the load — if the load fails, we
        // allow retrying on the next call (H7 fix).
        let img = image::open(&self.path)
            .map_err(|e| MiddlewareError::ImageLoadFailed(format!("{}: {}", self.path, e)))?;

        // Only mark consumed AFTER successful load
        self.consumed = true;

        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());

        Ok(Some(FrameData {
            rgb_data: rgb.into_raw(),
            width: w,
            height: h,
        }))
    }

    fn is_streaming(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_provider_returns_error_then_allows_retry() {
        let mut provider = FileInputProvider::new("nonexistent.png");
        // Should fail because file does not exist.
        assert!(provider.next_frame().is_err());
        // Should still error (not consumed, allows retry).
        assert!(provider.next_frame().is_err());
        // consumed should still be false since it never succeeded
        assert!(!provider.consumed);
    }

    #[test]
    fn file_provider_is_not_streaming() {
        let provider = FileInputProvider::new("test.png");
        assert!(!provider.is_streaming());
    }

    #[test]
    fn frame_data_struct() {
        let frame = FrameData {
            rgb_data: vec![128; 3 * 10 * 10],
            width: 10,
            height: 10,
        };
        assert_eq!(frame.rgb_data.len(), 300);
    }
}
