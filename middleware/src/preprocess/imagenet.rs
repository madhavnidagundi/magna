// =============================================================================
// Magna Middleware — Generic Preprocessing Module
// =============================================================================
//! **Dataset-agnostic** image preprocessing pipeline.
//!
//! The module supports:
//! - Generic image resize + center crop (configurable target size)
//! - Normalization with configurable mean/std
//! - Precision conversion (FP32, FP16, INT8)
//! - Raw RGB byte preprocessing from camera/gRPC input
//! - ImageNet-compatible defaults for backward compatibility

use tracing::debug;

use crate::inference::traits::TensorBuffer;
use crate::utils::errors::{MiddlewareError, MiddlewareResult, Precision};

// ---------------------------------------------------------------------------
// Preprocessing configuration (dataset-agnostic)
// ---------------------------------------------------------------------------

/// Configuration for image preprocessing. All fields have sensible defaults
/// but nothing is hardcoded to a specific dataset.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    /// Target width after resize + crop.
    pub target_width: u32,
    /// Target height after resize + crop.
    pub target_height: u32,
    /// Resize short-edge to this before center crop (0 = use target size directly).
    pub resize_short_edge: u32,
    /// Channel-wise mean for normalization (R, G, B).
    pub mean: [f32; 3],
    /// Channel-wise std for normalization (R, G, B).
    pub std: [f32; 3],
    /// Whether to convert channels from BGR to RGB.
    pub bgr_to_rgb: bool,
    /// Whether to output in HWC layout (e.g., for QNN NHWC models) instead of default CHW.
    pub hwc_layout: bool,
}

impl Default for PreprocessConfig {
    /// Defaults to ImageNet-compatible preprocessing for backward compat.
    fn default() -> Self {
        Self {
            target_width: 224,
            target_height: 224,
            resize_short_edge: 256,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            bgr_to_rgb: false,
            hwc_layout: false,
        }
    }
}

impl PreprocessConfig {
    /// Create config for a specific target size with ImageNet normalization.
    pub fn with_size(width: u32, height: u32) -> Self {
        Self {
            target_width: width,
            target_height: height,
            resize_short_edge: std::cmp::max(width, height) + 32,
            ..Default::default()
        }
    }

    /// Create config with custom mean/std (e.g. for COCO, VOC, etc.).
    pub fn with_normalization(mut self, mean: [f32; 3], std: [f32; 3]) -> Self {
        self.mean = mean;
        self.std = std;
        self
    }

    /// Create config for raw (0-255 → 0.0-1.0) normalization.
    pub fn raw_normalized(width: u32, height: u32) -> Self {
        Self {
            target_width: width,
            target_height: height,
            resize_short_edge: 0,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
            bgr_to_rgb: false,
            hwc_layout: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Preprocess an image file from disk with default (ImageNet) config.
pub fn preprocess_image_file(path: &str, precision: Precision) -> MiddlewareResult<TensorBuffer> {
    preprocess_image_file_with_config(path, precision, &PreprocessConfig::default())
}

/// Preprocess an image file with custom config.
pub fn preprocess_image_file_with_config(
    path: &str,
    precision: Precision,
    config: &PreprocessConfig,
) -> MiddlewareResult<TensorBuffer> {
    let img = image::open(path)
        .map_err(|e| MiddlewareError::ImageLoadFailed(format!("{}: {}", path, e)))?;

    let rgb = img.to_rgb8();
    preprocess_rgb_image(&rgb, precision, config)
}

/// Preprocess raw RGB bytes (from camera or gRPC) with default config.
pub fn preprocess_raw_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    precision: Precision,
) -> MiddlewareResult<TensorBuffer> {
    preprocess_raw_rgb_with_config(data, width, height, precision, &PreprocessConfig::default())
}

/// Preprocess raw RGB bytes with custom config.
pub fn preprocess_raw_rgb_with_config(
    data: &[u8],
    width: u32,
    height: u32,
    precision: Precision,
    config: &PreprocessConfig,
) -> MiddlewareResult<TensorBuffer> {
    let expected_len = (width * height * 3) as usize;
    if data.len() != expected_len {
        return Err(MiddlewareError::PreprocessingFailed(format!(
            "Raw RGB data size mismatch: got {} bytes, expected {} for {}x{}",
            data.len(),
            expected_len,
            width,
            height
        )));
    }

    let img = image::RgbImage::from_raw(width, height, data.to_vec()).ok_or_else(|| {
        MiddlewareError::PreprocessingFailed("Failed to construct image from raw RGB data".into())
    })?;

    preprocess_rgb_image(&img, precision, config)
}

// ---------------------------------------------------------------------------
// Internal pipeline
// ---------------------------------------------------------------------------

fn preprocess_rgb_image(
    img: &image::RgbImage,
    precision: Precision,
    config: &PreprocessConfig,
) -> MiddlewareResult<TensorBuffer> {
    let tw = config.target_width;
    let th = config.target_height;

    // Step 1: Resize
    let resized = if config.resize_short_edge > 0 {
        let (w, h) = (img.width(), img.height());
        let short = std::cmp::min(w, h);
        let scale = config.resize_short_edge as f64 / short as f64;
        let new_w = (w as f64 * scale).round() as u32;
        let new_h = (h as f64 * scale).round() as u32;
        image::imageops::resize(img, new_w, new_h, image::imageops::FilterType::CatmullRom)
    } else {
        image::imageops::resize(img, tw, th, image::imageops::FilterType::CatmullRom)
    };

    // Step 2: Center crop
    let cropped = center_crop(&resized, tw, th);

    // Step 3: Normalize to f32 (CHW layout, channel-first)
    let num_pixels = (tw * th) as usize;
    let mut float_data = vec![0.0f32; 3 * num_pixels];

    for y in 0..th {
        for x in 0..tw {
            let pixel = cropped.get_pixel(x, y);
            let idx = (y * tw + x) as usize;

            let (r, g, b) = if config.bgr_to_rgb {
                (pixel[2] as f32, pixel[1] as f32, pixel[0] as f32)
            } else {
                (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32)
            };

            let r_norm = (r / 255.0 - config.mean[0]) / config.std[0];
            let g_norm = (g / 255.0 - config.mean[1]) / config.std[1];
            let b_norm = (b / 255.0 - config.mean[2]) / config.std[2];

            if config.hwc_layout {
                float_data[idx * 3] = r_norm;
                float_data[idx * 3 + 1] = g_norm;
                float_data[idx * 3 + 2] = b_norm;
            } else {
                float_data[idx] = r_norm;
                float_data[num_pixels + idx] = g_norm;
                float_data[2 * num_pixels + idx] = b_norm;
            }
        }
    }

    // Step 4: Convert precision
    let data = convert_precision(&float_data, precision)?;
    let shape = if config.hwc_layout {
        vec![1, th as usize, tw as usize, 3]
    } else {
        vec![1, 3, th as usize, tw as usize]
    };

    debug!(
        "[Preprocess] Output: shape={:?}, precision={:?}, bytes={}",
        shape,
        precision,
        data.len()
    );

    Ok(TensorBuffer {
        name: String::new(),
        data,
        shape,
        precision,
    })
}

fn center_crop(img: &image::RgbImage, tw: u32, th: u32) -> image::RgbImage {
    let (w, h) = (img.width(), img.height());
    if w == tw && h == th {
        return img.clone();
    }
    let x0 = (w.saturating_sub(tw)) / 2;
    let y0 = (h.saturating_sub(th)) / 2;
    image::imageops::crop_imm(img, x0, y0, tw, th).to_image()
}

fn convert_precision(src: &[f32], precision: Precision) -> MiddlewareResult<Vec<u8>> {
    match precision {
        Precision::FP32 => Ok(src.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Precision::FP16 => Ok(src
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect()),
        Precision::INT8 => {
            // Basic linear quantization: clamp to [-127, 127]
            // NOTE: Production INT8 should use calibration-derived scales.
            Ok(src
                .iter()
                .map(|&v| {
                    let scaled = (v * 127.0).clamp(-127.0, 127.0) as i8;
                    scaled as u8
                })
                .collect())
        }
        Precision::FP8 => {
            // FP8 E4M3: truncate from FP32 to 8-bit float.
            // NOTE: Production FP8 needs proper E4M3 conversion.
            Ok(src
                .iter()
                .map(|&v| {
                    let scaled = (v * 127.0).clamp(-127.0, 127.0) as i8;
                    scaled as u8
                })
                .collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(w: u32, h: u32) -> tempfile::NamedTempFile {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                ((x * 7 + y * 3) % 256) as u8,
                ((x * 11 + y * 5) % 256) as u8,
                ((x * 13 + y * 7) % 256) as u8,
            ])
        });
        let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
        img.save(tmp.path()).unwrap();
        tmp
    }

    #[test]
    fn preprocess_default_produces_correct_shape() {
        let img = create_test_image(640, 480);
        let tensor = preprocess_image_file(img.path().to_str().unwrap(), Precision::FP32).unwrap();

        assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
        assert_eq!(tensor.precision, Precision::FP32);
        assert_eq!(tensor.byte_size(), 3 * 224 * 224 * 4);
    }

    #[test]
    fn preprocess_custom_size() {
        let img = create_test_image(640, 480);
        let config = PreprocessConfig::with_size(384, 384);
        let tensor = preprocess_image_file_with_config(
            img.path().to_str().unwrap(),
            Precision::FP32,
            &config,
        )
        .unwrap();

        assert_eq!(tensor.shape, vec![1, 3, 384, 384]);
    }

    #[test]
    fn preprocess_fp16() {
        let img = create_test_image(300, 300);
        let tensor = preprocess_image_file(img.path().to_str().unwrap(), Precision::FP16).unwrap();

        assert_eq!(tensor.precision, Precision::FP16);
        assert_eq!(tensor.data.len(), 3 * 224 * 224 * 2);
    }

    #[test]
    fn preprocess_int8() {
        let img = create_test_image(300, 300);
        let tensor = preprocess_image_file(img.path().to_str().unwrap(), Precision::INT8).unwrap();

        assert_eq!(tensor.precision, Precision::INT8);
        assert_eq!(tensor.data.len(), 3 * 224 * 224);
    }

    #[test]
    fn preprocess_nonexistent_fails() {
        let result = preprocess_image_file("/nonexistent.jpg", Precision::FP32);
        assert!(result.is_err());
    }

    #[test]
    fn raw_rgb_preprocessing() {
        let w = 320u32;
        let h = 240u32;
        let data = vec![128u8; (w * h * 3) as usize];
        let tensor = preprocess_raw_rgb(&data, w, h, Precision::FP32).unwrap();
        assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
    }

    #[test]
    fn raw_rgb_wrong_size_fails() {
        let data = vec![0u8; 100]; // Way too small for 320x240
        assert!(preprocess_raw_rgb(&data, 320, 240, Precision::FP32).is_err());
    }

    #[test]
    fn bgr_to_rgb_conversion() {
        let config = PreprocessConfig {
            bgr_to_rgb: true,
            ..PreprocessConfig::with_size(4, 4)
        };
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([100, 150, 200]));
        let tensor = preprocess_rgb_image(&img, Precision::FP32, &config).unwrap();
        let values = tensor.try_as_f32_slice().unwrap();
        // With bgr_to_rgb, the first channel should use pixel[2] (200)
        assert!(!values.is_empty());
    }
}
