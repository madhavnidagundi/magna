// =============================================================================
// Integration Test — Preprocessing Pipeline
// =============================================================================
//! Tests the ImageNet preprocessing pipeline using programmatically generated
//! test images (no external files needed).

use magna_middleware::preprocess::imagenet;
use magna_middleware::utils::errors::Precision;

/// Generate a test image and save it to a tmp file.
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
fn preprocess_produces_correct_shape() {
    let img = create_test_image(640, 480);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::FP32).unwrap();

    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
    assert_eq!(tensor.precision, Precision::FP32);
    assert_eq!(tensor.byte_size(), 3 * 224 * 224 * 4);
}

#[test]
fn preprocess_square_image() {
    let img = create_test_image(224, 224);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::FP32).unwrap();

    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
}

#[test]
fn preprocess_portrait_image() {
    let img = create_test_image(480, 640);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::FP32).unwrap();

    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
}

#[test]
fn preprocess_large_image() {
    let img = create_test_image(1920, 1080);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::FP32).unwrap();

    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
}

#[test]
fn preprocess_small_image() {
    let img = create_test_image(100, 80);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::FP32).unwrap();

    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
}

#[test]
fn preprocess_int8_precision() {
    let img = create_test_image(300, 300);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::INT8).unwrap();

    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
    assert_eq!(tensor.precision, Precision::INT8);
    // INT8 → 1 byte per element.
    assert_eq!(tensor.data.len(), 3 * 224 * 224);
}

#[test]
fn preprocess_fp16_precision() {
    let img = create_test_image(300, 300);
    let tensor =
        imagenet::preprocess_image_file(img.path().to_str().unwrap(), Precision::FP16).unwrap();

    assert_eq!(tensor.precision, Precision::FP16);
    // FP16 → 2 bytes per element.
    assert_eq!(tensor.data.len(), 3 * 224 * 224 * 2);
}

#[test]
fn preprocess_nonexistent_file_fails() {
    let result = imagenet::preprocess_image_file("/nonexistent/image.jpg", Precision::FP32);
    assert!(result.is_err());
}

#[test]
fn normalized_values_are_reasonable() {
    // Create a solid grey (128, 128, 128) image.
    let img = image::RgbImage::from_pixel(224, 224, image::Rgb([128, 128, 128]));
    let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
    img.save(tmp.path()).unwrap();

    let tensor =
        imagenet::preprocess_image_file(tmp.path().to_str().unwrap(), Precision::FP32).unwrap();

    let values = tensor.try_as_f32_slice().unwrap();
    assert_eq!(values.len(), 3 * 224 * 224);

    // Normalized values for 128/255 ≈ 0.502
    // After ImageNet normalization: (0.502 - mean) / std
    // For R: (0.502 - 0.485) / 0.229 ≈ 0.074
    // All values should be finite and within a reasonable range.
    for &v in values {
        assert!(f32::is_finite(v), "Non-finite value found: {}", v);
        assert!(v.abs() < 10.0, "Value out of expected range: {}", v);
    }
}

#[test]
fn raw_rgb_preprocessing() {
    let w = 320u32;
    let h = 240u32;
    let data = vec![128u8; (w * h * 3) as usize];

    let tensor = imagenet::preprocess_raw_rgb(&data, w, h, Precision::FP32).unwrap();
    assert_eq!(tensor.shape, vec![1, 3, 224, 224]);
    assert_eq!(tensor.precision, Precision::FP32);
}
