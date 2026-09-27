#!/usr/bin/env bash

# =============================================================================
# Magna Middleware - MobileNetV2 Optimization and Inference Script
# =============================================================================
# This script optimizes a MobileNetV2 ONNX model using the Magna middleware
# and runs inference on the ImageNet validation dataset.
#
# Usage:
#   ./optimize_and_infer_mobilenetv2.sh
#
# Requirements:
#   - NVIDIA hardware with TensorRT installed
#   - Rust toolchain for building middleware
#   - CUDA and TensorRT libraries
# =============================================================================

set -e  # Exit on any error

echo "=============================================================="
echo "Magna Middleware - MobileNetV2 Optimization & Inference"
echo "=============================================================="

# Project paths
PROJECT_DIR="/home/thor/Documents/magna/mi-isal"
MODELS_DIR="$PROJECT_DIR/models"
MIDDLEWARE_DIR="$PROJECT_DIR/middleware"
IMAGENET_DIR="$PROJECT_DIR/imagenet_val/archive/imagenet-val"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

# Model paths
ONNX_MODEL="$MODELS_DIR/mobilenetv2.onnx"
ENGINE_MODEL="$MODELS_DIR/mobilenetv2_nvidia_optimized_int8.engine"
CALIB_CACHE="$MODELS_DIR/mobilenetv2_calibration.cache"

# Middleware binary
MW_BIN="$MIDDLEWARE_DIR/target/release/magna"

# Check if we have the ONNX model
if [ ! -f "$ONNX_MODEL" ]; then
    echo "Error: ONNX model not found at $ONNX_MODEL"
    exit 1
fi

echo "Found MobileNetV2 ONNX model: $ONNX_MODEL"

# Build middleware if needed
if [ ! -f "$MW_BIN" ]; then
    echo "Building middleware..."
    cd "$MIDDLEWARE_DIR"
    cargo build --release --features nvidia
fi

# Check if we have TensorRT tools
TRTEXEC=$(which trtexec 2>/dev/null || echo "")
if [ -z "$TRTEXEC" ]; then
    echo "Warning: trtexec not found. Trying common locations..."
    for candidate in /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec /usr/bin/trtexec; do
        if [ -x "$candidate" ]; then
            TRTEXEC="$candidate"
            break
        fi
    done
fi

if [ -z "$TRTEXEC" ]; then
    echo "Error: trtexec not found. Please install TensorRT."
    exit 1
fi

echo "Using trtexec: $TRTEXEC"

# Step 1: Compile ONNX to TensorRT engine
echo ""
echo "Step 1: Compiling ONNX to TensorRT engine..."
echo "============================================================"

# Create calibration cache if it doesn't exist
if [ ! -f "$CALIB_CACHE" ]; then
    echo "Creating calibration cache..."
    # We'll use a small subset of ImageNet for calibration
    python3 -c "
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Create a small calibration dataset
calib_dir = '/tmp/mobilenetv2_calib'
os.makedirs(calib_dir, exist_ok=True)

# Use a few images from ImageNet for calibration
imagenet_path = '$IMAGENET_DIR'
images = []
for root, _, files in os.walk(imagenet_path):
    for f in files:
        if f.endswith('.JPEG'):
            images.append(os.path.join(root, f))
            if len(images) >= 100:
                break
    if len(images) >= 100:
        break

# Save calibration cache as numpy array
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

calib_data = []
for img_path in images:
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        calib_data.append(img.numpy())
    except:
        continue

# Save as numpy array
np.save('$CALIB_CACHE', np.array(calib_data))
print(f'Saved calibration data for {len(calib_data)} images')
"
else
    echo "Using existing calibration cache: $CALIB_CACHE"
fi

# Compile the ONNX model to TensorRT engine with INT8 optimization
echo "Compiling ONNX to TensorRT engine with INT8 optimization..."
"$MIDDLEWARE_DIR/compile_onnx.sh" \
    "$ONNX_MODEL" \
    --precision int8 \
    --calib "$CALIB_CACHE" \
    --output "$ENGINE_MODEL" \
    --workspace 4096 \
    --streams 2 \
    --sparsity

echo "Engine compilation complete!"
echo "============================================================"

# Step 2: Verify engine works with middleware
echo ""
echo "Step 2: Verifying engine with middleware..."
echo "============================================================"

# Test inference with a single image
echo "Testing inference with sample image..."
if [ -f "$MIDDLEWARE_DIR/assets/sample.jpg" ]; then
    echo "Using existing sample image for testing..."
    $MW_BIN \
        --model "$ENGINE_MODEL" \
        --image "$MIDDLEWARE_DIR/assets/sample.jpg" \
        --backend nvidia \
        --precision int8 \
        --labels "$MIDDLEWARE_DIR/tests/assets/imagenet_labels.txt" \
        --warmup 5 \
        --benchmark-iters 1
else
    echo "No sample image found, skipping test inference..."
fi

echo "============================================================"

# Step 3: Run inference on ImageNet dataset (limited to 100 images for demo)
echo ""
echo "Step 3: Running inference on ImageNet dataset (limited to 100 images)..."
echo "============================================================"

# Create a small test directory with a few images
TEST_DIR="/tmp/mobilenetv2_test_images"
mkdir -p "$TEST_DIR"

# Copy a few images from ImageNet for testing
echo "Copying sample images from ImageNet for testing..."
find "$IMAGENET_DIR" -name "*.JPEG" -type f | head -100 | while read -r img; do
    cp "$img" "$TEST_DIR/"
done

echo "Copied 100 sample images to test directory"

# Run middleware inference on test images
echo "Running middleware inference on test images..."
$MW_BIN \
    --model "$ENGINE_MODEL" \
    --image-dir "$TEST_DIR" \
    --backend nvidia \
    --precision int8 \
    --labels "$MIDDLEWARE_DIR/tests/assets/imagenet_labels.txt" \
    --warmup 10 \
    --benchmark-iters 100

echo "============================================================"

# Step 4: Summary
echo ""
echo "Step 4: Summary"
echo "============================================================"
echo "Optimization completed successfully!"
echo "Engine created: $ENGINE_MODEL"
echo "Calibration cache: $CALIB_CACHE"
echo "Test images processed: 100"
echo ""
echo "To run full ImageNet evaluation, use:"
echo "  python3 $SCRIPTS_DIR/eval/evaluate_trt_native.py"
echo ""
echo "To run inference on full ImageNet dataset:"
echo "  $MW_BIN --model \"$ENGINE_MODEL\" --image-dir \"$IMAGENET_DIR\" --backend nvidia --precision int8"
echo "============================================================"

echo ""
echo "Magna Middleware - MobileNetV2 Optimization & Inference Complete!"