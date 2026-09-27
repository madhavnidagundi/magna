#!/usr/bin/env bash

# Magna Middleware Benchmark Script
# Runs MobileNetV4 and EfficientNetV2 inference and compares with baseline results

set -e

MIDDLEWARE_DIR="/home/thor/Documents/magna/mi-isal/middleware"
MODELS_DIR="/home/thor/Documents/magna/mi-isal/models"
ASSETS_DIR="/home/thor/Documents/magna/mi-isal/assets"
BIN="/home/thor/Documents/magna/mi-isal/target/release/magna"

# Ensure Rust environment
. "$HOME/.cargo/env"
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "===================================================="
echo "Magna Middleware Benchmarking on NVIDIA Thor"
echo "===================================================="

run_bench() {
    local model_path=$1
    local model_name=$2
    
    if [ ! -f "$model_path" ]; then
        echo ""
        echo ">>> [SKIP] $model_name: Engine file not found at $model_path"
        return
    fi

    echo ""
    echo ">>> Running Benchmark for: $model_name"
    echo ">>> Model: $model_path"
    
    MAGNA_BENCH_MODEL="$model_path" \
    MAGNA_BENCH_IMAGE="$ASSETS_DIR/sample.jpg" \
    cargo bench --manifest-path "$MIDDLEWARE_DIR/Cargo.toml" --features nvidia
}

# MobileNetV4
run_bench "$MODELS_DIR/mobilenetv4_nvidia_optimized_int8.engine" "MobileNetV4 (INT8 Engine)"

# EfficientNetV2
run_bench "$MODELS_DIR/efficientnet_v2_s_nvidia_optimized_int8.engine" "EfficientNetV2 (INT8 Engine)"

echo ""
echo "===================================================="
echo "Benchmark Complete"
echo "===================================================="
