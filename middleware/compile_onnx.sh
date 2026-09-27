#!/bin/bash
# =============================================================================
# Magna Middleware — ONNX to TensorRT AOT Compiler (Orin-Optimized)
# =============================================================================
# Run this script on your target NVIDIA hardware via SSH to compile your ONNX
# model into an optimized TensorRT .engine artifact.
#
# Orin-specific optimizations:
#   - 4GB workspace (optimal for Orin GPU memory)
#   - Tactic sources: CUBLAS, CUBLAS_LT, CUDNN for best kernel selection
#   - INT8+FP16 mixed precision with calibration cache support
#   - Optional DLA offloading (Orin has 2 DLA cores)
#   - Sparsity support for pruned models (MobileNet-class)
#   - Multi-stream for better GPU utilization

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_onnx_model> [options]"
    echo ""
    echo "Options:"
    echo "  --precision <fp32|fp16|int8>  Precision mode (default: fp32)"
    echo "  --output <path>              Output engine path (default: <input>.engine)"
    echo "  --calib <cache_path>         Path to INT8 calibration cache"
    echo "  --calib-dir <dir>            Directory of calibration images"
    echo "  --dla <0|1>                  Enable DLA core (0 or 1)"
    echo "  --workspace <MB>            Workspace size in MB (default: 4096)"
    echo "  --sparsity                   Enable sparsity optimizations"
    echo "  --streams <N>               Number of streams (default: 2)"
    echo "  --profile                    Export layer profiling info"
    echo ""
    echo "Example: $0 models/resnet50.onnx --precision int8 --calib models/resnet50_calibration.cache"
    exit 1
fi

ONNX_PATH=$1
shift

# Defaults (Orin-optimized)
PRECISION="fp32"
OUTPUT_PATH="${ONNX_PATH%.*}.engine"
WORKSPACE=4096
CALIB_CACHE=""
CALIB_DIR=""
DLA_CORE=""
SPARSITY=""
STREAMS=2
PROFILE=""

# Parse options
while [ "$#" -gt 0 ]; do
    case "$1" in
        --precision) PRECISION="$2"; shift 2 ;;
        --output)    OUTPUT_PATH="$2"; shift 2 ;;
        --calib)     CALIB_CACHE="$2"; shift 2 ;;
        --calib-dir) CALIB_DIR="$2"; shift 2 ;;
        --dla)       DLA_CORE="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --sparsity)  SPARSITY="yes"; shift ;;
        --streams)   STREAMS="$2"; shift 2 ;;
        --profile)   PROFILE="yes"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  [Magna AOT] Compiling ONNX to TensorRT Engine (Orin)      "
echo "============================================================"
echo "  Input ONNX : $ONNX_PATH"
echo "  Precision  : $PRECISION"
echo "  Output     : $OUTPUT_PATH"
echo "  Workspace  : ${WORKSPACE}MB"
echo "============================================================"

# Ensure trtexec is available
TRTEXEC=""
for candidate in /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec /usr/bin/trtexec; do
    if [ -x "$candidate" ]; then
        TRTEXEC="$candidate"
        break
    fi
done

if [ -z "$TRTEXEC" ]; then
    TRTEXEC=$(which trtexec 2>/dev/null || true)
fi

if [ -z "$TRTEXEC" ]; then
    echo "Error: 'trtexec' not found."
    echo "Common location on Jetson/Orin: /usr/src/tensorrt/bin/trtexec"
    exit 1
fi

echo "Using trtexec: $TRTEXEC"

# Build compilation flags
FLAGS="--onnx=$ONNX_PATH --saveEngine=$OUTPUT_PATH --workspace=$WORKSPACE"

# Precision flags
case "$PRECISION" in
    fp16)
        FLAGS="$FLAGS --fp16"
        ;;
    int8)
        FLAGS="$FLAGS --int8 --fp16 --precisionConstraints=prefer"
        if [ -n "$CALIB_CACHE" ]; then
            FLAGS="$FLAGS --calib=$CALIB_CACHE"
            echo "  Calibration: $CALIB_CACHE"
        fi
        ;;
    fp8)
        FLAGS="$FLAGS --fp8 --fp16"
        ;;
    fp32)
        # No extra flags
        ;;
    *)
        echo "Error: Unknown precision '$PRECISION'. Use: fp32, fp16, int8, fp8"
        exit 1
        ;;
esac

# Orin-specific tactic sources for optimal kernel selection
FLAGS="$FLAGS --tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN"

# DLA offloading
if [ -n "$DLA_CORE" ]; then
    FLAGS="$FLAGS --useDLACore=$DLA_CORE --allowGPUFallback"
    echo "  DLA Core   : $DLA_CORE (with GPU fallback)"
fi

# Sparsity
if [ -n "$SPARSITY" ]; then
    FLAGS="$FLAGS --sparsity=enable"
    echo "  Sparsity   : enabled"
fi

# Multi-stream
if [ "$STREAMS" -gt 1 ]; then
    FLAGS="$FLAGS --streams=$STREAMS"
    echo "  Streams    : $STREAMS"
fi

# Profiling
if [ -n "$PROFILE" ]; then
    PROFILE_PATH="${OUTPUT_PATH%.*}_profile.json"
    FLAGS="$FLAGS --exportProfile=$PROFILE_PATH"
    echo "  Profile    : $PROFILE_PATH"
fi

# Benchmarking flags
FLAGS="$FLAGS --avgRuns=100 --duration=10 --verbose"

echo ""
echo "Running: $TRTEXEC $FLAGS"
echo ""

# Execute
$TRTEXEC $FLAGS 2>&1 | tee "${OUTPUT_PATH%.*}_build.log"

echo ""
echo "============================================================"
echo "  Compilation Successful!"
echo "  Engine: $OUTPUT_PATH"
echo ""
echo "  Run with middleware:"
echo "  ./target/release/magna --model \"$OUTPUT_PATH\" --image test_image.jpg"
echo "============================================================"
