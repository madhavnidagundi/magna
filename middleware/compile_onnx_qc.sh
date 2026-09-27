#!/bin/bash
# =============================================================================
# Magna Middleware — ONNX to QNN Compiler (Snapdragon/Radxa Q6A)
# =============================================================================
# Run this script on your host (x86) to compile your ONNX model into an optimized
# QNN .dlc or native .bin artifact.
#
# Requirements: QAIRT SDK installed and QNN_SDK_ROOT set.

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <path_to_onnx_model> [options]"
    echo ""
    echo "Options:"
    echo "  --precision <fp32|fp16|int8>  Precision mode (default: fp32)"
    echo "  --output <path>              Output file path (.dlc or .bin)"
    echo "  --calib-list <path>          Path to calibration list file (required for INT8)"
    echo "  --native-ctx                 Generate native context binary (.bin) on device"
    echo "  --board <user@host>          SCP results to board (reads RADXA_USER@RADXA_HOST if not set)"
    echo "  --board-dir <path>           Remote path on board (default: \$RADXA_RUNTIME_ROOT/models)"
    echo ""
    echo "Environment variables:"
    echo "  RADXA_HOST          - Hostname/IP of the Radxa board"
    echo "  RADXA_USER          - SSH username (default: radxa)"
    echo "  RADXA_RUNTIME_ROOT  - Remote QNN runtime directory (default: ~/qnn_runtime)"
    echo ""
    echo "Example: $0 models/resnet50.onnx --precision int8 --calib-list calib_list.txt"
    exit 1
fi

ONNX_PATH=$1
shift

# Defaults — all configurable via environment variables, no hardcoded paths
PRECISION="fp32"
OUTPUT_PATH="${ONNX_PATH%.*}.dlc"
CALIB_LIST=""
NATIVE_CTX=""

# Board connection — resolved from env vars, overridable via CLI flags
RADXA_USER="${RADXA_USER:-radxa}"
RADXA_HOST="${RADXA_HOST:-}"
RADXA_RUNTIME_ROOT="${RADXA_RUNTIME_ROOT:-~/qnn_runtime}"
BOARD=""
BOARD_DIR="${RADXA_RUNTIME_ROOT}/models"

# Parse options
while [ "$#" -gt 0 ]; do
    case "$1" in
        --precision) PRECISION="$2"; shift 2 ;;
        --output)    OUTPUT_PATH="$2"; shift 2 ;;
        --calib-list) CALIB_LIST="$2"; shift 2 ;;
        --native-ctx) NATIVE_CTX="yes"; shift ;;
        --board)     BOARD="$2"; shift 2 ;;
        --board-dir) BOARD_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# If --board not explicitly set, try to construct from env vars
if [ -z "$BOARD" ] && [ -n "$RADXA_HOST" ]; then
    BOARD="${RADXA_USER}@${RADXA_HOST}"
fi

echo "============================================================"
echo "  [Magna AOT] Compiling ONNX to QNN (QAIRT)                 "
echo "============================================================"
echo "  Input ONNX : $ONNX_PATH"
echo "  Precision  : $PRECISION"
echo "  Output     : $OUTPUT_PATH"
echo "============================================================"

# Find tools
QAIRT_CONVERTER=$(which qairt-converter || true)
if [ -z "$QAIRT_CONVERTER" ]; then
    if [ -n "$QNN_SDK_ROOT" ]; then
        ARCH_DIR="x86_64-linux-clang"
        QAIRT_CONVERTER="$QNN_SDK_ROOT/bin/$ARCH_DIR/qairt-converter"
        QAIRT_QUANTIZER="$QNN_SDK_ROOT/bin/$ARCH_DIR/qairt-quantizer"
    fi
else
    QAIRT_QUANTIZER=$(which qairt-quantizer || true)
fi

if [ ! -x "$QAIRT_CONVERTER" ]; then
    echo "Error: 'qairt-converter' not found. Please activate QAIRT environment."
    exit 1
fi

FP_DLC="${ONNX_PATH%.*}_fp.dlc"

# Step 1: ONNX -> FP DLC
echo "-> Converting ONNX to FP DLC..."
$QAIRT_CONVERTER --input_network "$ONNX_PATH" --output_path "$FP_DLC"

FINAL_DLC="$FP_DLC"

# Step 2: Quantization (if INT8)
if [ "$PRECISION" = "int8" ]; then
    if [ -z "$CALIB_LIST" ]; then
        echo "Error: --calib-list is required for INT8 precision."
        exit 1
    fi
    if [ ! -x "$QAIRT_QUANTIZER" ]; then
        echo "Error: 'qairt-quantizer' not found."
        exit 1
    fi
    
    INT8_DLC="${ONNX_PATH%.*}_int8.dlc"
    echo "-> Quantizing to INT8 DLC..."
    $QAIRT_QUANTIZER --input_dlc "$FP_DLC" --input_list "$CALIB_LIST" --output_dlc "$INT8_DLC"
    FINAL_DLC="$INT8_DLC"
fi

cp "$FINAL_DLC" "$OUTPUT_PATH"

echo "-> Created $OUTPUT_PATH"

# Step 3: Native context generation
if [ -n "$NATIVE_CTX" ]; then
    if [ -z "$BOARD" ]; then
        echo "Error: --native-ctx requires --board or RADXA_HOST to be set."
        exit 1
    fi
    
    BIN_NAME="$(basename ${OUTPUT_PATH%.*}_native)"
    
    echo "-> SCPing DLC to board for native context generation..."
    ssh "$BOARD" "mkdir -p $BOARD_DIR ${RADXA_RUNTIME_ROOT}/contexts"
    scp "$OUTPUT_PATH" "${BOARD}:${BOARD_DIR}/"
    
    echo "-> Running qnn-context-binary-generator on board..."
    ssh "$BOARD" "export LD_LIBRARY_PATH=${RADXA_RUNTIME_ROOT}/lib:\$LD_LIBRARY_PATH && \
                  ${RADXA_RUNTIME_ROOT}/bin/qnn-context-binary-generator \
                  --model ${RADXA_RUNTIME_ROOT}/lib/libQnnModelDlc.so \
                  --backend ${RADXA_RUNTIME_ROOT}/lib/libQnnHtp.so \
                  --dlc_path ${BOARD_DIR}/$(basename $OUTPUT_PATH) \
                  --binary_file $BIN_NAME \
                  --output_dir ${RADXA_RUNTIME_ROOT}/contexts"
    
    echo "-> Context binary created on board: ${RADXA_RUNTIME_ROOT}/contexts/${BIN_NAME}.bin"
elif [ -n "$BOARD" ]; then
    echo "-> SCPing DLC to board..."
    ssh "$BOARD" "mkdir -p $BOARD_DIR"
    scp "$OUTPUT_PATH" "${BOARD}:${BOARD_DIR}/"
fi

echo "============================================================"
echo "  Compilation Successful!"
echo "============================================================"
