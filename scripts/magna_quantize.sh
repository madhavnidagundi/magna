#!/bin/bash
# scripts/magna_quantize.sh
# End-to-end execution of the Qualcomm QNN Quantization Pipeline

set -e

# Default values
SDK_ROOT="/opt/qairt"
ONNX_PATH=""
DATASET_PATH=""
OUTPUT_DLC=""
INPUT_DIM=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --qairt_sdk) SDK_ROOT="$2"; shift ;;
        --onnx) ONNX_PATH="$2"; shift ;;
        --dataset) DATASET_PATH="$2"; shift ;;
        --output_dlc) OUTPUT_DLC="$2"; shift ;;
        --input_dim) INPUT_DIM="$2"; shift ;;
        --num_images) NUM_IMAGES="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$ONNX_PATH" ] || [ -z "$DATASET_PATH" ] || [ -z "$OUTPUT_DLC" ]; then
    echo "Usage: ./scripts/magna_quantize.sh --qairt_sdk <path_to_sdk> --onnx <path> --dataset <path> --output_dlc <path> [--input_dim 'input 1,3,224,224']"
    echo ""
    echo "Example:"
    echo "./scripts/magna_quantize.sh \\"
    echo "    --qairt_sdk /home/ml6/Documents/magna_edgeai/qairt/2.47.0.260601 \\"
    echo "    --onnx models/efficientnet_v2_s.onnx \\"
    echo "    --dataset models/calib \\"
    echo "    --output_dlc models/efficientnet_v2_s_int8.dlc \\"
    echo "    --input_dim 'input 1,3,224,224'"
    exit 1
fi

if [ -f "$OUTPUT_DLC" ]; then
    echo "======================================================="
    echo " Quantized DLC already exists: $OUTPUT_DLC"
    echo " Skipping quantization step to save time!"
    echo "======================================================="
    exit 0
fi

echo "======================================================="
echo " Step 1: Building Docker Image (qualcomm-optimizer)"
echo "======================================================="
docker build -f docker/Dockerfile.qualcomm-optimizer -t qualcomm-optimizer .

echo ""
echo "======================================================="
echo " Step 2: Running End-to-End QAIRT Pipeline in Docker"
echo "======================================================="

# Build the docker run command dynamically based on optional input_dim
DOCKER_CMD=(docker run --rm
    -v "$SDK_ROOT:/opt/qairt"
    -v "$(pwd):/workspace"
    qualcomm-optimizer
    --onnx "/workspace/$ONNX_PATH"
    --dataset "/workspace/$DATASET_PATH"
    --output_dlc "/workspace/$OUTPUT_DLC"
)

if [ -n "$INPUT_DIM" ]; then
    # Add input_dim flag by splitting the string (e.g. 'input 1,3,224,224' -> 'input' '1,3,224,224')
    DOCKER_CMD+=(--input_dim $INPUT_DIM)
fi

if [ -n "$NUM_IMAGES" ]; then
    DOCKER_CMD+=(--num_images $NUM_IMAGES)
fi

echo "Executing: ${DOCKER_CMD[@]}"
"${DOCKER_CMD[@]}"

echo ""
echo "======================================================="
echo " Pipeline Complete!"
echo " Generated DLC: $OUTPUT_DLC"
echo "======================================================="
