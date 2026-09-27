#!/bin/bash
set -e

: "${QAIRT_SDK_ROOT:?Set QAIRT_SDK_ROOT to your QAIRT SDK installation path}"
: "${RADXA_HOST:?Set RADXA_HOST to the Radxa board hostname/IP, e.g. radxa-dragon-q6a.local}"

# Input quantization params for mobilenetv4_medium_int8.dlc, confirmed via
# qairt-dlc-info (see RADXA_NEXT_STEPS.md Phase 1). Re-derive if ever pointed
# at a different model.
QUANT_SCALE=0.018658448011
QUANT_OFFSET=-114.0

echo "=========================================================="
echo " 1. Quantizing ONNX to INT8 DLC on Host PC "
echo "=========================================================="
./scripts/magna_quantize.sh \
    --qairt_sdk "$QAIRT_SDK_ROOT" \
    --onnx models/mobilenetv4_medium_fp32.onnx \
    --dataset models/calib_data \
    --output_dlc models/mobilenetv4_test.dlc \
    --input_dim "input 1,3,224,224" \
    --num_images 100

echo ""
echo "=========================================================="
echo " 2. Uploading DLC to Board & NPU Compilation "
echo "=========================================================="
./target/release/misal deploy models/mobilenetv4_test.dlc \
    --server "http://${RADXA_HOST}:50051"
./target/release/misal prepare-context mobilenetv4_test.dlc \
    --server "http://${RADXA_HOST}:50051" --force

echo ""
echo "=========================================================="
echo " 3. End-to-End Inference over gRPC "
echo "=========================================================="
SAMPLE_RAW=$(ls models/eval_data/raw/eval_*.raw | head -1)
./target/release/misal infer "$SAMPLE_RAW" \
    --server "http://${RADXA_HOST}:50051" \
    --shape 1,3,224,224 \
    --quant-scale "$QUANT_SCALE" --quant-offset="$QUANT_OFFSET"

echo ""
echo "=========================================================="
echo " 4. OpenTelemetry Metrics Validation "
echo "=========================================================="
curl -s "http://${RADXA_HOST}:9090/metrics" | grep "inference"
echo ""
echo "=========================================================="
echo " PIPELINE TEST FULLY COMPLETE! "
echo "=========================================================="
