#!/usr/bin/env bash

# Directory to store logs
LOG_DIR="/home/nvidia/magna/middleware/logs"
mkdir -p "$LOG_DIR"

# Timestamp for unique log file name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/magna_${TIMESTAMP}.log"

# Run Magna with camera input (INT8 quantized) and capture both stdout and stderr to the log file
cargo run --release --bin magna -- --model mobilenet --precision int8 --camera /dev/video0 2>&1 | tee "$LOG_FILE"
