# Magna Edge AI Middleware

[![Middleware Quality](https://github.com/MAGNA-Global/mi-isal/actions/workflows/middleware-quality.yml/badge.svg)](https://github.com/MAGNA-Global/mi-isal/actions/workflows/middleware-quality.yml)
[![Security Audit](https://github.com/MAGNA-Global/mi-isal/actions/workflows/security-audit.yml/badge.svg)](https://github.com/MAGNA-Global/mi-isal/actions/workflows/security-audit.yml)
[![Orin](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-orin.yml/badge.svg)](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-orin.yml)
[![Thor](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-thor.yml/badge.svg)](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-thor.yml)
[![TDA4](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-tda4.yml/badge.svg)](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-tda4.yml)
[![Radxa](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-radxa.yml/badge.svg)](https://github.com/MAGNA-Global/mi-isal/actions/workflows/hardware-validation-radxa.yml)

Magna is a modular, hardware-agnostic Edge AI middleware designed to execute AOT (Ahead-of-Time) compiled inference engines on embedded hardware. It now fully supports automated hardware-aware model optimizations natively during runtime ingestion and generic multi-output tensor pipelines (useful for complex tasks like Object Detection).

## Project Structure

- `middleware/`: Core Rust inference middleware server.
- `middleware/proto/`: gRPC protobuf definitions for the client-server API.
- `models/`: Subdirectory for storing compiled models (`.engine`, `.plan`, `.onnx`, etc.).
- `python_client/`: Python gRPC client scripts and utilities.
- `scripts/`: Utility scripts organized into `benchmark/`, `eval/`, and `quantization/`.

## Quality Gate

The middleware follows a zero-warning Rust policy and strict dependency security scanning. Every contribution should pass:

```bash
cd middleware
cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo audit
cargo deny check
```

Backend-specific linting is also required for `cpu`, `nvidia`, `qualcomm`, and `ti`. See `CONTRIBUTING.md` for the full checklist.
For our comprehensive dependency security policies, see [SUPPLY_CHAIN_SECURITY.md](SUPPLY_CHAIN_SECURITY.md).

## Build Prerequisites

To build the Rust middleware, you need the following installed:

- **Rust Toolchain**: Stable via `rustup`
- **Protobuf Compiler**: `protoc` (e.g., `sudo apt-get install protobuf-compiler`)
- **System Dependencies**: `libssl-dev`, `pkg-config`, `cmake`

Alternatively, you can use the provided VS Code Dev Container (`.devcontainer/devcontainer.json`) which pre-installs all necessary dependencies.

### Build Configuration (NVIDIA)

When building with the `nvidia` feature, the build script automatically searches for CUDA and TensorRT installations in standard paths (like `/usr/local/cuda` and `/opt/cuda`), and detects the system architecture (e.g., `x86_64-linux-gnu` or `aarch64-linux-gnu`).

If you have a custom JetPack installation or non-standard paths, you can override these using the following environment variables:

- `CUDA_ROOT` (or `CUDA_PATH`): Path to your CUDA installation (e.g., `/usr/local/cuda-12.0`).
- `TENSORRT_INCLUDE`: Path to TensorRT headers (e.g., `/usr/include/x86_64-linux-gnu`).
- `TENSORRT_LIB`: Path to TensorRT libraries (e.g., `/usr/lib/x86_64-linux-gnu`).

**Troubleshooting Build Failures:**
If the build fails complaining about missing headers (`cuda_runtime_api.h` or `NvInfer.h`), ensure:
1. You have installed `tensorrt-dev` or equivalent packages.
2. If your paths are non-standard, export `CUDA_ROOT`, `TENSORRT_INCLUDE`, and `TENSORRT_LIB` before running `cargo build`.

## Backend Support Status

Magna is designed to run on multiple hardware accelerators. The current support status is:

- **NVIDIA / TensorRT**: **Fully Supported**. Optimized for NVIDIA Orin/Thor edge devices featuring automated generic AOT optimization (INT8 calibration, multi-stream execution, Tensor Core auto-tuning, workspace manipulation) directly to TensorRT.
- **CPU Fallback**: **Supported**. Works natively on non-accelerated CPU profiles.
- **Qualcomm (QAIRT/QNN)**: **Fully Supported**. Optimized for Radxa Dragon Q6A and Hexagon DSPs using automated ONNX-to-INT8 `.dlc` quantization pipelines and native C++ QNN FFI execution.
- **TI (TDA4)**: **Stub**. Implementation planned, currently placeholder.

### Qualcomm Model Optimization Pipeline
To prepare models for the Qualcomm backend, use the containerized optimization pipeline to convert any `.onnx` model into a quantized INT8 `.dlc` model:

1. Build the optimizer image:
   ```bash
   docker build -f docker/Dockerfile.qualcomm-optimizer -t qualcomm-optimizer .
   ```
2. Run the quantization pipeline (requires the QAIRT SDK mounted to `/opt/qairt`):
   ```bash
   docker run --rm \
       -v /path/to/qairt/sdk:/opt/qairt \
       -v $(pwd):/workspace \
       qualcomm-optimizer \
       --onnx /workspace/models/model.onnx \
       --dataset /workspace/data \
       --output_dlc /workspace/models/model_int8.dlc \
       --input_dim input 1,3,224,224
   ```

## Getting Started

1. Set up your environment (installing Rust and prerequisites).
2. Ensure you have your exported models in the `models/` folder.
3. Build the middleware for your target backend using Cargo features (e.g., `nvidia`, `cpu`, `qualcomm`, `ti`). Exactly one target must be specified:
   ```bash
   cd middleware
   # For NVIDIA Orin
   cargo build --release --features nvidia

   # For Qualcomm Radxa
   cargo build --release --features qualcomm
   ```
4. Run the middleware server by passing the path to the compiled engine (`.plan` for NVIDIA, `.dlc` for Qualcomm):
   ```bash
   # NVIDIA Example
   cargo run --release --features nvidia -- --model ../models/mobilenetv4_medium_fp32.plan --port 50051

   # Qualcomm Example
   cargo run --release --features qualcomm -- --model ../models/efficientnet_v2_s_int8.dlc --port 50051
   ```

### Command-Line Arguments

When starting the middleware, use the following arguments to configure the server:

- `--model`: **(Required)** The file path to the Ahead-of-Time (AOT) compiled inference engine (e.g., `.engine`, `.plan`, or `.onnx` file) to be loaded.
- `--port`: _(Optional)_ The network port that the gRPC server will listen on (default is usually 50051).
