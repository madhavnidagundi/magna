# Magna Edge AI Middleware: Hardware Setup & Execution Guide

Setting up and deploying the **Magna Edge AI Middleware** requires building the core Rust inference server with the appropriate backend features and preparing Ahead-of-Time (AOT) compiled TensorRT engines optimized for the target architecture.

Below is the complete end-to-end setup guide for both **NVIDIA Thor** and **NVIDIA Orin** devices from scratch.

---

## 1. Common System & Environment Prerequisites

Regardless of the target hardware, run the following on the host/edge device to prepare the core compilation toolchains:

```bash
# 1. Install system dependencies and Protobuf compiler (for gRPC transport)
sudo apt-get update
sudo apt-get install -y build-essential curl cmake libssl-dev pkg-config protobuf-compiler

# 2. Install stable Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 3. Clone the repository and prepare the Python client virtual environment
git clone https://github.com/Alex-MAGNA/mi-isal.git
cd mi-isal
python3 -m venv .venv
source .venv/bin/activate
```

---

## 2. Setup & Execution Steps for NVIDIA Thor (Blackwell)

**Thor** supports native **FP8/INT8 precision**, leverages **TensorRT 10.x+**, and utilizes **CUDA Graphs** inside the middleware for near-zero latency execution. All compilation, execution, and serving steps are handled natively by the high-performance Rust middleware.

### Step 2.1: Build the Optimized TensorRT Engine First
Use the core `magna` Rust CLI entrypoint to compile the baseline ONNX model into a highly optimized Ahead-of-Time (AOT) TensorRT engine:

```bash
cd middleware

# Build optimized FP8/INT8 TensorRT engine directly via the Rust middleware
cargo run --release --bin magna --features nvidia -- \
    --build ../models/mobilenetv4.onnx \
    --output ../models/mobilenetv4_thor.engine \
    --precision fp8 \
    --backend nvidia
```

### Step 2.2: Run the Model on a Dataset
Evaluate end-to-end Top-1 accuracy, latency, and throughput natively against the full validation dataset using the Rust middleware engine.

> **Dataset Installation:** Download and install the official ImageNet validation dataset from Kaggle:
> **(https://www.kaggle.com/datasets/titericz/imagenet1k-val)**
> 
> Ensure the archive is extracted such that validation images are located under `imagenet_val/archive/imagenet-val/`.

Run the dataset benchmark via the middleware CLI:
```bash
cargo run --release --bin magna --features nvidia -- \
    --model ../models/mobilenetv4_thor.engine \
    --image-dir ../imagenet_val/archive/imagenet-val \
    --synset-mapping ../assets/synset_to_id.json \
    --labels ../middleware/tests/assets/imagenet_labels.txt \
    --backend nvidia
```

### Step 2.3: Run the Model on a Dummy Image to Test Running
To quickly test engine loading, inference correctness, and obtain class output rankings without full dataset traversal, run on a sample test image:

```bash
cargo run --release --bin magna --features nvidia -- \
    --model ../models/mobilenetv4_thor.engine \
    --image ../assets/sample.jpg \
    --labels ../middleware/tests/assets/imagenet_labels.txt \
    --backend nvidia
```

### Step 2.4: Server Type of Running the Model
Launch the standalone production-grade gRPC inference server wrapper directly via the middleware to stream high-throughput real-time requests:

```bash
# Ensure CUDA and TRT libraries are in your library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Launch the gRPC inference server listening on port 50051
cargo run --release --bin magna_server --features nvidia -- \
    --model ../models/mobilenetv4_thor.engine \
    --backend nvidia \
    --precision fp8 \
    --address 127.0.0.1:50051
```

*(Alternatively, you can cross-compile a standalone release binary utilizing the automated compiler utility: `cargo run --release --bin create_middleware -- -m ../models/mobilenetv4.onnx --hw THOR -o magna_server_thor --om ../models/mobilenetv4_thor.engine`)*

> **Note:** During engine initialization on Thor, the middleware automatically executes `trt_capture_graph` to trace and capture the execution pipeline into a high-speed CUDA Graph.

---

## 3. Setup & Execution Steps for NVIDIA Orin (Ampere)

**Orin** devices run on standard JetPack environments targeting **TensorRT 8.6+**, optimizing for **INT8/FP16 mixed precision**, **4GB memory workspace** limits, and optional **DLA core offloading**.

### Step 3.1: Install Orin System & Middleware Requirements
Ensure the base runtime environments matching `middleware/requirements/nvidia.txt` are satisfied:
```bash
sudo apt-get install -y tensorrt-dev cuda-toolkit-11-8
```

### Step 3.2: Compile the ONNX Model to an Orin TensorRT Engine
Use the provided native optimization script designed to tune tactic sources (`CUBLAS`, `CUBLAS_LT`, `CUDNN`) and configure dual-stream execution:

```bash
cd middleware

# Compile ONNX to an INT8 engine utilizing a calibration cache and 4GB Workspace tuning
./compile_onnx.sh ../models/mobilenetv4.onnx \
    --precision int8 \
    --workspace 4096 \
    --streams 2 \
    --output ../models/mobilenetv4_orin.engine
```
*(Optional: Pass `--dla 0` or `--dla 1` to offload inference onto Orin's Deep Learning Accelerator cores).*

### Step 3.3: Build and Launch the Middleware Server for Orin
Compile the core server binary with the generic `nvidia` feature enabled:

```bash
cargo build --release --features nvidia

# Run the middleware server passing the Orin engine
./target/release/magna --model ../models/mobilenetv4_orin.engine --port 50051
```

---

## 4. Running Real-Time Client Inference (Both Platforms)

Once the middleware server is successfully listening on either device, start the semantic camera inference client to process live video frames over optimized gRPC streams:

```bash
# From the project root directory
cd python_client

# Ensure dependencies are installed in your active venv
pip install opencv-python grpcio

# Launch live streaming client (connects to localhost:50051 and reads /dev/video1)
python camera_inference.py
```

The client captures compressed JPEG frames, streams them to the middleware server, calculates end-to-end true FPS metrics, and saves the annotated top-1/top-5 prediction output overlay to `inference_output.mp4`.
