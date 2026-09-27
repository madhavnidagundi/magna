# Model Optimization Results: NVIDIA Jetson AGX Orin

This document outlines the performance improvements achieved by running hardware-aware optimizations on the base ONNX models using the enhanced compilation pipeline.

## Hardware Environment
* **Platform:** NVIDIA Jetson AGX Orin
* **TensorRT Version:** 8.5.0.2
* **JetPack Version:** R35.2.1

---

## 1. MobileNetV4 (Medium) INT8

| Metric | Original Engine (Basic trtexec) | Orin-Optimized Engine | Improvement / Change |
| :--- | :--- | :--- | :--- |
| **Throughput (FPS)** | 1,348.78 qps | **2,334.69 qps** | **1.73x Speedup** 🚀 |
| **GPU Latency (Mean)** | 0.740 ms | 0.854 ms | Slightly higher due to concurrent streams |
| **Host Latency (Mean)** | 0.768 ms | 0.891 ms | Slightly higher due to concurrent streams |
| **Engine file size** | ~11 MB | ~11 MB | Same size |

**Analysis for MobileNetV4:**
The massive throughput improvement (1.73x) is primarily driven by the introduction of `--streams=2`, allowing the GPU to process two inference streams concurrently. The minor increase in single-inference latency is highly acceptable given the near-doubling of system bandwidth.

---

## 2. EfficientNetV2 (Small) INT8

| Metric | Original Engine (Basic trtexec) | Orin-Optimized Engine | Improvement / Change |
| :--- | :--- | :--- | :--- |
| **Throughput (FPS)** | 339.18 qps | **741.24 qps** | **2.19x Speedup** 🚀 |
| **GPU Latency (Mean)** | 2.946 ms | **2.695 ms** | **8.5% Faster** latency |
| **Host Latency (Mean)** | 2.976 ms | **2.734 ms** | **8.1% Faster** latency |
| **Engine file size** | ~25 MB | ~25 MB | Same size |

**Analysis for EfficientNetV2:**
EfficientNetV2 demonstrates compounding benefits from the optimization pipeline. Not only does the multi-stream execution double the throughput (up to 2.19x), but the kernel auto-tuning (`+CUBLAS,+CUBLAS_LT,+CUDNN`) successfully discovers faster execution paths for the complex EfficientNet blocks, resulting in an 8.5% raw latency reduction.

---

## Key Hardware-Specific Optimizations Applied

The following optimizations were applied during compilation by the new middleware `optimizer.rs` pipeline (and `compile_onnx.sh`):

1. **Multi-stream Execution (`--streams=2`)**: Overlaps data transfer and compute, dramatically improving overall batch throughput.
2. **Explicit Tactic Sources (`--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN`)**: Forces TensorRT to evaluate an expanded search space of matrix-multiplication kernels optimized for the Orin architecture.
3. **Optimized Workspace Size (`--workspace=4096`)**: Allocates 4GB of workspace to TensorRT during compilation, ensuring it has enough memory to select high-performance kernels that require large scratch buffers.
4. **Strict Precision Constraints (`--precisionConstraints=prefer`)**: Prevents TensorRT from silently falling back to unoptimized higher-precision implementations unless mathematically necessary.
