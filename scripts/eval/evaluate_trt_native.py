"""
evaluate_trt_native.py
======================
Evaluates MobileNetV4 and EfficientNetV2 TensorRT engines (.plan files)
on the ImageNet validation set.

Labels: Uses ImageNet-ReaL labels embedded in timm (Beyer et al. 2020),
        which are correctly indexed for PyTorch models (0-999 alphabetical).

No pycuda required — uses ctypes + CUDA runtime directly.
"""

import os
import time
import json
import pkgutil
import ctypes
import numpy as np
import tensorrt as trt
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
IMAGES_DIR = "/home/thor/Documents/magna/mi-isal/imagenet_val/archive/imagenet-val"

MODELS = {
    "MobileNetV4-Medium (INT8)":   "/home/thor/Documents/magna/mi-isal/models/mobilenetv4_nvidia_optimized_int8.engine",
    "EfficientNetV2-S   (INT8)":   "/home/thor/Documents/magna/mi-isal/models/efficientnet_v2_s_nvidia_optimized_int8.engine",
}

NUM_IMAGES  = 50000   # change to 50000 for full eval
WARMUP_RUNS = 20
# ──────────────────────────────────────────────────────────────────────────────


def load_real_labels(images_dir: str):
    """
    Load ImageNet-ReaL labels (embedded in timm) for images present on disk.
    Returns (image_paths, labels) sorted by filename.
    """
    from pathlib import Path
    image_paths_map = {p.name: str(p) for p in Path(images_dir).rglob('ILSVRC2012_val_*.JPEG')}
    
    real_bytes = pkgutil.get_data('timm.data', os.path.join('_info', 'imagenet_real_labels.json'))
    real_list  = json.loads(real_bytes.decode('utf-8'))          # list of 50 000 label lists

    paths, labels = [], []
    for i, lbl_set in enumerate(real_list):
        fname = f'ILSVRC2012_val_{i + 1:08d}.JPEG'
        fpath = image_paths_map.get(fname)
        if lbl_set and fpath and os.path.getsize(fpath) > 0:
            paths.append(fpath)
            labels.append(lbl_set)          # keep full multi-label set
        if len(paths) >= NUM_IMAGES:
            break

    return paths, labels


def preprocess(image_path: str) -> np.ndarray:
    """Standard ImageNet preprocessing (matches torchvision defaults)."""
    img  = Image.open(image_path).convert('RGB')
    img  = img.resize((256, 256), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    img  = img.crop((l, t, l + 224, t + 224))
    arr  = np.asarray(img, dtype=np.float32) / 255.0
    arr  = np.transpose(arr, (2, 0, 1))                         # HWC -> CHW
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr  = (arr - mean) / std
    return np.ascontiguousarray(arr[np.newaxis], dtype=np.float32)  # (1,3,224,224)


def evaluate_engine(engine_path: str, image_paths: list, real_labels: list) -> dict:
    """Run inference with TensorRT and compute Top-1/Top-5 accuracy."""

    # ── Load engine ──────────────────────────────────────────────────────────
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)   # show warnings
    runtime    = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # ── CUDA via ctypes ───────────────────────────────────────────────────────
    libcudart = ctypes.CDLL("libcudart.so")
    cudaMalloc          = libcudart.cudaMalloc
    cudaFree            = libcudart.cudaFree
    cudaMemcpy          = libcudart.cudaMemcpy
    cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize
    H2D, D2H            = 1, 2

    in_bytes  = int(np.prod((1, 3, 224, 224)) * 4)
    out_bytes = int(np.prod((1, 1000))         * 4)

    d_in  = ctypes.c_void_p()
    d_out = ctypes.c_void_p()
    cudaMalloc(ctypes.byref(d_in),  ctypes.c_size_t(in_bytes))
    cudaMalloc(ctypes.byref(d_out), ctypes.c_size_t(out_bytes))
    bindings = [int(d_in.value), int(d_out.value)]

    h_out = np.empty((1, 1000), dtype=np.float32)

    # ── Warmup ────────────────────────────────────────────────────────────────
    # Ensure dynamic shapes are set
    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    context.set_input_shape(in_name, (1, 3, 224, 224))
    context.set_tensor_address(in_name, int(d_in.value))
    context.set_tensor_address(out_name, int(d_out.value))

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy = np.ascontiguousarray(dummy)
    cudaMemcpy(d_in, ctypes.c_void_p(dummy.ctypes.data),
               ctypes.c_size_t(dummy.nbytes), H2D)
    for _ in range(WARMUP_RUNS):
        context.execute_async_v3(0)
    cudaDeviceSynchronize()

    # ── Evaluation ────────────────────────────────────────────────────────────
    top1_correct = 0
    top5_correct = 0
    latencies    = []
    processed    = 0

    for img_path, lbl_set in zip(image_paths, real_labels):
        try:
            h_in = preprocess(img_path)
        except Exception:
            continue

        cudaMemcpy(d_in, ctypes.c_void_p(h_in.ctypes.data),
                   ctypes.c_size_t(h_in.nbytes), H2D)

        t0 = time.perf_counter()
        context.execute_async_v3(0)
        cudaDeviceSynchronize()
        latencies.append(time.perf_counter() - t0)

        cudaMemcpy(ctypes.c_void_p(h_out.ctypes.data), d_out,
                   ctypes.c_size_t(h_out.nbytes), D2H)

        flat     = h_out.flatten()
        top5_idx = np.argsort(flat)[-5:][::-1].tolist()
        top1_idx = top5_idx[0]

        # ReaL: correct if ANY of the model's top-k overlaps with ReaL label set
        if top1_idx in lbl_set:
            top1_correct += 1
        if any(p in lbl_set for p in top5_idx):
            top5_correct += 1

        processed += 1
        if processed % 200 == 0:
            print(f"    {processed}/{len(image_paths)} images …"
                  f"  Top-1 so far: {100*top1_correct/processed:.1f}%")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cudaFree(d_in)
    cudaFree(d_out)

    avg_ms  = np.mean(latencies) * 1000
    p99_ms  = np.percentile(latencies, 99) * 1000
    fps     = 1000 / avg_ms

    return {
        "processed":  processed,
        "top1":       top1_correct / processed * 100,
        "top5":       top5_correct / processed * 100,
        "avg_ms":     avg_ms,
        "p99_ms":     p99_ms,
        "fps":        fps,
    }


def main():
    print("=" * 62)
    print("  TensorRT Native Evaluation — ImageNet-ReaL Labels")
    print("=" * 62)

    print(f"\nLoading ImageNet-ReaL labels for up to {NUM_IMAGES} images …")
    image_paths, real_labels = load_real_labels(IMAGES_DIR)
    print(f"  Loaded {len(image_paths)} valid images with ReaL labels.")

    results = {}
    for model_name, plan_path in MODELS.items():
        if not os.path.exists(plan_path):
            print(f"\n[SKIP] {model_name}: engine not found at {plan_path}")
            continue

        print(f"\n{'─'*62}")
        print(f"  Evaluating: {model_name}")
        print(f"  Engine    : {os.path.basename(plan_path)}")
        print(f"{'─'*62}")

        r = evaluate_engine(plan_path, image_paths, real_labels)
        results[model_name] = r

        print(f"\n  Images Evaluated : {r['processed']}")
        print(f"  Top-1 Accuracy   : {r['top1']:.2f}%")
        print(f"  Top-5 Accuracy   : {r['top5']:.2f}%")
        print(f"  Avg Latency      : {r['avg_ms']:.2f} ms")
        print(f"  P99 Latency      : {r['p99_ms']:.2f} ms")
        print(f"  Throughput       : {r['fps']:.1f} FPS")

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*62}")
        print(f"  SUMMARY (ImageNet-ReaL, n={NUM_IMAGES})")
        print(f"{'='*62}")
        print(f"  {'Model':<32} {'Top-1':>6} {'Top-5':>6} {'FPS':>7} {'Latency':>9}")
        print(f"  {'-'*60}")
        for name, r in results.items():
            print(f"  {name:<32} {r['top1']:>5.2f}% {r['top5']:>5.2f}% "
                  f"{r['fps']:>7.1f} {r['avg_ms']:>7.2f} ms")
        print(f"{'='*62}")


if __name__ == "__main__":
    main()
