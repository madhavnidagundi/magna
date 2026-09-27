import os
import time
import torch
import timm
import numpy as np
import tensorrt as trt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "imagenette"
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
DEVICE = torch.device("cuda")
BATCH_SIZE = 1
LIMIT_IMAGES = 300
WARMUP = 20

# -----------------------------
# Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
dataset = torch.utils.data.Subset(dataset, range(min(LIMIT_IMAGES, len(dataset))))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

num_images = len(dataset)

# ============================================================
# 1️⃣ PyTorch FP32
# ============================================================
print("\nLoading PyTorch FP32...")
model_fp32 = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
model_fp32.eval()

# Warmup
dummy = torch.randn((1,3,224,224), device=DEVICE)
with torch.no_grad():
    for _ in range(WARMUP):
        model_fp32(dummy)

torch.cuda.synchronize()

correct = 0
latencies = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        start = time.time()
        outputs = model_fp32(images)
        torch.cuda.synchronize()
        end = time.time()

        latencies.append(end - start)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

pt32_acc = 100 * correct / num_images
pt32_latency = np.mean(latencies) * 1000
pt32_fps = 1000 / pt32_latency

# ============================================================
# 2️⃣ PyTorch FP16 (AMP)
# ============================================================
print("\nRunning PyTorch FP16 (AMP)...")
model_fp16 = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
model_fp16.eval()

with torch.no_grad():
    for _ in range(WARMUP):
        with torch.cuda.amp.autocast():
            model_fp16(dummy)

torch.cuda.synchronize()

correct = 0
latencies = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        start = time.time()
        with torch.cuda.amp.autocast():
            outputs = model_fp16(images)
        torch.cuda.synchronize()
        end = time.time()

        latencies.append(end - start)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

pt16_acc = 100 * correct / num_images
pt16_latency = np.mean(latencies) * 1000
pt16_fps = 1000 / pt16_latency

# ============================================================
# 3️⃣ TensorRT FP16
# ============================================================
print("\nRunning TensorRT FP16...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("mobilenetv4.plan", "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

correct = 0
latencies = []

for images, labels in loader:
    input_tensor = images.to(DEVICE)
    output_tensor = torch.empty((BATCH_SIZE, 1000), device=DEVICE)

    bindings = [
        int(input_tensor.data_ptr()),
        int(output_tensor.data_ptr())
    ]

    start = time.time()
    context.execute_v2(bindings)
    torch.cuda.synchronize()
    end = time.time()

    latencies.append(end - start)

    pred = torch.argmax(output_tensor, dim=1)
    correct += int(pred.cpu().numpy() == labels.numpy())

trt_acc = 100 * correct / num_images
trt_latency = np.mean(latencies) * 1000
trt_fps = 1000 / trt_latency

# ============================================================
# RESULTS
# ============================================================
print("\n==============================")
print("        FINAL RESULTS")
print("==============================\n")

print(f"PyTorch FP32  | Acc: {pt32_acc:.2f}% | Latency: {pt32_latency:.3f} ms | FPS: {pt32_fps:.2f}")
print(f"PyTorch FP16  | Acc: {pt16_acc:.2f}% | Latency: {pt16_latency:.3f} ms | FPS: {pt16_fps:.2f}")
print(f"TensorRT FP16 | Acc: {trt_acc:.2f}% | Latency: {trt_latency:.3f} ms | FPS: {trt_fps:.2f}")

print("\nSpeedups (vs PyTorch FP32):")
print(f"PyTorch FP16 Speedup : {pt32_latency / pt16_latency:.2f}x")
print(f"TensorRT Speedup     : {pt32_latency / trt_latency:.2f}x")
