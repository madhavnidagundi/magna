import os
import time
import torch
import timm
import numpy as np
import tensorrt as trt
import ctypes
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "imagenette"
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
DEVICE = torch.device("cuda")
BATCH_SIZE = 1
LIMIT_IMAGES = 200

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

# -----------------------------
# 1️⃣ PyTorch Benchmark
# -----------------------------
print("\nLoading PyTorch model...")
model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
model.eval()

pt_correct = 0
pt_latencies = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        torch.cuda.synchronize()
        start = time.time()
        outputs = model(images)
        torch.cuda.synchronize()
        end = time.time()

        pt_latencies.append(end - start)

        preds = outputs.argmax(dim=1)
        pt_correct += (preds == labels).sum().item()

pt_acc = 100 * pt_correct / len(dataset)
pt_latency = np.mean(pt_latencies) * 1000

print(f"\nPyTorch Accuracy: {pt_acc:.2f}%")
print(f"PyTorch Avg Latency: {pt_latency:.3f} ms")

# -----------------------------
# 2️⃣ TensorRT Benchmark
# -----------------------------
print("\nLoading TensorRT engine...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open("mobilenetv4.plan", "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

import pycuda.driver as cuda
import pycuda.autoinit

input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
d_output = cuda.mem_alloc(np.prod(output_shape) * 4)
stream = cuda.Stream()

trt_correct = 0
trt_latencies = []

for images, labels in loader:
    input_np = images.numpy().astype(np.float32)

    cuda.memcpy_htod_async(d_input, input_np, stream)

    start = time.time()
    context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    stream.synchronize()
    end = time.time()

    trt_latencies.append(end - start)

    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    pred = np.argmax(output, axis=1)
    trt_correct += int(pred == labels.numpy())

trt_acc = 100 * trt_correct / len(dataset)
trt_latency = np.mean(trt_latencies) * 1000

print(f"\nTensorRT Accuracy: {trt_acc:.2f}%")
print(f"TensorRT Avg Latency: {trt_latency:.3f} ms")

print("\nComparison Complete.")
