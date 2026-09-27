import os
import time
import torch
import timm
import numpy as np
import tensorrt as trt
import onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =============================
# CONFIG
# =============================
DATA_DIR = "imagenette"
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
DEVICE = torch.device("cuda")
BATCH_SIZE = 1
LIMIT_IMAGES = 300
WARMUP = 20

# =============================
# Dataset
# =============================
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
# 1️⃣ PyTorch Eager FP32
# ============================================================
print("\nRunning PyTorch Eager FP32...")
model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
model.eval()

dummy = torch.randn((1,3,224,224), device=DEVICE)

with torch.no_grad():
    for _ in range(WARMUP):
        model(dummy)

torch.cuda.synchronize()

pt_correct = 0
pt_lat = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        start = time.time()
        outputs = model(images)
        torch.cuda.synchronize()
        end = time.time()

        pt_lat.append(end - start)
        preds = outputs.argmax(dim=1)
        pt_correct += (preds == labels).sum().item()

pt_acc = 100 * pt_correct / num_images
pt_latency = np.mean(pt_lat) * 1000
pt_fps = 1000 / pt_latency


# ============================================================
# 2️⃣ PyTorch → ONNX → TensorRT (build in Python)
# ============================================================
print("\nBuilding TensorRT engine from ONNX inside Python...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("mobilenetv4.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

engine_python = builder.build_engine(network, config)
context_python = engine_python.create_execution_context()

py_correct = 0
py_lat = []

for images, labels in loader:
    input_tensor = images.to(DEVICE)
    output_tensor = torch.empty((BATCH_SIZE, 1000), device=DEVICE)

    bindings = [
        int(input_tensor.data_ptr()),
        int(output_tensor.data_ptr())
    ]

    start = time.time()
    context_python.execute_v2(bindings)
    torch.cuda.synchronize()
    end = time.time()

    py_lat.append(end - start)
    pred = torch.argmax(output_tensor, dim=1)
    py_correct += int(pred.cpu().numpy() == labels.numpy())

py_acc = 100 * py_correct / num_images
py_latency = np.mean(py_lat) * 1000
py_fps = 1000 / py_latency


# ============================================================
# 3️⃣ ONNX → TensorRT (prebuilt engine)
# ============================================================
print("\nLoading prebuilt TensorRT engine...")

with open("mobilenetv4_fp32.plan", "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

trt_correct = 0
trt_lat = []

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

    trt_lat.append(end - start)
    pred = torch.argmax(output_tensor, dim=1)
    trt_correct += int(pred.cpu().numpy() == labels.numpy())

trt_acc = 100 * trt_correct / num_images
trt_latency = np.mean(trt_lat) * 1000
trt_fps = 1000 / trt_latency


# ============================================================
# RESULTS
# ============================================================
print("\n=========================================")
print("         THREE-WAY FP32 COMPARISON")
print("=========================================\n")

print("1) PyTorch Eager")
print(f"Accuracy : {pt_acc:.2f}%")
print(f"Latency  : {pt_latency:.3f} ms")
print(f"FPS      : {pt_fps:.2f}\n")

print("2) PyTorch → ONNX → TensorRT (Python-built)")
print(f"Accuracy : {py_acc:.2f}%")
print(f"Latency  : {py_latency:.3f} ms")
print(f"FPS      : {py_fps:.2f}\n")

print("3) ONNX → TensorRT (trtexec-built)")
print(f"Accuracy : {trt_acc:.2f}%")
print(f"Latency  : {trt_latency:.3f} ms")
print(f"FPS      : {trt_fps:.2f}\n")

print("Speedups vs PyTorch Eager:")
print(f"Python-built TRT  : {pt_latency / py_latency:.2f}x")
print(f"Prebuilt TRT      : {pt_latency / trt_latency:.2f}x")
