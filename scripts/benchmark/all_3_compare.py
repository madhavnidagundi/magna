import os
import time
import ctypes
import urllib.request
import tarfile
import torch
import timm
import numpy as np
import onnx
import tensorrt as trt
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "imagenette2-160"
DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
ONNX_PATH = "mobilenetv4.onnx"
TRT_ENGINE_PATH = "mobilenetv4.plan"
TVM_LIB_PATH = "mobilenetv4_tvm.so"

DEVICE = torch.device("cuda")
BATCH_SIZE = 1
LIMIT_IMAGES = 2000
TARGET = tvm.target.Target("cuda")

# -----------------------------
# Download Imagenette if missing
# -----------------------------
if not os.path.exists(DATA_DIR):
    print("Downloading Imagenette dataset...")
    urllib.request.urlretrieve(DATA_URL, "imagenette.tgz")
    with tarfile.open("imagenette.tgz", "r:gz") as tar:
        tar.extractall()
    print("Download complete.")

# -----------------------------
# Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
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
pt_fps = 1000 / pt_latency

print(f"\nPyTorch Accuracy: {pt_acc:.2f}%")
print(f"PyTorch Avg Latency: {pt_latency:.3f} ms")
print(f"PyTorch FPS: {pt_fps:.2f}")

# -----------------------------
# Export ONNX if missing
# -----------------------------
if not os.path.exists(ONNX_PATH):
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=13,
        input_names=["input"],
        output_names=["output"]
    )

onnx_model = onnx.load(ONNX_PATH)

# -----------------------------
# 2️⃣ TVM Benchmark (FIXED TIMING)
# -----------------------------
print("\nLoading TVM model...")

if os.path.exists(TVM_LIB_PATH):
    lib = tvm.runtime.load_module(TVM_LIB_PATH)
else:
    mod, params = relay.frontend.from_onnx(
        onnx_model, {"input": (1, 3, 224, 224)}
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=TARGET, params=params)
    lib.export_library(TVM_LIB_PATH)

dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))

tvm_correct = 0
tvm_latencies = []

for images, labels in loader:
    input_np = images.numpy().astype("float32")

    module.set_input("input", input_np)

    start = time.time()
    module.run()
    dev.sync()   # IMPORTANT FIX
    end = time.time()

    tvm_latencies.append(end - start)

    output = module.get_output(0).numpy()
    pred = np.argmax(output, axis=1)
    tvm_correct += int(pred == labels.numpy())

tvm_acc = 100 * tvm_correct / len(dataset)
tvm_latency = np.mean(tvm_latencies) * 1000
tvm_fps = 1000 / tvm_latency

print(f"\nTVM Accuracy: {tvm_acc:.2f}%")
print(f"TVM Avg Latency: {tvm_latency:.3f} ms")
print(f"TVM FPS: {tvm_fps:.2f}")

# -----------------------------
# 3️⃣ TensorRT Benchmark (FIXED + REALISTIC)
# -----------------------------
print("\nLoading TensorRT engine...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if not os.path.exists(TRT_ENGINE_PATH):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_PATH, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    engine = builder.build_engine(network, config)

    with open(TRT_ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())
else:
    with open(TRT_ENGINE_PATH, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_index = engine.get_binding_index("input")
output_index = 1 - input_index
output_shape = engine.get_binding_shape(output_index)

# CUDA via ctypes
libcudart = ctypes.CDLL("libcudart.so")
cudaMalloc = libcudart.cudaMalloc
cudaMemcpy = libcudart.cudaMemcpy
cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# Allocate ONCE
input_bytes = np.prod((1, 3, 224, 224)) * 4
output_bytes = np.prod(output_shape) * 4

d_input = ctypes.c_void_p()
d_output = ctypes.c_void_p()

cudaMalloc(ctypes.byref(d_input), ctypes.c_size_t(input_bytes))
cudaMalloc(ctypes.byref(d_output), ctypes.c_size_t(output_bytes))

bindings = [0] * engine.num_bindings
bindings[input_index] = int(d_input.value)
bindings[output_index] = int(d_output.value)

output_host = np.empty(output_shape, dtype=np.float32)

trt_correct = 0
trt_latencies = []

for images, labels in loader:
    input_np = images.numpy().astype(np.float32)

    cudaMemcpy(
        d_input,
        ctypes.c_void_p(input_np.ctypes.data),
        ctypes.c_size_t(input_np.nbytes),
        cudaMemcpyHostToDevice
    )

    start = time.time()
    context.execute_v2(bindings)
    cudaDeviceSynchronize()
    end = time.time()

    trt_latencies.append(end - start)

    cudaMemcpy(
        ctypes.c_void_p(output_host.ctypes.data),
        d_output,
        ctypes.c_size_t(output_host.nbytes),
        cudaMemcpyDeviceToHost
    )

    pred = np.argmax(output_host, axis=1)
    trt_correct += int(pred == labels.numpy())

trt_acc = 100 * trt_correct / len(dataset)
trt_latency = np.mean(trt_latencies) * 1000
trt_fps = 1000 / trt_latency

print(f"\nTensorRT Accuracy: {trt_acc:.2f}%")
print(f"TensorRT Avg Latency: {trt_latency:.3f} ms")
print(f"TensorRT FPS: {trt_fps:.2f}")

print("\n=============================")
print("FINAL COMPARISON")
print("=============================")
print(f"PyTorch  | Acc: {pt_acc:.2f}% | Latency: {pt_latency:.3f} ms | FPS: {pt_fps:.2f}")
print(f"TVM      | Acc: {tvm_acc:.2f}% | Latency: {tvm_latency:.3f} ms | FPS: {tvm_fps:.2f}")
print(f"TensorRT | Acc: {trt_acc:.2f}% | Latency: {trt_latency:.3f} ms | FPS: {trt_fps:.2f}")
print("=============================")