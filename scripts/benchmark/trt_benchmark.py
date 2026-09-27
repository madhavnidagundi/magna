import os
import time
import ctypes
import numpy as np
import onnx
import tensorrt as trt
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ==========================
# CONFIG
# ==========================

ONNX_PATH = "mobilenetv4.onnx"
TRT_ENGINE_PATH = "mobilenetv4_orin_fp32.plan"
TVM_LIB_PATH = "mobilenetv4_tvm.so"
IMAGENET_VAL_PATH = "imagenet_val"

NUM_RUNS = 50
WARMUP = 10
TARGET = tvm.target.Target("cuda")

np.random.seed(0)

# ==========================
# LOAD ONNX
# ==========================

print("\n===== Loading ONNX =====")
onnx_model = onnx.load(ONNX_PATH)

input_name = onnx_model.graph.input[0].name
input_shape = tuple(
    dim.dim_value
    for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
)

print("Input Shape:", input_shape)

# ==========================
# DATASET
# ==========================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = ImageFolder(IMAGENET_VAL_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ==========================
# TVM (CACHED)
# ==========================

print("\n===== TVM =====")

if os.path.exists(TVM_LIB_PATH):
    print("Loading cached TVM module...")
    lib = tvm.runtime.load_module(TVM_LIB_PATH)
else:
    print("Building TVM...")
    mod, params = relay.frontend.from_onnx(
        onnx_model, {input_name: input_shape}
    )

    start = time.time()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=TARGET, params=params)
    print("TVM Build Time:", time.time() - start)

    lib.export_library(TVM_LIB_PATH)

dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))

# ==========================
# TensorRT (CACHED, TRT 8.5)
# ==========================

print("\n===== TensorRT =====")

logger = trt.Logger(trt.Logger.WARNING)

if os.path.exists(TRT_ENGINE_PATH):
    print("Loading cached TensorRT engine...")
    with open(TRT_ENGINE_PATH, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
else:
    print("Building TensorRT engine...")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    engine = builder.build_engine(network, config)

    with open(TRT_ENGINE_PATH, "wb") as f:
        f.write(engine.serialize())

context = engine.create_execution_context()

input_index = engine.get_binding_index(input_name)
output_index = 1 - input_index
output_shape = engine.get_binding_shape(output_index)

# ==========================
# CUDA via ctypes
# ==========================

libcudart = ctypes.CDLL("libcudart.so")

cudaMalloc = libcudart.cudaMalloc
cudaMemcpy = libcudart.cudaMemcpy
cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# Allocate output buffer once
output_host = np.empty(output_shape, dtype=np.float32)

# ==========================
# METRICS
# ==========================

tvm_correct = 0
trt_correct = 0
total = 0

tvm_latencies = []
trt_latencies = []

# ==========================
# INFERENCE LOOP
# ==========================

for img, label in loader:

    input_data = img.numpy().astype("float32")

    # Allocate device memory for this batch
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()

    cudaMalloc(
        ctypes.byref(d_input),
        ctypes.c_size_t(int(input_data.nbytes))
    )

    cudaMalloc(
        ctypes.byref(d_output),
        ctypes.c_size_t(int(output_host.nbytes))
    )

    bindings = [0] * engine.num_bindings
    bindings[input_index] = int(d_input.value)
    bindings[output_index] = int(d_output.value)

    # ---------------- TVM ----------------

    module.set_input(input_name, input_data)

    start = time.time()
    module.run()
    tvm_latencies.append((time.time() - start) * 1000)

    tvm_output = module.get_output(0).numpy()
    tvm_pred = np.argmax(tvm_output)

    # ---------------- TRT ----------------

    cudaMemcpy(
        d_input,
        ctypes.c_void_p(input_data.ctypes.data),
        ctypes.c_size_t(int(input_data.nbytes)),
        cudaMemcpyHostToDevice
    )

    start = time.time()
    context.execute_v2(bindings)
    cudaDeviceSynchronize()
    trt_latencies.append((time.time() - start) * 1000)

    cudaMemcpy(
        ctypes.c_void_p(output_host.ctypes.data),
        d_output,
        ctypes.c_size_t(int(output_host.nbytes)),
        cudaMemcpyDeviceToHost
    )

    trt_pred = np.argmax(output_host)

    # ---------------- Accuracy ----------------

    if tvm_pred == label.item():
        tvm_correct += 1

    if trt_pred == label.item():
        trt_correct += 1

    total += 1

# ==========================
# FINAL METRICS
# ==========================

tvm_latency_mean = np.mean(tvm_latencies)
trt_latency_mean = np.mean(trt_latencies)

print("\n==============================")
print("TVM Accuracy:", tvm_correct / total)
print("TensorRT Accuracy:", trt_correct / total)
print("TVM Latency (ms):", tvm_latency_mean)
print("TensorRT Latency (ms):", trt_latency_mean)
print("TVM FPS:", 1000 / tvm_latency_mean)
print("TensorRT FPS:", 1000 / trt_latency_mean)
print("==============================")