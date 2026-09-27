import os
import time
import ctypes
import numpy as np
import torch
import timm
import onnx
import tvm
import tensorrt as trt
from tvm import relay
from tvm.contrib import graph_executor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
from thop import profile

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
DATA_DIR = "imagenette2-160"

ONNX_FP32 = "model_fp32.onnx"
ONNX_INT8 = "model_int8.onnx"
TRT_ENGINE = "model_trt_int8.plan"
TVM_LIB = "model_tvm_fp32.so"

DEVICE = torch.device("cuda")
TARGET = tvm.target.Target("cuda")

BATCH_SIZE = 1
LIMIT_IMAGES = 200

# ============================================================
# DATA
# ============================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
dataset = torch.utils.data.Subset(dataset, range(min(LIMIT_IMAGES, len(dataset))))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading PyTorch model...")
model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
model.eval()

dummy = torch.randn(1,3,224,224).to(DEVICE)

# ============================================================
# MACs / FLOPs
# ============================================================
macs, params = profile(model, inputs=(dummy,), verbose=False)
flops = macs * 2

print(f"MACs: {macs/1e6:.2f} M")
print(f"FLOPs: {flops/1e6:.2f} M")

# ============================================================
# EXPORT FP32 ONNX
# ============================================================
if not os.path.exists(ONNX_FP32):
    print("Exporting FP32 ONNX...")
    torch.onnx.export(
        model,
        dummy,
        ONNX_FP32,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

# ============================================================
# STORE INT8 ONNX (FOR RECORD ONLY)
# ============================================================
class Reader(CalibrationDataReader):
    def __init__(self, loader):
        self.loader = iter(loader)
    def get_next(self):
        try:
            images,_ = next(self.loader)
            return {"input": images.numpy()}
        except StopIteration:
            return None

if not os.path.exists(ONNX_INT8):
    print("Creating INT8 ONNX (storage only)...")
    quantize_static(
        ONNX_FP32,
        ONNX_INT8,
        Reader(loader),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )

# ============================================================
# TENSORRT INT8 CALIBRATOR
# ============================================================
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, loader):
        super().__init__()
        self.loader = iter(loader)
        self.device_input = ctypes.c_void_p()
        size = 1*3*224*224*4
        ctypes.CDLL("libcudart.so").cudaMalloc(ctypes.byref(self.device_input), size)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        try:
            images,_ = next(self.loader)
            data = images.numpy().astype(np.float32)
            ctypes.CDLL("libcudart.so").cudaMemcpy(
                self.device_input,
                data.ctypes.data,
                data.nbytes,
                1,
            )
            return [int(self.device_input.value)]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

# ============================================================
# BUILD TENSORRT INT8 ENGINE
# ============================================================
if not os.path.exists(TRT_ENGINE):
    print("Building TensorRT INT8 engine...")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX_FP32, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = ImageCalibrator(loader)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT build failed")

    with open(TRT_ENGINE, "wb") as f:
        f.write(serialized_engine)

# ============================================================
# TENSORRT BENCHMARK
# ============================================================
print("Running TensorRT INT8...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(TRT_ENGINE,"rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_idx = engine.get_binding_index("input")
output_idx = 1 - input_idx
out_shape = engine.get_binding_shape(output_idx)

libcudart = ctypes.CDLL("libcudart.so")
cudaMalloc = libcudart.cudaMalloc
cudaMemcpy = libcudart.cudaMemcpy
cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize

cudaMemcpyH2D = 1
cudaMemcpyD2H = 2

input_bytes = np.prod((1,3,224,224))*4
output_bytes = np.prod(out_shape)*4

d_input = ctypes.c_void_p()
d_output = ctypes.c_void_p()

cudaMalloc(ctypes.byref(d_input), input_bytes)
cudaMalloc(ctypes.byref(d_output), output_bytes)

bindings = [0]*engine.num_bindings
bindings[input_idx] = int(d_input.value)
bindings[output_idx] = int(d_output.value)

output_host = np.empty(out_shape, dtype=np.float32)

trt_correct = 0
trt_lat = []

for images,labels in loader:
    inp = images.numpy().astype(np.float32)

    cudaMemcpy(d_input, inp.ctypes.data, inp.nbytes, cudaMemcpyH2D)

    start = time.time()
    context.execute_v2(bindings)
    cudaDeviceSynchronize()
    end = time.time()

    trt_lat.append(end-start)

    cudaMemcpy(output_host.ctypes.data,d_output,output_host.nbytes,cudaMemcpyD2H)
    trt_correct += int(np.argmax(output_host,1)==labels.numpy())

# ============================================================
# TVM FP32 BUILD
# ============================================================
print("Compiling TVM FP32...")

onnx_model = onnx.load(ONNX_FP32)
mod, params = relay.frontend.from_onnx(
    onnx_model, {"input": (1,3,224,224)}
)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=TARGET, params=params)

lib.export_library(TVM_LIB)

dev = tvm.cuda(0)
module = graph_executor.GraphModule(lib["default"](dev))

tvm_correct = 0
tvm_lat = []

for images,labels in loader:
    inp = images.numpy().astype("float32")
    module.set_input("input", inp)

    start = time.time()
    module.run()
    dev.sync()
    end = time.time()

    tvm_lat.append(end-start)

    out = module.get_output(0).numpy()
    tvm_correct += int(np.argmax(out,1)==labels.numpy())

# ============================================================
# RESULTS
# ============================================================
def summarize(name, correct, lat, size_path):
    acc = 100*correct/len(dataset)
    latency = np.mean(lat)*1000
    fps = 1000/latency
    size = os.path.getsize(size_path)/1e6 if size_path else 0
    print(f"{name:12s} | Acc: {acc:.2f}% | Lat: {latency:.2f} ms | FPS: {fps:.2f} | Size: {size:.2f} MB")

print("\n============= FINAL RESULTS =============")
summarize("TensorRT INT8", trt_correct, trt_lat, TRT_ENGINE)
summarize("TVM FP32", tvm_correct, tvm_lat, TVM_LIB)
print(f"MACs: {macs/1e6:.2f} M")
print(f"FLOPs: {flops/1e6:.2f} M")
print("=========================================")