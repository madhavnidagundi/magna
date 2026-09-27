import ctypes
import numpy as np
import tensorrt as trt
import time

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open("models/mobilenetv4_nvidia_optimized_int8.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

libcudart = ctypes.CDLL("libcudart.so")
cudaMalloc = libcudart.cudaMalloc
cudaMemcpy = libcudart.cudaMemcpy
cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize
H2D, D2H = 1, 2

in_bytes = int(np.prod((1, 3, 224, 224)) * 4)
out_bytes = int(np.prod((1, 1000)) * 4)
d_in, d_out = ctypes.c_void_p(), ctypes.c_void_p()
cudaMalloc(ctypes.byref(d_in), ctypes.c_size_t(in_bytes))
cudaMalloc(ctypes.byref(d_out), ctypes.c_size_t(out_bytes))

context.set_input_shape("x", (1, 3, 224, 224))
context.set_tensor_address("x", int(d_in.value))
context.set_tensor_address("output", int(d_out.value))

h_in = np.random.randn(1, 3, 224, 224).astype(np.float32)
h_in = np.ascontiguousarray(h_in)
cudaMemcpy(d_in, ctypes.c_void_p(h_in.ctypes.data), ctypes.c_size_t(h_in.nbytes), H2D)

ret = context.execute_async_v3(0)
cudaDeviceSynchronize()

h_out = np.empty((1, 1000), dtype=np.float32)
cudaMemcpy(ctypes.c_void_p(h_out.ctypes.data), d_out, ctypes.c_size_t(h_out.nbytes), D2H)

print("Execute returned:", ret)
print("Output sample:", h_out[0, :10])
print("Output min:", h_out.min(), "max:", h_out.max(), "mean:", h_out.mean())
