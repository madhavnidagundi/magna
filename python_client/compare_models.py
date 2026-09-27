import os
import re
import numpy as np
import onnx_tool

MODELS_DIR = "/home/nvidia/magna/models"
LOGS_DIR = "/home/nvidia/magna/middleware/logs"

# The four ONNX models we want to compare:
ONNX_MODELS = {
    "MobileNetV4 (FP32)": os.path.join(MODELS_DIR, "mobilenetv4_medium_fp32.onnx"),
    # It looks like the original INT8 is compiled to TRT .engine directly from the FP32.onnx over calibration,
    # but we will just compare whatever ONNX models are present, or at least the raw size of .plan files.
    "EfficientNetV2 (FP32)": os.path.join(MODELS_DIR, "efficientnet_v2_s_fp32.onnx"),
}

PLAN_FILES = {
    "MobileNetV4 FP32 Plan": os.path.join(MODELS_DIR, "mobilenetv4_medium_fp32.plan"),
    "MobileNetV4 INT8 Plan": os.path.join(MODELS_DIR, "mobilenetv4_medium_int8.plan"),
    "EfficientNetV2 FP32 Plan": os.path.join(MODELS_DIR, "efficientnet_v2_s_fp32.plan"),
    "EfficientNetV2 INT8 Plan": os.path.join(MODELS_DIR, "efficientnet_v2_s_int8.plan"),
}

LOG_FILES = {
    "MobileNetV4 FP32 Inference": os.path.join(LOGS_DIR, "magna_20260304_151302.log"),
    "MobileNetV4 INT8 Inference": os.path.join(LOGS_DIR, "magna_int8_20260306_141821.log"),
}

def analyze_onnx(name, path):
    if not os.path.exists(path):
        return None
    
    # Calculate file size
    size_mb = os.path.getsize(path) / (1024 * 1024)
    
    try:
        # Provide concrete shape for dynamic ONNX batch dimensions
        import io
        import sys
        
        # Capture stdout to prevent onnx-tool from blasting large tables to console
        old_out = sys.stdout
        sys.stdout = io.StringIO()
        
        from onnx_tool import model_profile
        model_info = model_profile(path, dynamic_shapes={'input': [1, 3, 224, 224]})
        
        sys.stdout = old_out
        
        # onnx-tool model_profile returns None, but saves global states or if we use model_profile we get nothing 
        # let's just initialize the graph safely:
        graph = onnx_tool.Graph(path)
        graph.shape_infer({'input': [1, 3, 224, 224]})
        graph.profile()
        
        macs = graph.macs / 1e6 # GMACs / MMACs
        params = graph.params / 1e6 # M parameters
        memory = graph.memory / (1024 * 1024) # Memory in MB
        
        return {
            "size_mb": size_mb,
            "macs_m": macs,
            "params_m": params,
            "memory_mb": memory
        }
    except Exception as e:
        sys.stdout = old_out if 'old_out' in locals() else sys.stdout
        print(f"Failed to analyze {name}: {e}")
        return {"size_mb": size_mb, "macs_m": 0, "params_m": 0, "memory_mb": 0}

def analyze_logs(log_path):
    if not os.path.exists(log_path):
        return None
        
    latencies = []
    
    # regex to match: Inference completed in 7.427 ms
    regex = re.compile(r"Inference completed in ([\d\.]+) ms")
    
    with open(log_path, 'r') as f:
        for line in f:
            match = regex.search(line)
            if match:
                latencies.append(float(match.group(1)))
                
    if len(latencies) == 0:
        return None
        
    # skip warmup
    if len(latencies) > 100:
        latencies = latencies[10:]
        
    return {
        "count": len(latencies),
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "fps_approx": 1000.0 / np.mean(latencies)
    }

def print_separator():
    print("-" * 100)

print_separator()
print(f"{'Model Profile (Architectural Complexity)':^100}")
print_separator()
print(f"{'Model Name':<25} | {'Size (MB)':<12} | {'Params (M)':<12} | {'MACs (M)':<12} | {'Memory Footprint (MB)':<25}")
print_separator()

for name, path in ONNX_MODELS.items():
    res = analyze_onnx(name, path)
    if res:
        print(f"{name:<25} | {res['size_mb']:<12.2f} | {res['params_m']:<12.2f} | {res['macs_m']:<12.2f} | {res['memory_mb']:<25.2f}")

print("\n")
print_separator()
print(f"{'Compiled TensorRT Engine Sizes vs Native ONNX':^100}")
print_separator()
for name, path in PLAN_FILES.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"{name:<25} -> {size:.2f} MB")

print("\n")
print_separator()
print(f"{'Jetson Orin Engine Actual Pipeline Benchmarks (49000 Images)':^100}")
print_separator()
print(f"{'Log / Model':<30} | {'Images Tested':<15} | {'Avg Latency':<15} | {'Est. GPU FPS':<15}")
print_separator()

for name, path in LOG_FILES.items():
    stats = analyze_logs(path)
    if stats:
        print(f"{name:<30} | {stats['count']:<15} | {stats['mean_ms']:<11.2f} ms | {stats['fps_approx']:<15.2f}")

print_separator()

