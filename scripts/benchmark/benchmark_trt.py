import os
import time
import ctypes
import psutil
import torch
import torchvision.models as models
import timm
import numpy as np
import onnx
import tensorrt as trt
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# ==========================================
# CONFIGURATION
# ==========================================
# Path to the directory where images WERE EXTRACTED
IMAGES_DIR = "/home/nvidia/magna/imagenet_val/images"
LABEL_PATH = "/home/nvidia/magna/imagenet_val/val_labels.txt"
MODELS_DIR = "/home/nvidia/magna/models"

# Benchmark Config
LIMIT_IMAGES = 1000
BATCH_SIZE = 1
DEVICE = torch.device("cuda")

TEST_MODELS = {
    "MobileNetV4": {
        "pth_path": os.path.join(MODELS_DIR, "mobilenetv4_medium.pth"),
        "onnx_path": os.path.join(MODELS_DIR, "mobilenetv4_medium_fp32.onnx"),
        "trt_path": os.path.join(MODELS_DIR, "mobilenetv4_medium_fp32.plan"),
        "loader": lambda: _load_mobilenet_v4()
    },
    "EfficientNetV2": {
        "pth_path": os.path.join(MODELS_DIR, "efficientnet_v2_s.pth"),
        "onnx_path": os.path.join(MODELS_DIR, "efficientnet_v2_s_fp32.onnx"),
        "trt_path": os.path.join(MODELS_DIR, "efficientnet_v2_s_fp32.plan"),
        "loader": lambda: _load_efficientnet_v2()
    }
}

def _load_mobilenet_v4():
    model = timm.create_model('mobilenetv4_conv_medium.e500_r256_in1k', pretrained=False)
    model.load_state_dict(torch.load(TEST_MODELS["MobileNetV4"]["pth_path"]))
    return model

def _load_efficientnet_v2():
    model = models.efficientnet_v2_s()
    model.load_state_dict(torch.load(TEST_MODELS["EfficientNetV2"]["pth_path"]))
    return model

# ==========================================
# DATASET (Loading from Extracted Directory)
# ==========================================
class ImageDirectoryDataset(Dataset):
    def __init__(self, images_dir, label_path, transform=None, limit=None,shuffle=True):
        self.images_dir = images_dir
        self.transform = transform
        
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Directory not found: {images_dir}. Please run extraction first.")

        # Find all valid image files
        valid_extensions = ('.jpeg', '.jpg', '.png', '.JPEG')
        self.image_files = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir) 
            if f.lower().endswith(valid_extensions)
        ])

        if not self.image_files:
            raise RuntimeError(f"No valid images found in {images_dir}. Extraction might be incomplete.")

        # Load labels
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                # Standard ImageNet labels are 1-indexed (1-1000). Convert to 0-indexed.
                self.labels = [int(line.strip()) - 1 for line in f if line.strip()]
        else:
            raise FileNotFoundError(f"Label file not found: {label_path}")

        # Limit images for benchmarking
        if limit:
            self.image_files = self.image_files[:limit]
            self.labels = self.labels[:limit]

        print(f"Dataset initialized with {len(self.image_files)} images from {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        return img, label

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Try to initialize the dataset
try:
    dataset = ImageDirectoryDataset(IMAGES_DIR, LABEL_PATH, transform=transform, limit=LIMIT_IMAGES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
except Exception as e:
    print(f"\n[CRITICAL ERROR] Failed to load dataset: {e}")
    print("Ensure extraction is complete and labels are present.")
    exit(1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_memory_usage():
    """Returns total RAM usage in MB for the current process"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def build_engine(onnx_path, engine_path):
    if os.path.exists(engine_path):
        print(f"TensorRT engine already exists at {engine_path}")
        return
    print(f"Building TensorRT Engine from {onnx_path} (FP32 precision)...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return 
            
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 
    
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 224, 224), (BATCH_SIZE, 3, 224, 224), (BATCH_SIZE, 3, 224, 224))
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    print("TensorRT Engine build complete.")

def run_tensorrt_benchmark(engine_path):
    print(f"\nEvaluating TensorRT engine: {engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        
    context = engine.create_execution_context()
    if hasattr(context, 'set_input_shape'):
        context.set_input_shape("input", (BATCH_SIZE, 3, 224, 224))
    else:
        context.set_binding_shape(0, (BATCH_SIZE, 3, 224, 224))
    
    libcudart = ctypes.CDLL("libcudart.so")
    cudaMalloc = libcudart.cudaMalloc
    cudaMemcpy = libcudart.cudaMemcpy
    cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    
    # Unified binding logic
    def get_idx(name):
        try: return engine.get_binding_index(name)
        except: return -1

    input_index = get_idx("input")
    output_index = get_idx("output")
    if output_index == -1: output_index = 1 - input_index

    input_shape = context.get_binding_shape(input_index)
    output_shape = context.get_binding_shape(output_index)
    
    input_bytes = np.prod(input_shape) * 4 
    output_bytes = np.prod(output_shape) * 4 
    
    d_input = ctypes.c_void_p()
    d_output = ctypes.c_void_p()
    cudaMalloc(ctypes.byref(d_input), ctypes.c_size_t(input_bytes))
    cudaMalloc(ctypes.byref(d_output), ctypes.c_size_t(output_bytes))
    
    bindings = [0] * engine.num_bindings
    bindings[input_index] = int(d_input.value)
    bindings[output_index] = int(d_output.value)
    
    output_host = np.empty(output_shape, dtype=np.float32)

    # Warmup
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    cudaMemcpy(d_input, ctypes.c_void_p(dummy_input.ctypes.data), ctypes.c_size_t(dummy_input.nbytes), cudaMemcpyHostToDevice)
    for _ in range(20):
        context.execute_v2(bindings)
    cudaDeviceSynchronize()

    correct = 0
    latencies = []
    start_mem = get_memory_usage()
    
    # Inference Run
    for images, labels in loader:
        input_np = images.numpy().astype(np.float32)
        cudaMemcpy(d_input, ctypes.c_void_p(input_np.ctypes.data), ctypes.c_size_t(input_np.nbytes), cudaMemcpyHostToDevice)
        
        start_time = time.time()
        context.execute_v2(bindings)
        cudaDeviceSynchronize()
        end_time = time.time()
        
        latencies.append(end_time - start_time)
        cudaMemcpy(ctypes.c_void_p(output_host.ctypes.data), d_output, ctypes.c_size_t(output_host.nbytes), cudaMemcpyDeviceToHost)
        
        pred = np.argmax(output_host, axis=1)
        expected = labels.numpy() if hasattr(labels, 'numpy') else labels
        correct += int(pred == expected)
        
    end_mem = get_memory_usage()

    avg_latency_ms = np.mean(latencies) * 1000
    fps = (1000 / avg_latency_ms) * BATCH_SIZE
    accuracy = (100.0 * correct / len(dataset))
    mem_used = max(0, end_mem - start_mem)
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)

    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Latency:   {avg_latency_ms:.3f} ms")
    print(f"FPS:       {fps:.2f} images/s")
    print(f"RAM Shift: {mem_used:.2f} MB")
    print(f"Engine Sz: {engine_size_mb:.2f} MB")
    
    return {
        "accuracy": accuracy,
        "latency_ms": avg_latency_ms,
        "fps": fps,
        "engine_size_mb": engine_size_mb
    }

# ==========================================
# MAIN ROUTINE
# ==========================================
if __name__ == "__main__":
    benchmark_results = {}
    
    for name, config in TEST_MODELS.items():
        print(f"\n" + "="*50)
        print(f" Benchmark: {name} ")
        print("="*50)
        
        onnx_file = config["onnx_path"]
        if not os.path.exists(onnx_file):
            print(f"ONNX model {name} missing. Exporting from PyTorch...")
            pt_model = config["loader"]().eval().to(DEVICE)
            dummy_in = torch.randn(1, 3, 224, 224).to(DEVICE)
            torch.onnx.export(
                pt_model, dummy_in, onnx_file, export_params=True, opset_version=13,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            del pt_model
            torch.cuda.empty_cache()
        else:
            print(f"Found existing ONNX model: {onnx_file}")

        build_engine(onnx_file, config["trt_path"])
        benchmark_results[name] = run_tensorrt_benchmark(config["trt_path"])

    print("\n\n" + "="*75)
    print(" FINAL SYSTEM BENCHMARK COMPARISON (FP32) ")
    print("="*75)
    print(f"{'Model':<20} | {'Acc (%)':<8} | {'Latency(ms)':<11} | {'FPS':<7} | {'Engine(MB)':<12}")
    print("-" * 75)
    for name, metrics in benchmark_results.items():
        print(f"{name:<20} | {metrics['accuracy']:.2f} | {metrics['latency_ms']:<11.3f} | {metrics['fps']:<7.2f} | {metrics['engine_size_mb']:<12.2f}")
    print("="*75)
