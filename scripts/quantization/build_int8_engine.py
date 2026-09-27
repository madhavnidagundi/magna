import os
import ctypes
import numpy as np
import tensorrt as trt
import torch
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
ONNX_MODEL = "/home/thor/Documents/magna/mi-isal/models/mobilenetv4_medium.onnx"
ENGINE_OUTPUT = "/home/thor/Documents/magna/mi-isal/models/mobilenetv4_nvidia_optimized_int8.engine"
CALIBRATION_CACHE = "/home/thor/Documents/magna/mi-isal/models/mobilenetv4_medium_calibration.cache"

IMAGES_DIR = "/home/thor/Documents/magna/mi-isal/imagenet_val/archive/imagenet-val"
CALIBRATION_BATCH_SIZE = 32
CALIBRATION_BATCHES = 20  # Total 640 images for calibration
INPUT_SHAPE = (3, 224, 224)
INPUT_NAME = "x"  # Default onnx export input name for PyTorch
# ──────────────────────────────────────────────────────────────────────────────

class ImageNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, images_dir, batch_size, total_batches, cache_file):
        super().__init__()
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.cache_file = cache_file
        
        from pathlib import Path
        all_paths = sorted([str(p) for p in Path(images_dir).rglob('*.JPEG')])
        self.files = all_paths[:self.batch_size * self.total_batches]
        self.images_dir = ""
        
        self.batch_idx = 0
        
        # Allocate GPU memory using PyTorch
        self.device_mem = torch.empty((self.batch_size, *INPUT_SHAPE), dtype=torch.float32, device='cuda')

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= self.total_batches:
            return None
            
        start_idx = self.batch_idx * self.batch_size
        batch_files = self.files[start_idx : start_idx + self.batch_size]
        
        batch_data = np.zeros((self.batch_size, *INPUT_SHAPE), dtype=np.float32)
        
        for i, fname in enumerate(batch_files):
            # Same preprocessing as evaluation
            img_path = fname # fname is now an absolute path
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            w, h = img.size
            l, t = (w - 224) // 2, (h - 224) // 2
            img = img.crop((l, t, l + 224, t + 224))
            
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            arr = (arr - mean) / std
            batch_data[i] = arr

        # Copy data to GPU
        torch_data = torch.from_numpy(batch_data).to('cuda')
        self.device_mem.copy_(torch_data)
        
        self.batch_idx += 1
        print(f"Calibrating batch {self.batch_idx}/{self.total_batches}")
        
        # Return memory pointer
        return [int(self.device_mem.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_int8_engine():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    print("Parsing ONNX model {}...".format(ONNX_MODEL))
    with open(ONNX_MODEL, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Add optimization profile
    print("Setting optimization profile for dynamic inputs...")
    profile = builder.create_optimization_profile()
    
    # Try to find input name dynamically
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Found input tensor: {input_name}")
    
    profile.set_shape(
        input_name, 
        (1, *INPUT_SHAPE),                            # min
        (CALIBRATION_BATCH_SIZE, *INPUT_SHAPE),       # opt
        (CALIBRATION_BATCH_SIZE, *INPUT_SHAPE)        # max
    )
    config.add_optimization_profile(profile)

    # Enable INT8 and set calibrator
    print("Setting up INT8 calibration...")
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = ImageNetCalibrator(
        IMAGES_DIR, 
        CALIBRATION_BATCH_SIZE, 
        CALIBRATION_BATCHES, 
        CALIBRATION_CACHE
    )

    # Some memory limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1 << 30))  # 2 GB

    print("Building TensorRT engine (this may take a few minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("Failed to build engine")
        return

    print(f"Saving compiled engine to {ENGINE_OUTPUT}")
    with open(ENGINE_OUTPUT, "wb") as f:
        f.write(engine_bytes)
    print("Optimization and serialization completed!")

if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL):
        print(f"ONNX model not found: {ONNX_MODEL}")
    else:
        build_int8_engine()
