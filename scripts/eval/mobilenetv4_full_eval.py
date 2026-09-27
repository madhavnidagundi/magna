import os
import time
import torch
import timm
import numpy as np
import requests
import onnx
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------
# FIX: Prevent ONNX Runtime ARM thread crash
# ------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DATA_DIR = "imagenette"
BATCH_SIZE = 16
LIMIT_IMAGES = 300
MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"
DEVICE = torch.device("cuda")

# ------------------------------------------------
# Download Imagenette (small ImageNet subset)
# ------------------------------------------------
if not os.path.exists(DATA_DIR):
    print("Downloading Imagenette...")
    os.system("wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz")
    os.system("tar -xzf imagenette2-160.tgz")
    os.rename("imagenette2-160", DATA_DIR)

print("Dataset ready.")

# ------------------------------------------------
# Data loader
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
dataset = torch.utils.data.Subset(dataset, range(min(LIMIT_IMAGES, len(dataset))))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------
# Load PyTorch model
# ------------------------------------------------
print("\nLoading PyTorch model...")
model = timm.create_model(MODEL_NAME, pretrained=True)
model = model.to(DEVICE)
model.eval()

# ------------------------------------------------
# Evaluate PyTorch (GPU)
# ------------------------------------------------
print("\nRunning PyTorch inference (GPU)...")
correct = 0
total = 0
latencies = []

with torch.no_grad():
    for images, labels in tqdm(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        torch.cuda.synchronize()
        start = time.time()
        outputs = model(images)
        torch.cuda.synchronize()
        end = time.time()

        latencies.append((end - start) / images.size(0))

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

pt_acc = 100 * correct / total
pt_latency = np.mean(latencies) * 1000

print(f"\nPyTorch Accuracy: {pt_acc:.2f}%")
print(f"PyTorch Avg Latency: {pt_latency:.2f} ms")

# ------------------------------------------------
# Export to ONNX (STATIC SHAPE)
# ------------------------------------------------
print("\nExporting to ONNX (static batch=1)...")

dummy = torch.randn(1,3,224,224).to(DEVICE)

torch.onnx.export(
    model,
    dummy,
    "mobilenetv4.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"]
)

print("ONNX export done.")

# ------------------------------------------------
# ONNX Runtime CPU
# ------------------------------------------------
print("\nRunning ONNX inference (CPU)...")

session = ort.InferenceSession(
    "mobilenetv4.onnx",
    providers=["CPUExecutionProvider"]
)

print("ONNX Providers:", session.get_providers())

correct = 0
total = 0
latencies = []

for images, labels in tqdm(loader):
    # Since ONNX is static batch=1, run per image
    for i in range(images.size(0)):
        image = images[i:i+1].numpy()

        start = time.time()
        outputs = session.run(None, {"input": image})[0]
        end = time.time()

        latencies.append(end - start)

        pred = np.argmax(outputs, axis=1)[0]
        correct += int(pred == labels[i].item())
        total += 1

onnx_acc = 100 * correct / total
onnx_latency = np.mean(latencies) * 1000

print(f"\nONNX Accuracy: {onnx_acc:.2f}%")
print(f"ONNX Avg Latency: {onnx_latency:.2f} ms")

print("\nComparison Complete.")
