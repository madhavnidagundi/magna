import os
import argparse
import subprocess
import numpy as np
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to INT8 QNN DLC using QAIRT")
    parser.add_argument("--onnx", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--dataset", type=str, required=True, help="Path to calibration dataset (must have subfolders for classes)")
    parser.add_argument("--output_dlc", type=str, required=True, help="Output INT8 DLC path")
    parser.add_argument("--num_images", type=int, default=100, help="Number of images to use for calibration")
    parser.add_argument("--input_dim", type=str, nargs=2, action="append", metavar=('NAME', 'DIMS'), help="Input name and dimensions (e.g. --input_dim input 1,3,224,224)")
    return parser.parse_args()

from torch.utils.data import Dataset
from PIL import Image
import glob

class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG'):
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

def get_crop_size(input_dim):
    """Extract (height, width) from the first --input_dim entry's NCHW dims
    string. There is no hardcoded default shape — --input_dim must be given
    for calibration image preprocessing to match the model's real input
    size; a mismatched crop size would silently produce wrong calibration
    data for any model that isn't exactly 224x224."""
    if not input_dim:
        raise ValueError(
            "--input_dim is required to preprocess calibration images "
            "(e.g. --input_dim input 1,3,224,224) — there is no default "
            "input shape."
        )
    _, dims = input_dim[0]
    parts = [int(p) for p in dims.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected NCHW dims (4 comma-separated values), got: {dims}")
    _, _, height, width = parts
    return height, width


def prepare_calibration_data(dataset_path, num_images, output_dir, input_dim):
    # 1. Check if the directory already has .raw files and a .txt list
    existing_raws = glob.glob(os.path.join(dataset_path, '**', '*.raw'), recursive=True)
    if existing_raws:
        print(f"Found {len(existing_raws)} existing .raw calibration files in {dataset_path}!")
        print("Generating a fresh calibration list to ensure correct container paths...")

        os.makedirs(output_dir, exist_ok=True)
        input_list_path = os.path.join(output_dir, "input_list.txt")
        with open(input_list_path, "w") as f:
            for raw in existing_raws[:num_images]:
                f.write(f"{os.path.abspath(raw)}\n")
        return input_list_path

    # 2. Otherwise, preprocess images from the directory
    os.makedirs(output_dir, exist_ok=True)

    crop_h, crop_w = get_crop_size(input_dim)
    # Standard ImageNet resize:crop ratio (256:224), scaled to the model's
    # actual crop size instead of hardcoding 256/224.
    resize_size = (round(crop_h * 256 / 224), round(crop_w * 256 / 224))
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop((crop_h, crop_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading image dataset from {dataset_path}...")
    dataset = FlatImageDataset(dataset_path, transform=transform)

    raw_files = []
    limit = min(num_images, len(dataset))
    if limit == 0:
        raise ValueError(f"No images or .raw files found in {dataset_path}!")

    print(f"Preparing {limit} calibration images into {output_dir}...")

    for i in range(limit):
        img_tensor, _ = dataset[i]
        # QNN expects flattened float32 array
        img_np = img_tensor.numpy().astype(np.float32)

        raw_path = os.path.abspath(os.path.join(output_dir, f"calib_{i}.raw"))
        img_np.tofile(raw_path)
        raw_files.append(raw_path)

    input_list_path = os.path.join(output_dir, "input_list.txt")
    with open(input_list_path, "w") as f:
        for raw_file in raw_files:
            f.write(f"{raw_file}\n")

    return input_list_path

def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    workspace = os.path.dirname(os.path.abspath(args.output_dlc))
    if not workspace:
        workspace = "."

    calib_dir = os.path.join(workspace, "calib_data")
    input_list_path = prepare_calibration_data(args.dataset, args.num_images, calib_dir, args.input_dim)

    fp32_dlc = args.output_dlc.replace(".dlc", "") + "_fp32.dlc"

    print("\n=======================================================")
    print(" Step 1: Converting ONNX to FP32 DLC")
    print("=======================================================")
    conv_cmd = [
        "qairt-converter",
        "--input_network", os.path.abspath(args.onnx),
        "--output_path", os.path.abspath(fp32_dlc)
    ]

    if args.input_dim:
        for name, dims in args.input_dim:
            conv_cmd.extend(["--source_model_input_shape", name, dims])

    print("Command:", " ".join(conv_cmd))
    subprocess.run(conv_cmd, check=True)

    print("\n=======================================================")
    print(" Step 2: Quantizing FP32 DLC to INT8 DLC")
    print("=======================================================")
    quant_cmd = [
        "qairt-quantizer",
        "--input_dlc", os.path.abspath(fp32_dlc),
        "--input_list", os.path.abspath(input_list_path),
        "--output_dlc", os.path.abspath(args.output_dlc)
    ]
    print("Command:", " ".join(quant_cmd))
    subprocess.run(quant_cmd, check=True)

    print(f"\nSuccess! Generated INT8 DLC at: {os.path.abspath(args.output_dlc)}")

if __name__ == "__main__":
    main()
