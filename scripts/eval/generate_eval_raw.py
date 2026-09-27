import os
import sys
import csv
import json
import argparse
import numpy as np
import shutil
from torchvision import datasets, transforms
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.dataset_split import stratified_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Generate NCHW .raw files for QNN evaluation on the Radxa board")
    parser.add_argument("--data_dir", type=str, default="imagenette2-160/val", help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="eval_data/raw", help="Directory to save .raw files")
    parser.add_argument("--num_images", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--target_board_dir", type=str, default="/home/radxa/qc_magna_final/eval_data/raw", help="Absolute path on the board where raw files will reside")
    parser.add_argument("--calibration_manifest", type=str, default="", help="Path to calibration_manifest.csv to exclude those images")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading dataset from {args.data_dir}...")
    try:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    class_mapping_path = os.path.join(args.output_dir, "class_to_idx.json")
    with open(class_mapping_path, "w") as f:
        json.dump(dataset.class_to_idx, f, indent=2)
    print(f"Exported class_to_idx mapping to {class_mapping_path}")

    excluded_paths = set()
    if args.calibration_manifest and os.path.exists(args.calibration_manifest):
        with open(args.calibration_manifest, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                excluded_paths.add(row['relative_path'])
        print(f"Loaded {len(excluded_paths)} excluded paths from calibration manifest.")

    limit = min(args.num_images, len(dataset) - len(excluded_paths))
    if limit <= 0:
        print("No images available for evaluation after exclusion!")
        return

    required_bytes = limit * 3 * 224 * 224 * 4
    safety_margin = 2 * 1024 * 1024 * 1024 # 2 GiB
    total_required = required_bytes + safety_margin
    free_bytes = shutil.disk_usage(args.output_dir).free

    if free_bytes < total_required:
        print(f"Error: Insufficient free space. Need {total_required / (1024**3):.2f} GB (including 2GB margin), but only {free_bytes / (1024**3):.2f} GB available.")
        sys.exit(1)

    print(f"Performing stratified sampling for {limit} evaluation images...")
    selected_indices, manifest_data = stratified_sample(dataset, limit, seed=42, excluded_paths=excluded_paths)

    manifest_path = os.path.join(args.output_dir, "evaluation_manifest.csv")
    with open(manifest_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'relative_path', 'class_index', 'class_name'])
        writer.writeheader()
        writer.writerows(manifest_data)

    subset = torch.utils.data.Subset(dataset, selected_indices)

    board_paths = []
    labels = []
    print(f"Generating {limit} .raw files in NCHW format...")

    for i in range(limit):
        img_tensor, gt_label = subset[i]

        # PyTorch output is (C, H, W). The ONNX model is NCHW.
        img_np = img_tensor.numpy().astype(np.float32)

        filename = f"eval_{i:04d}.raw"
        raw_path = os.path.abspath(os.path.join(args.output_dir, filename))
        img_np.tofile(raw_path)

        # Generate board-specific absolute path
        board_paths.append(f"{args.target_board_dir.rstrip('/')}/{filename}")
        labels.append(gt_label)

    input_list_path = os.path.join(args.output_dir, "input_list_board.txt")
    with open(input_list_path, "w") as f:
        for b_path in board_paths:
            f.write(f"{b_path}\n")

    labels_path = os.path.join(args.output_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"Successfully generated {limit} .raw files.")
    print(f"Input list saved to {input_list_path}")
    print(f"Ground truth labels saved to {labels_path}")
    print("Please copy the generated files to your Radxa board and use them with qnn-net-run.")

if __name__ == "__main__":
    main()
