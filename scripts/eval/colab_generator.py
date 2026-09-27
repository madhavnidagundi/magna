import os
import csv
import json
import random
import shutil
import argparse
from collections import defaultdict

# --- Standalone Dependencies ---
try:
    import torch
    import numpy as np
    from torchvision import datasets, transforms
    import kagglehub
except ImportError:
    print("Please install required dependencies: pip install kagglehub torch torchvision numpy")
    sys.exit(1)


def stratified_sample(dataset, num_images, seed=42, excluded_paths=None):
    """
    Samples num_images evenly across all classes in the dataset.
    Returns:
        selected_indices: List of dataset indices chosen.
        manifest_data: List of dicts with keys ['index', 'relative_path', 'class_index', 'class_name']
    """
    if excluded_paths is None:
        excluded_paths = set()

    rng = random.Random(seed)

    # 1. Group indices by class, ignoring excluded paths
    class_indices = defaultdict(list)
    index_to_rel_path = {}

    for idx, (path, label) in enumerate(dataset.samples):
        rel_path = os.path.relpath(path, dataset.root)
        rel_path = rel_path.replace('\\', '/')

        if rel_path in excluded_paths:
            continue

        class_indices[label].append(idx)
        index_to_rel_path[idx] = rel_path

    available_images = sum(len(indices) for indices in class_indices.values())
    if available_images < num_images:
        raise ValueError(f"Requested {num_images} images but only {available_images} are available after exclusion.")

    # 2. Determine counts per class using shuffled class order for remainders
    num_classes = len(dataset.classes)
    class_labels = sorted(class_indices.keys())
    rng.shuffle(class_labels)

    base = num_images // num_classes
    remainder = num_images % num_classes

    selected_indices = []

    for position, label in enumerate(class_labels):
        count = base + (1 if position < remainder else 0)
        available = class_indices[label]
        if count > len(available):
            selected = available
        else:
            selected = rng.sample(available, count)
        selected_indices.extend(selected)

    # 3. Backfill if some classes didn't have enough images
    remaining_needed = num_images - len(selected_indices)
    if remaining_needed > 0:
        remaining_pool = list(set(index_to_rel_path.keys()) - set(selected_indices))
        selected_indices.extend(rng.sample(remaining_pool, remaining_needed))

    rng.shuffle(selected_indices)

    # 4. Generate manifest data
    manifest_data = []
    for idx in selected_indices:
        _, label = dataset.samples[idx]
        manifest_data.append({
            'index': idx,
            'relative_path': index_to_rel_path[idx],
            'class_index': label,
            'class_name': dataset.classes[label]
        })

    return selected_indices, manifest_data

def main():
    parser = argparse.ArgumentParser(description="Colab Standalone Eval Generator")
    parser.add_argument("--num_images", type=int, default=50000)
    parser.add_argument("--calibration_manifest", type=str, default="calibration_manifest.csv")
    parser.add_argument("--output_zip", type=str, default="eval_data_50k.zip")

    # Use parse_known_args so Jupyter kernel arguments (like -f) don't crash it
    args, unknown = parser.parse_known_args()

    print("1. Downloading ImageNet validation dataset via Kagglehub...")
    dataset_base = kagglehub.dataset_download("titericz/imagenet1k-val")

    # The actual images are typically in an 'imagenet-val' subdirectory
    data_dir = os.path.join(dataset_base, "imagenet-val")
    if not os.path.exists(data_dir):
        # Fallback if the folder structure is different
        data_dir = dataset_base

    print(f"Dataset located at: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("2. Loading dataset into PyTorch...")
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Prep output staging directory
    staging_dir = "eval_data"
    raw_dir = os.path.join(staging_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Class mapping
    with open(os.path.join(staging_dir, "class_to_idx.json"), "w") as f:
        json.dump(dataset.class_to_idx, f, indent=2)

    # Exclusions
    excluded_paths = set()
    if os.path.exists(args.calibration_manifest):
        with open(args.calibration_manifest, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                excluded_paths.add(row['relative_path'])
        print(f"Loaded {len(excluded_paths)} excluded paths from calibration manifest.")
    else:
        print("No calibration manifest found. Proceeding without exclusions.")

    limit = min(args.num_images, len(dataset) - len(excluded_paths))

    print(f"3. Performing stratified sampling for {limit} evaluation images...")
    selected_indices, manifest_data = stratified_sample(dataset, limit, seed=42, excluded_paths=excluded_paths)

    manifest_path = os.path.join(staging_dir, "evaluation_manifest.csv")
    with open(manifest_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'relative_path', 'class_index', 'class_name'])
        writer.writeheader()
        writer.writerows(manifest_data)

    subset = torch.utils.data.Subset(dataset, selected_indices)
    labels = []

    print(f"4. Generating {limit} NCHW float32 raw tensors...")
    for i in range(limit):
        img_tensor, gt_label = subset[i]

        # PyTorch is CHW. Add N dimension implicitly by treating it as 1CHW later.
        img_np = img_tensor.numpy().astype(np.float32)

        filename = f"eval_{i:05d}.raw"
        img_np.tofile(os.path.join(raw_dir, filename))
        labels.append(gt_label)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1} / {limit}")

    # Write labels.txt
    with open(os.path.join(staging_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(f"{label}\n")

    print(f"5. Compressing to {args.output_zip}...")
    zip_basename = args.output_zip.replace('.zip', '')
    shutil.make_archive(zip_basename, 'zip', staging_dir)

    print(f"Success! Created {args.output_zip}")
    print("You can now download this ZIP file and transfer it to the Radxa board.")

if __name__ == "__main__":
    import sys
    main()
