import os
import glob
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate QNN NPU Results")
    parser.add_argument("--results_dir", type=str, default="output_eval", help="Directory containing Result_X folders")
    parser.add_argument("--labels_file", type=str, default="eval_data/labels.txt", help="Path to ground truth labels.txt")
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory {args.results_dir} not found. Please SCP it from the board.")
        return

    if not os.path.exists(args.labels_file):
        print(f"Error: Labels file {args.labels_file} not found.")
        return

    # Load ground truth labels
    with open(args.labels_file, "r") as f:
        gt_labels = [int(line.strip()) for line in f.readlines() if line.strip()]

    result_folders = glob.glob(os.path.join(args.results_dir, "Result_*"))
    # Sort numerically by Result_X
    result_folders.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))

    if len(result_folders) == 0:
        print("No Result_X folders found!")
        return

    print(f"Found {len(result_folders)} result folders. Evaluating...")

    top1_correct = 0
    top5_correct = 0
    total = 0

    for i, folder in enumerate(result_folders):
        raw_files = glob.glob(os.path.join(folder, "*.raw"))
        if not raw_files:
            continue

        raw_file = raw_files[0]
        file_size = os.path.getsize(raw_file)

        # Float32 output
        if file_size == 4000:
            data = np.fromfile(raw_file, dtype=np.float32)
        # INT8 or UINT8 output
        elif file_size == 1000:
            # Try uint8 first since dequantization offsets often shift into positive bounds
            data = np.fromfile(raw_file, dtype=np.uint8)
        else:
            data = np.fromfile(raw_file, dtype=np.int8)

        gt_class = gt_labels[i]

        # Top 5 predictions
        top5_idx = np.argsort(data)[-5:][::-1]
        top1_idx = top5_idx[0]

        if top1_idx == gt_class:
            top1_correct += 1
        if gt_class in top5_idx:
            top5_correct += 1

        total += 1

    if total > 0:
        print("="*50)
        print("   MobileNetV4 QNN NPU Hardware Evaluation")
        print("="*50)
        print(f"Total Images Evaluated: {total}")
        print(f"Top-1 Accuracy: {100.0 * top1_correct / total:.2f}%")
        print(f"Top-5 Accuracy: {100.0 * top5_correct / total:.2f}%")
        print("="*50)
    else:
        print("Failed to read any results.")

if __name__ == "__main__":
    main()
