import os
import random
from collections import defaultdict

def stratified_sample(dataset, num_images, seed=42, excluded_paths=None):
    """
    Samples num_images evenly across all classes in the dataset.
    Returns:
        selected_indices: List of dataset indices chosen.
        manifest_data: List of dicts with keys ['index', 'relative_path', 'class_index']
    """
    if excluded_paths is None:
        excluded_paths = set()

    rng = random.Random(seed)

    # 1. Group indices by class, ignoring excluded paths
    class_indices = defaultdict(list)
    index_to_rel_path = {}

    for idx, (path, label) in enumerate(dataset.samples):
        rel_path = os.path.relpath(path, dataset.root)

        # Windows/Linux normalization
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
            # This happens if a class has fewer images than 'count'
            # We take all available, and we'll have to backfill later.
            selected = available
        else:
            selected = rng.sample(available, count)

        selected_indices.extend(selected)

    # 3. Backfill if some classes didn't have enough images
    remaining_needed = num_images - len(selected_indices)
    if remaining_needed > 0:
        remaining_pool = list(set(index_to_rel_path.keys()) - set(selected_indices))
        selected_indices.extend(rng.sample(remaining_pool, remaining_needed))

    # Shuffle the final selection so output isn't sorted by class
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
