"""Create non-overlapping, class-balanced split folders from the supplied data."""

import json
import os
import random
import re
import shutil

import config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_NAMES = ("train", "validation", "test")


def source_group(filename):
    """Group Roboflow export variants by their shared original image ID."""
    image_id = filename.split(".rf.", 1)[0]
    image_id = re.sub(r"_(jpg|jpeg|png|webp|bmp)$", "", image_id, flags=re.IGNORECASE)
    return image_id.lower()


def collect_source_groups():
    groups_by_class = {name: {} for name in config.CLASS_NAMES}
    for source_split in (config.TRAIN_DIR, config.TEST_DIR):
        for class_name in config.CLASS_NAMES:
            class_dir = os.path.join(source_split, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                path = os.path.join(class_dir, filename)
                if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                    groups_by_class[class_name].setdefault(source_group(filename), []).append(path)

    missing = [name for name, groups in groups_by_class.items() if not groups]
    if missing:
        raise FileNotFoundError(
            "No images found for these classes in train/ or test/: " + ", ".join(missing)
        )
    return groups_by_class


def split_groups(groups_by_class):
    rng = random.Random(config.DATA_SPLIT_SEED)
    assignments = {split: {name: [] for name in config.CLASS_NAMES} for split in SPLIT_NAMES}
    for class_name, groups in groups_by_class.items():
        group_names = sorted(groups)
        rng.shuffle(group_names)
        if len(group_names) < 3:
            raise ValueError(f"Class '{class_name}' needs at least 3 unique source images.")
        train_end = max(1, int(len(group_names) * (1 - config.VALIDATION_SPLIT - config.TEST_SPLIT)))
        validation_end = max(train_end + 1, int(len(group_names) * (1 - config.TEST_SPLIT)))
        validation_end = min(validation_end, len(group_names) - 1)
        partitions = {
            "train": group_names[:train_end],
            "validation": group_names[train_end:validation_end],
            "test": group_names[validation_end:],
        }
        for split_name, names in partitions.items():
            assignments[split_name][class_name] = names
    return assignments


def prepare_dataset():
    groups_by_class = collect_source_groups()
    assignments = split_groups(groups_by_class)
    if os.path.isdir(config.PREPARED_DATASET_DIR) and os.listdir(config.PREPARED_DATASET_DIR):
        raise FileExistsError(
            f"Prepared dataset already contains files: {config.PREPARED_DATASET_DIR}. "
            "Move it aside before generating a new split."
        )

    manifest = {
        "seed": config.DATA_SPLIT_SEED,
        "source_groups": {},
        "counts": {},
    }
    try:
        for split_name in SPLIT_NAMES:
            manifest["counts"][split_name] = {}
            for class_name in config.CLASS_NAMES:
                output_dir = os.path.join(config.PREPARED_DATASET_DIR, split_name, class_name)
                os.makedirs(output_dir, exist_ok=True)
                group_names = assignments[split_name][class_name]
                manifest["counts"][split_name][class_name] = len(group_names)
                for group_name in group_names:
                    source_path = sorted(groups_by_class[class_name][group_name])[0]
                    output_name = f"{group_name}{os.path.splitext(source_path)[1].lower()}"
                    shutil.copy2(source_path, os.path.join(output_dir, output_name))
                    manifest["source_groups"][group_name] = split_name
        with open(os.path.join(config.PREPARED_DATASET_DIR, "manifest.json"), "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
    except Exception:
        shutil.rmtree(config.PREPARED_DATASET_DIR, ignore_errors=True)
        raise
    return manifest


if __name__ == "__main__":
    result = prepare_dataset()
    print(f"Prepared dataset at {config.PREPARED_DATASET_DIR}")
    for split_name, counts in result["counts"].items():
        print(f"{split_name}: {counts}")