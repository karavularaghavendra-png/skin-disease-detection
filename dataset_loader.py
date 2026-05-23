"""
Memory-efficient dataset_loader.py
Loads file paths only — images loaded in batches during training
"""

import os
import json
import numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_paths(folder, class_names):
    paths, labels = [], []
    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            print(f"  Warning: Skipping missing folder: {cls_dir}")
            continue
        files = [f for f in os.listdir(cls_dir)
                 if os.path.splitext(f)[1].lower() in VALID_EXTS]
        print(f"  [{cls}] {len(files)} images")
        for fname in files:
            paths.append(os.path.join(cls_dir, fname))
            labels.append(label)
    return paths, labels


def validate_dataset(dataset_path):
    subdirs = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]

    if "train" in subdirs:
        train_dir = os.path.join(dataset_path, "train")
        class_names = sorted([
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ])
        split_mode = "presplit"
    else:
        class_names = sorted(subdirs)
        split_mode = "flat"

    if not class_names:
        raise ValueError(
            f"No images found in any class folder under '{dataset_path}'.\n"
            "Ensure images are in JPG, JPEG, PNG, or BMP format."
        )

    print(f"\n  Found {len(class_names)} classes: {class_names}")
    print(f"  Dataset structure: {split_mode}\n")

    total = 0
    print("  DATASET VALIDATION")
    print("  " + "="*40)
    for split in (["train", "val", "test"] if split_mode == "presplit" else [""]):
        folder = os.path.join(dataset_path, split) if split else dataset_path
        count = 0
        for cls in class_names:
            cls_dir = os.path.join(folder, cls)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir)
                         if os.path.splitext(f)[1].lower() in VALID_EXTS])
                count += n
        label = split if split else "total"
        print(f"  {label:<8} -> {count:>5} images")
        total += count

    print(f"\n  Total: {total} images across {len(class_names)} classes")
    print("  " + "="*40 + "\n")

    return class_names, split_mode


def load_and_split(dataset_path):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")

    class_names, split_mode = validate_dataset(dataset_path)

    if split_mode == "presplit":
        print("  Loading train image paths...")
        X_train, y_train = get_image_paths(
            os.path.join(dataset_path, "train"), class_names)

        val_dir  = os.path.join(dataset_path, "val")
        test_dir = os.path.join(dataset_path, "test")

        print("  Loading val image paths...")
        X_val, y_val = get_image_paths(val_dir, class_names) \
            if os.path.exists(val_dir) else (X_train[:100], y_train[:100])

        print("  Loading test image paths...")
        X_test, y_test = get_image_paths(test_dir, class_names) \
            if os.path.exists(test_dir) else (X_val, y_val)

    else:
        print("  Loading all image paths...")
        X_all, y_all = get_image_paths(dataset_path, class_names)
        idx = np.random.permutation(len(X_all))
        X_all = [X_all[i] for i in idx]
        y_all = [y_all[i] for i in idx]
        n = len(X_all)
        t, v = int(n * 0.70), int(n * 0.15)
        X_train, y_train = X_all[:t],    y_all[:t]
        X_val,   y_val   = X_all[t:t+v], y_all[t:t+v]
        X_test,  y_test  = X_all[t+v:],  y_all[t+v:]

    os.makedirs("utils", exist_ok=True)
    with open("utils/class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"  Class names saved to utils/class_names.json")
    print(f"\n  Loaded:")
    print(f"     Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_names
