"""
================================================================
  SKIN DISEASE DETECTION — Dataset Organizer & Preprocessor
================================================================
  Step 1: Run this AFTER manually sorting images into folders
  Step 2: It will clean, resize, augment & prepare your dataset
================================================================
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json

# ──────────────────────────────────────────────
#  CONFIG — Change these paths as needed
# ──────────────────────────────────────────────
RAW_DATA_DIR   = "raw_images"          # Where you manually sorted images
OUTPUT_DIR     = "dataset/skin_dataset"  # Final cleaned dataset goes here
IMAGE_SIZE     = (224, 224)            # Required by MobileNetV2
AUGMENT_TARGET = 500                   # Min images per class after augmentation
VALID_EXTS     = {".png", ".jpg", ".jpeg"}

CLASSES = ["acne", "eczema", "fungal", "normal", "psoriasis"]

# ──────────────────────────────────────────────
#  STEP 1: Manual Sorting Guide (printed for user)
# ──────────────────────────────────────────────
def print_sorting_guide():
    print("\n" + "="*60)
    print("  📁 MANUAL SORTING — DO THIS FIRST (one time only)")
    print("="*60)
    print("""
  Create this folder structure and place your images inside:

  raw_images/
  ├── acne/         ← put all acne images here
  ├── eczema/       ← put all eczema images here
  ├── psoriasis/    ← put all psoriasis images here
  └── fungal/       ← put all fungal images here

  Tips:
  • Just drag and drop — names don't matter
  • PNG files are fine
  • Aim for ~500 images per folder
  • Skip blurry or very dark images

  Once done, run this script again and press Enter to continue.
""")
    input("  ✅ Press Enter when you've sorted your images...")

# ──────────────────────────────────────────────
#  STEP 2: Validate Raw Images
# ──────────────────────────────────────────────
def validate_images(class_name, src_dir):
    valid, corrupt = [], []
    for fname in os.listdir(src_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VALID_EXTS:
            continue
        fpath = os.path.join(src_dir, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()  # Detect corrupt files
            valid.append(fpath)
        except Exception:
            corrupt.append(fname)

    print(f"  [{class_name}] ✅ Valid: {len(valid)}  ❌ Corrupt: {len(corrupt)}")
    if corrupt:
        print(f"    Skipped corrupt files: {corrupt[:5]}{'...' if len(corrupt)>5 else ''}")
    return valid

# ──────────────────────────────────────────────
#  STEP 3: Clean & Resize Single Image
# ──────────────────────────────────────────────
def clean_image(src_path, dst_path):
    with Image.open(src_path) as img:
        # Convert to RGB (removes alpha channel from PNG)
        img = img.convert("RGB")
        # Resize to 224x224
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        # Save as JPEG for consistency & smaller size
        img.save(dst_path, "JPEG", quality=90)

# ──────────────────────────────────────────────
#  STEP 4: Augmentation (generates new images)
# ──────────────────────────────────────────────
def augment_image(src_path, dst_dir, base_name, count):
    """Creates `count` augmented versions of one image."""
    with Image.open(src_path) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)

    saved = 0
    attempts = 0
    while saved < count and attempts < count * 3:
        attempts += 1
        aug = img.copy()

        # Random horizontal flip
        if random.random() > 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)

        # Random rotation (-25 to +25 degrees)
        angle = random.uniform(-25, 25)
        aug = aug.rotate(angle, fillcolor=(128, 128, 128))

        # Random brightness (0.7 to 1.3)
        brightness = random.uniform(0.7, 1.3)
        aug = ImageEnhance.Brightness(aug).enhance(brightness)

        # Random contrast (0.8 to 1.2)
        contrast = random.uniform(0.8, 1.2)
        aug = ImageEnhance.Contrast(aug).enhance(contrast)

        # Random slight blur (30% chance)
        if random.random() < 0.3:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=0.8))

        # Random zoom crop (80–100% of image)
        zoom = random.uniform(0.80, 1.0)
        w, h = aug.size
        new_w, new_h = int(w * zoom), int(h * zoom)
        left = random.randint(0, w - new_w)
        top  = random.randint(0, h - new_h)
        aug = aug.crop((left, top, left + new_w, top + new_h))
        aug = aug.resize(IMAGE_SIZE, Image.LANCZOS)

        out_path = os.path.join(dst_dir, f"{base_name}_aug{saved+1}.jpg")
        aug.save(out_path, "JPEG", quality=85)
        saved += 1

    return saved

# ──────────────────────────────────────────────
#  STEP 5: Split into train / val / test
# ──────────────────────────────────────────────
def split_dataset(all_images, train=0.70, val=0.15):
    random.shuffle(all_images)
    n = len(all_images)
    t = int(n * train)
    v = int(n * val)
    return all_images[:t], all_images[t:t+v], all_images[t+v:]

# ──────────────────────────────────────────────
#  MAIN PIPELINE
# ──────────────────────────────────────────────
def main():
    print("\n🩺 Skin Disease Dataset Processor")
    print("="*60)

    # Check if raw_images folder exists
    if not os.path.exists(RAW_DATA_DIR):
        print_sorting_guide()

    # Verify all class folders exist
    missing = [c for c in CLASSES if not os.path.isdir(os.path.join(RAW_DATA_DIR, c))]
    if missing:
        print(f"\n❌ Missing folders in raw_images/: {missing}")
        print("   Please create them and add your images.")
        return

    print("\n📊 Step 1: Validating your images...\n")
    class_images = {}
    for cls in CLASSES:
        src = os.path.join(RAW_DATA_DIR, cls)
        valid = validate_images(cls, src)
        class_images[cls] = valid

    total_raw = sum(len(v) for v in class_images.values())
    print(f"\n  Total valid images found: {total_raw}")

    if total_raw == 0:
        print("❌ No valid images found. Please add images to raw_images/ folders.")
        return

    # ── Clean & Organize ──
    print("\n🔧 Step 2: Cleaning & resizing images...\n")

    temp_dir = "dataset/_temp_processed"
    os.makedirs(temp_dir, exist_ok=True)

    for cls in CLASSES:
        cls_temp = os.path.join(temp_dir, cls)
        os.makedirs(cls_temp, exist_ok=True)
        cleaned = []
        for i, src_path in enumerate(class_images[cls]):
            dst_path = os.path.join(cls_temp, f"{cls}_{i:04d}.jpg")
            try:
                clean_image(src_path, dst_path)
                cleaned.append(dst_path)
            except Exception as e:
                print(f"    ⚠️  Skipped {os.path.basename(src_path)}: {e}")
        class_images[cls] = cleaned
        print(f"  [{cls}] Cleaned {len(cleaned)} images → 224×224 RGB JPEG")

    # ── Augmentation ──
    print("\n🔄 Step 3: Augmenting underrepresented classes...\n")

    for cls in CLASSES:
        current = len(class_images[cls])
        if current < AUGMENT_TARGET:
            needed = AUGMENT_TARGET - current
            cls_temp = os.path.join(temp_dir, cls)
            sources = class_images[cls].copy()
            aug_count = 0
            idx = 0
            while aug_count < needed:
                src = sources[idx % len(sources)]
                base = f"{cls}_aug_src{idx}"
                per_img = min(5, needed - aug_count)
                aug_count += augment_image(src, cls_temp, base, per_img)
                idx += 1

            # Refresh file list
            all_files = [os.path.join(cls_temp, f) for f in os.listdir(cls_temp)
                         if f.endswith(".jpg")]
            class_images[cls] = all_files
            print(f"  [{cls}] {current} → {len(all_files)} images (added {aug_count} augmented)")
        else:
            print(f"  [{cls}] {current} images — no augmentation needed ✅")

    # ── Train / Val / Test Split ──
    print("\n📂 Step 4: Splitting into train / val / test (70/15/15)...\n")

    stats = {}
    for cls in CLASSES:
        train_imgs, val_imgs, test_imgs = split_dataset(class_images[cls])

        for split, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            dst_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

        stats[cls] = {
            "train": len(train_imgs),
            "val":   len(val_imgs),
            "test":  len(test_imgs),
            "total": len(class_images[cls])
        }
        print(f"  [{cls}]  Train: {len(train_imgs)}  Val: {len(val_imgs)}  Test: {len(test_imgs)}")

    # ── Save class_names.json ──
    class_names_path = os.path.join("utils", "class_names.json")
    os.makedirs("utils", exist_ok=True)
    with open(class_names_path, "w") as f:
        json.dump(CLASSES, f, indent=2)
    print(f"\n  ✅ Saved class names → {class_names_path}")

    # ── Save dataset stats ──
    stats_path = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Cleanup temp ──
    shutil.rmtree(temp_dir)
    print(f"  🗑️  Cleaned up temp files")

    # ── Final Summary ──
    print("\n" + "="*60)
    print("  ✅ DATASET PROCESSING COMPLETE!")
    print("="*60)
    total_final = sum(s["total"] for s in stats.values())
    print(f"""
  📊 Final Dataset Summary:
  ┌─────────────┬───────┬─────┬──────┐
  │ Class       │ Train │ Val │ Test │
  ├─────────────┼───────┼─────┼──────┤""")
    for cls, s in stats.items():
        print(f"  │ {cls:<11} │ {s['train']:>5} │ {s['val']:>3} │ {s['test']:>4} │")
    print(f"""  └─────────────┴───────┴─────┴──────┘
  Total images: {total_final}
  Output dir:   {OUTPUT_DIR}/

  🚀 Next Step — Train your model:
     python train_model.py --dataset {OUTPUT_DIR} --model transfer --epochs 20
""")

# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
