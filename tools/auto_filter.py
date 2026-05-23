"""
================================================================
  SKIN DISEASE — Auto Image Quality Filter
================================================================
  Automatically filters images based on:
  ✅ Sharpness (removes blurry images)
  ✅ Brightness (removes too dark / too bright)
  ✅ Resolution (removes too small images)
  ✅ Duplicates (removes similar/duplicate images)
  ✅ Keeps TOP 250 best quality images per class
  ✅ Rejected images moved to raw_images/_rejected/ (safe)
================================================================
  HOW TO RUN:
    python auto_filter.py
================================================================
"""

import os
import shutil
import hashlib
import numpy as np
from PIL import Image, ImageFilter

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
RAW_IMAGES_DIR = "raw_images"
REJECTED_DIR   = "raw_images/_rejected"
TARGET_KEEP    = 250
CLASSES        = ["acne", "eczema", "fungal", "normal", "psoriasis"]
VALID_EXTS     = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Quality thresholds
MIN_SIZE        = (100, 100)   # Minimum image resolution
BLUR_THRESHOLD  = 80.0         # Below this = blurry (higher = stricter)
MIN_BRIGHTNESS  = 30           # Below this = too dark (0-255)
MAX_BRIGHTNESS  = 230          # Above this = too bright (0-255)

# ──────────────────────────────────────────────
#  QUALITY SCORING FUNCTIONS
# ──────────────────────────────────────────────

def get_blur_score(img):
    """Higher score = sharper image. Uses Laplacian variance."""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    # Laplacian kernel
    laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    from PIL.ImageFilter import Kernel
    filtered = gray.filter(ImageFilter.FIND_EDGES)
    return np.array(filtered, dtype=np.float32).var()

def get_brightness(img):
    """Returns average brightness 0-255."""
    gray = np.array(img.convert("L"), dtype=np.float32)
    return gray.mean()

def get_file_hash(path):
    """MD5 hash to detect duplicates."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def score_image(path):
    """
    Returns (is_valid, score, reason)
    score: higher = better quality
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size

            # ── Check 1: Resolution ──
            if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                return False, 0, f"Too small ({w}x{h})"

            # ── Check 2: Brightness ──
            brightness = get_brightness(img)
            if brightness < MIN_BRIGHTNESS:
                return False, 0, f"Too dark (brightness={brightness:.1f})"
            if brightness > MAX_BRIGHTNESS:
                return False, 0, f"Too bright (brightness={brightness:.1f})"

            # ── Check 3: Blur ──
            blur = get_blur_score(img)
            if blur < BLUR_THRESHOLD:
                return False, 0, f"Too blurry (score={blur:.1f})"

            # ── Overall quality score ──
            # Normalize brightness penalty (ideal = 128)
            brightness_score = 100 - abs(brightness - 128) / 1.28
            # Resolution score
            res_score = min(100, (w * h) / (500 * 500) * 100)
            # Combined score
            final_score = (blur * 0.5) + (brightness_score * 0.3) + (res_score * 0.2)

            return True, final_score, "OK"

    except Exception as e:
        return False, 0, f"Cannot open: {e}"


# ──────────────────────────────────────────────
#  MAIN FILTER PIPELINE
# ──────────────────────────────────────────────
def process_class(cls):
    cls_dir = os.path.join(RAW_IMAGES_DIR, cls)
    rejected_dir = os.path.join(REJECTED_DIR, cls)
    os.makedirs(rejected_dir, exist_ok=True)

    # Get all image files
    files = [
        os.path.join(cls_dir, f)
        for f in os.listdir(cls_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
        and not f.startswith(".")
    ]

    if not files:
        print(f"  [{cls}] ⚠️  No images found!")
        return 0, 0, 0

    print(f"  [{cls}] Found {len(files)} images — analyzing...")

    # ── Step 1: Remove duplicates ──
    seen_hashes = set()
    unique_files = []
    dupes = 0
    for f in files:
        h = get_file_hash(f)
        if h in seen_hashes:
            shutil.move(f, os.path.join(rejected_dir, os.path.basename(f)))
            dupes += 1
        else:
            seen_hashes.add(h)
            unique_files.append(f)

    # ── Step 2: Score all unique images ──
    scored = []
    rejected_quality = 0
    for f in unique_files:
        valid, score, reason = score_image(f)
        if valid:
            scored.append((score, f))
        else:
            shutil.move(f, os.path.join(rejected_dir, os.path.basename(f)))
            rejected_quality += 1

    # ── Step 3: Sort by quality, keep top 250 ──
    scored.sort(key=lambda x: x[0], reverse=True)  # Best first

    kept = scored[:TARGET_KEEP]
    extra = scored[TARGET_KEEP:]

    # Move extra (lower quality) to rejected
    for _, f in extra:
        shutil.move(f, os.path.join(rejected_dir, os.path.basename(f)))

    total_rejected = dupes + rejected_quality + len(extra)
    print(f"  [{cls}] ✅ Kept: {len(kept)}  ❌ Removed: {total_rejected}  "
          f"(dupes={dupes}, low-quality={rejected_quality}, extras={len(extra)})")

    return len(kept), total_rejected, dupes


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  🩺 Auto Image Quality Filter")
    print("="*60)
    print(f"\n  Settings:")
    print(f"  • Target per class : {TARGET_KEEP} images")
    print(f"  • Min resolution   : {MIN_SIZE[0]}x{MIN_SIZE[1]} px")
    print(f"  • Blur threshold   : {BLUR_THRESHOLD}")
    print(f"  • Brightness range : {MIN_BRIGHTNESS} – {MAX_BRIGHTNESS}")
    print(f"\n  Rejected images → {REJECTED_DIR}/ (NOT deleted, safe to recover)\n")

    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"❌ Folder '{RAW_IMAGES_DIR}' not found!")
        print("   Make sure you run this from:")
        print("   C:\\Users\\karav\\OneDrive\\Desktop\\skin_disease_detection")
        input("\nPress Enter to exit...")
        return

    print("-"*60)
    print("  Processing classes...\n")

    total_kept = 0
    total_removed = 0

    for cls in CLASSES:
        cls_dir = os.path.join(RAW_IMAGES_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [{cls}] ⚠️  Folder not found, skipping...")
            continue
        kept, removed, dupes = process_class(cls)
        total_kept    += kept
        total_removed += removed

    # ── Summary ──
    print("\n" + "="*60)
    print("  ✅ AUTO FILTER COMPLETE!")
    print("="*60)
    print(f"""
  📊 Final Summary:
  • Total images kept    : {total_kept}
  • Total images removed : {total_removed}
  • Rejected saved in    : {REJECTED_DIR}/

  🚀 Next step — Run:
     python organize_dataset.py
""")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
