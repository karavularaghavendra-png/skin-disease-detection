"""
Prediction Module for Skin Disease Detection — with Test-Time Augmentation (TTA).

Provides:
    load_model_cached      : Load and cache the trained Keras model + class names
    predict_single_image   : Single-pass prediction (fast, file-path based)
    predict_with_tta       : 8-pass TTA prediction (robust, file-path based)
    analyze_prediction_reliability : Confidence/entropy/margin checks

TTA runs 8 deterministic augmented versions of the image through the model
and averages the softmax outputs. If the model is genuinely confident,
all 8 passes converge. If it's guessing, passes diverge — flagged as unreliable.

TTA Passes:
    0: Original (clean input)
    1: Horizontal flip (left/right arm symmetry)
    2: +20% brightness (outdoor / direct flash)
    3: -20% brightness (dim indoor / backlit)
    4: +15% contrast (DSLR / good camera)
    5: Gaussian blur (motion blur / soft focus)
    6: Center crop 90% (slight zoom-in)
    7: Warm color tint (incandescent indoor lighting)
"""

import os
import io
import json
import logging
import functools
import time as _time

# ── Centralised TF warning suppression (must be before TF import) ─────────
import logger as _logger_init  # noqa: F401 — sets TF env vars on import

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image as PILImage, ImageEnhance, ImageFilter

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger(__name__)

# ── GPU memory growth (prevents OOM on shared GPUs) ──────────
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 65.0     # Minimum confidence % to trust
ENTROPY_THRESHOLD = 1.2         # Max entropy — above = unreliable
MARGIN_THRESHOLD = 15.0         # Min gap % between top-1 and top-2
TTA_DISAGREEMENT_THR = 0.20     # Max std-dev across TTA passes
TTA_NUM_PASSES = 8              # Number of TTA augmentation passes

# ─────────────────────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────────────────────
_MODEL_CANDIDATES = [
    os.path.join("model", "skin_model.h5"),
    os.path.join("model", "best_model.h5"),
    os.path.join("models", "skin_disease_model.h5"),
]

_CLASS_MAP_CANDIDATES = [
    os.path.join("utils", "class_names.json"),
    os.path.join("model", "class_names.json"),
    os.path.join("models", "class_names.json"),
]

IMG_SIZE = (224, 224)

# PIL image bomb protection
PILImage.MAX_IMAGE_PIXELS = 50_000_000


def _find_first(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ─────────────────────────────────────────────────────────────
# Safe Model Loader (handles old Keras batch_shape issue)
# ─────────────────────────────────────────────────────────────
def _load_model_safe(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as e:
        if "batch_shape" not in str(e):
            raise

    import h5py
    print(f"Patching old Keras InputLayer format in {model_path}...")
    with h5py.File(model_path, "r+") as f:
        if "model_config" in f.attrs:
            cfg = f.attrs["model_config"]
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8")
            cfg = cfg.replace('"batch_shape"', '"batch_input_shape"')
            f.attrs["model_config"] = cfg

    return tf.keras.models.load_model(model_path, compile=False)


# ─────────────────────────────────────────────────────────────
# Cached Model Loading
# ─────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def load_model_cached():
    """
    Load and cache the model + class names.
    Returns:
        (model, class_names) — a tuple
    """
    model_path = _find_first(_MODEL_CANDIDATES)
    class_path = _find_first(_CLASS_MAP_CANDIDATES)

    if model_path is None:
        raise FileNotFoundError(
            "Trained model not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in _MODEL_CANDIDATES)
            + "\n\nPlease run: python train_model.py --dataset dataset/skin_dataset"
        )

    if class_path is None:
        raise FileNotFoundError(
            "Class names file not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in _CLASS_MAP_CANDIDATES)
            + "\n\nPlease run: python train_model.py --dataset dataset/skin_dataset"
        )

    model = _load_model_safe(model_path)

    with open(class_path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        class_names = raw
    elif isinstance(raw, dict):
        class_names = [k for k, _ in sorted(raw.items(), key=lambda x: x[1])]
    else:
        raise ValueError(f"Unexpected class_names format: {type(raw)}")

    print(f"Model loaded: {model_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    return model, class_names


def get_model():
    model, _ = load_model_cached()
    return model


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Read image from file path, resize, normalize, return (1,224,224,3) tensor.

    Uses the same normalization as training (rescale=1./255 to [0,1]).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def _preprocess_pil(image: PILImage.Image) -> np.ndarray:
    """Convert a PIL image to a model-ready (1, 224, 224, 3) float32 array.
    Must match the normalization used during training (rescale=1./255).
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, PILImage.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# Keep old name as alias for backward compatibility
_preprocess_image = preprocess_image


# ─────────────────────────────────────────────────────────────
# TTA Augmentation Helpers
# ─────────────────────────────────────────────────────────────
def _apply_tta_augmentation(image: PILImage.Image, aug_id: int) -> np.ndarray:
    """
    Apply one of 8 deterministic augmentations and return a preprocessed array.
    Deterministic = reproducible & covers known range of real-world variation.

    Pass 0: Original (no change)
    Pass 1: Horizontal flip (skin disease is spatially symmetric)
    Pass 2: +20% brightness (outdoor / direct flash)
    Pass 3: -20% brightness (dim indoor / backlit selfie)
    Pass 4: +15% contrast (DSLR / good camera)
    Pass 5: Gaussian blur radius=1 (soft focus / distance)
    Pass 6: Center crop 90% then resize (zoom-in)
    Pass 7: Warm color tint (indoor incandescent lighting)
    """
    img = image.convert("RGB")

    if aug_id == 0:
        pass  # Original
    elif aug_id == 1:
        img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
    elif aug_id == 2:
        img = ImageEnhance.Brightness(img).enhance(1.20)
    elif aug_id == 3:
        img = ImageEnhance.Brightness(img).enhance(0.80)
    elif aug_id == 4:
        img = ImageEnhance.Contrast(img).enhance(1.15)
    elif aug_id == 5:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    elif aug_id == 6:
        w, h = img.size
        left, top = int(w * 0.05), int(h * 0.05)
        right, bot = int(w * 0.95), int(h * 0.95)
        img = img.crop((left, top, right, bot))
    elif aug_id == 7:
        arr = np.array(img, dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.08, 0, 255)  # red up
        arr[..., 2] = np.clip(arr[..., 2] * 0.92, 0, 255)  # blue down
        img = PILImage.fromarray(arr.astype(np.uint8))

    return _preprocess_pil(img)


def load_image_from_path(image_path: str) -> PILImage.Image:
    """Load a PIL image from a file path with validation."""
    img = PILImage.open(image_path)
    img = img.convert("RGB")
    return img


def load_image_from_bytes(data: bytes) -> PILImage.Image:
    """Load and validate a PIL image from raw bytes (uploaded file)."""
    try:
        image = PILImage.open(io.BytesIO(data))
        image.verify()
        image = PILImage.open(io.BytesIO(data))  # re-open after verify
        image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {e}") from e


# ─────────────────────────────────────────────────────────────
# Prediction Reliability Analysis
# ─────────────────────────────────────────────────────────────
def _compute_prediction_entropy(preds: np.ndarray) -> float:
    """Compute Shannon entropy of the prediction distribution."""
    p = np.clip(preds, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def analyze_prediction_reliability(
    preds: np.ndarray,
    confidence_pct: float,
    all_pass_probs: np.ndarray = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> dict:
    """Analyze whether a prediction should be trusted.

    Args:
        preds: Averaged probability array (num_classes,)
        confidence_pct: Top-1 confidence as percentage (0-100)
        all_pass_probs: Optional (num_passes, num_classes) for TTA disagreement
        confidence_threshold: Min confidence % to trust
        entropy_threshold: Max entropy
        margin_threshold: Min margin %

    Returns:
        Dict with is_reliable, entropy, margin, tta_disagreement, warnings
    """
    warnings = []
    is_reliable = True

    # 1. Confidence threshold
    if confidence_pct < confidence_threshold:
        is_reliable = False
        warnings.append(
            f"Low confidence ({confidence_pct:.1f}%). "
            "The model is not confident about this prediction. "
            "Please try a clearer, well-lit close-up photo of the skin area."
        )

    # 2. Entropy check
    entropy = _compute_prediction_entropy(preds)
    if entropy > entropy_threshold:
        is_reliable = False
        warnings.append(
            "The model shows high uncertainty across multiple conditions. "
            "This may not be a typical skin disease image."
        )

    # 3. Margin check (top-1 vs top-2)
    sorted_preds = np.sort(preds)[::-1]
    if len(sorted_preds) >= 2:
        margin = float((sorted_preds[0] - sorted_preds[1]) * 100.0)
    else:
        margin = 100.0

    if margin < margin_threshold and confidence_pct >= confidence_threshold:
        warnings.append(
            f"Close match between top predictions (margin: {margin:.1f}%). "
            "Consider consulting a dermatologist for accurate diagnosis."
        )

    # 4. TTA disagreement check (only when TTA was used)
    tta_disagreement = 0.0
    if all_pass_probs is not None and len(all_pass_probs) > 1:
        top_idx = np.argmax(preds)
        top_class_probs = all_pass_probs[:, top_idx]
        tta_disagreement = float(np.std(top_class_probs))

        if tta_disagreement > TTA_DISAGREEMENT_THR:
            is_reliable = False
            warnings.append(
                f"High TTA disagreement ({tta_disagreement:.3f}) — "
                "predictions vary significantly across image augmentations."
            )

    return {
        "is_reliable": is_reliable,
        "confidence_pct": confidence_pct,
        "entropy": round(entropy, 4),
        "margin": round(margin, 2),
        "tta_disagreement": round(tta_disagreement, 4),
        "warnings": warnings,
    }


# ─────────────────────────────────────────────────────────────
# Single-Pass Prediction (fast — for health checks / legacy)
# ─────────────────────────────────────────────────────────────
def predict_single_image(image_path: str, use_tta: bool = False):
    """
    Run prediction on an image file.

    Args:
        image_path: Full path to the image file on disk.
        use_tta: If True, uses 8-pass TTA for more robust prediction.

    Returns:
        (top_results, raw_preds)
        top_results is a list of {"disease": str, "confidence": float (0-100)}
        sorted by confidence descending.

        When use_tta=True, also returns TTA metadata in top_results[0].
    """
    if use_tta:
        return _predict_with_tta(image_path)

    model, class_names = load_model_cached()

    img_tensor = preprocess_image(image_path)
    preds = model.predict(img_tensor, verbose=0)[0]

    # Get Top-3
    top_indices = preds.argsort()[-3:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            "disease":    class_names[idx],
            "confidence": float(preds[idx] * 100.0),
        })

    return results, preds


# ─────────────────────────────────────────────────────────────
# TTA Prediction (robust — primary for production)
# ─────────────────────────────────────────────────────────────
def _predict_with_tta(image_path: str, n_passes: int = TTA_NUM_PASSES):
    """
    Run TTA prediction: 8 augmented passes averaged for robust output.

    Each pass hits slightly different neurons. If the model is genuinely
    confident, all passes converge. If it's guessing, passes diverge.

    Returns:
        (top_results, avg_preds) — same interface as predict_single_image
        top_results[0] includes TTA metadata fields.
    """
    t_start = _time.perf_counter()

    model, class_names = load_model_cached()

    # Load as PIL for TTA augmentations
    pil_image = load_image_from_path(image_path)

    n_passes = min(n_passes, 8)
    all_pass_probs = []

    for aug_id in range(n_passes):
        try:
            arr = _apply_tta_augmentation(pil_image, aug_id)
            preds = model.predict(arr, verbose=0)[0]
            all_pass_probs.append(preds)
        except Exception as e:
            logger.warning(f"TTA pass {aug_id} failed, skipping: {e}")

    if not all_pass_probs:
        logger.error("All TTA passes failed — falling back to single-pass")
        return predict_single_image(image_path, use_tta=False)

    all_pass_probs = np.array(all_pass_probs)        # (n_passes, num_classes)
    avg_preds = np.mean(all_pass_probs, axis=0)       # (num_classes,)

    latency_ms = round((_time.perf_counter() - t_start) * 1000, 1)

    # Per-pass agreement
    per_pass_top = [
        class_names[int(np.argmax(p))] for p in all_pass_probs
    ]
    top_idx = int(np.argmax(avg_preds))
    top_class = class_names[top_idx]
    agreement_pct = round(
        per_pass_top.count(top_class) / len(per_pass_top) * 100, 1
    )

    # Build Top-3 results
    top_indices = avg_preds.argsort()[-3:][::-1]
    results = []
    for idx in top_indices:
        entry = {
            "disease":    class_names[idx],
            "confidence": float(avg_preds[idx] * 100.0),
        }
        results.append(entry)

    # Attach TTA metadata to top result
    results[0]["tta_used"] = True
    results[0]["tta_passes"] = len(all_pass_probs)
    results[0]["tta_agreement"] = agreement_pct
    results[0]["latency_ms"] = latency_ms

    return results, avg_preds, all_pass_probs


# ─────────────────────────────────────────────────────────────
# Retry wrapper for model loading
# ─────────────────────────────────────────────────────────────
def load_with_retry(loader_fn, max_attempts: int = 3, delay: int = 2):
    """Attempt to call loader_fn up to max_attempts times."""
    for attempt in range(1, max_attempts + 1):
        try:
            return loader_fn()
        except Exception as exc:
            if attempt < max_attempts:
                print(f"[WARNING] Model load attempt {attempt} failed: {exc} — retrying in {delay}s")
                _time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Model failed to load after {max_attempts} attempts. "
                    f"Last error: {exc}"
                ) from exc


# ─────────────────────────────────────────────────────────────
# Quick self-test (run directly: python predict.py --test)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    dummy = PILImage.fromarray(
        np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
    )

    print("TTA Augmentation Smoke-Test:")
    for i in range(8):
        arr = _apply_tta_augmentation(dummy, i)
        assert arr.shape == (1, 224, 224, 3), f"Pass {i}: wrong shape {arr.shape}"
        assert arr.min() >= 0.0 and arr.max() <= 1.0, f"Pass {i}: values out of [0,1]"
        print(f"  Pass {i}: shape={arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}  [OK]")

    print("\nReliability Analysis Smoke-Test:")
    fake_confident = np.array([0.85, 0.05, 0.04, 0.03, 0.03], dtype=np.float32)
    r1 = analyze_prediction_reliability(fake_confident, 85.0)
    print(f"  High-confidence: is_reliable={r1['is_reliable']}  [OK]")
    assert r1["is_reliable"]

    fake_uncertain = np.array([0.22, 0.20, 0.20, 0.19, 0.19], dtype=np.float32)
    r2 = analyze_prediction_reliability(fake_uncertain, 22.0)
    print(f"  Low-confidence:  is_reliable={r2['is_reliable']}  [OK]")
    assert not r2["is_reliable"]

    print("\nAll smoke-tests passed [OK]")
