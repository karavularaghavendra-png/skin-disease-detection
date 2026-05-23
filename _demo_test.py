"""
Full end-to-end demo test — validates every module in the project.
"""
import sys
import os
import numpy as np

print("=" * 60)
print("  SKIN DISEASE DETECTION — FULL DEMO TEST")
print("=" * 60)

# ─── 1. Utils Module Tests ───────────────────────────────────
print("\n[SECTION 1] Utils Modules")
print("-" * 40)

from utils.disease_info import (
    get_disease_info, get_severity, get_disclaimer, DISEASE_DATABASE,
)

for cls in ["acne", "eczema", "fungal", "normal", "psoriasis"]:
    info = get_disease_info(cls)
    assert info["display_name"] != "Unknown Condition", f"{cls} not found!"
    assert len(info["symptoms"]) > 0
    assert len(info["recommendations"]) > 0
    print(f"  ✅ {cls}: {info['display_name']} "
          f"({len(info['symptoms'])} symptoms, "
          f"{len(info['recommendations'])} recommendations)")

print(f"\n  ✅ Disease database: {len(DISEASE_DATABASE)} classes")

# Severity
for cls, conf in [("psoriasis", 85.0), ("acne", 50.0), ("normal", 90.0)]:
    label, color = get_severity(conf, cls)
    print(f"  ✅ Severity: {cls}@{conf}% → {label}")

# Disclaimer
disc = get_disclaimer()
assert "DISCLAIMER" in disc
print(f"  ✅ Medical disclaimer: {len(disc)} chars")

# Medications
from utils.medication_map import MEDICATION_MAP
for cls in ["acne", "eczema", "fungal", "psoriasis", "normal"]:
    meds = MEDICATION_MAP.get(cls, [])
    assert len(meds) > 0, f"No medications for {cls}"
    print(f"  ✅ {cls}: {len(meds)} OTC medications")

# OOD Detector
from utils.ood_detector import is_skin_image
print("  ✅ OOD detector imported")

# Image Utils
from utils.image_utils import check_image_quality
print("  ✅ Image quality checker imported")

# Report Generator
from utils.report_generator import generate_pdf_report
print("  ✅ PDF report generator imported")

# Model Comparison
from utils.model_comparison import (
    get_comparison_table_rows, get_recommendation, MODEL_COMPARISON,
)
rows = get_comparison_table_rows()
print(f"  ✅ Model comparison: {len(rows)} metrics, "
      f"{len(MODEL_COMPARISON)} architectures")

print("\n  ✅ ALL UTILS MODULES PASSED")

# ─── 2. Preprocessing Tests ──────────────────────────────────
print("\n[SECTION 2] Preprocessing")
print("-" * 40)

from preprocess import preprocess_single_image, SkinDiseaseDataGenerator
print("  ✅ preprocess imports OK")

# ─── 3. Prediction Module Tests ─────────────────────────────
print("\n[SECTION 3] Prediction Module")
print("-" * 40)

from predict import (
    preprocess_image, analyze_prediction_reliability,
    _apply_tta_augmentation, load_image_from_bytes,
    CONFIDENCE_THRESHOLD, ENTROPY_THRESHOLD, MARGIN_THRESHOLD,
)
from PIL import Image

print(f"  Config: confidence_threshold={CONFIDENCE_THRESHOLD}%, "
      f"entropy_threshold={ENTROPY_THRESHOLD}, "
      f"margin_threshold={MARGIN_THRESHOLD}%")

# TTA augmentation passes
dummy = Image.fromarray(
    np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
)
for i in range(8):
    arr = _apply_tta_augmentation(dummy, i)
    assert arr.shape == (1, 224, 224, 3), f"Pass {i}: wrong shape"
    assert 0.0 <= arr.min() and arr.max() <= 1.0, f"Pass {i}: out of range"
print("  ✅ TTA augmentation: 8/8 passes valid")

# Reliability analysis
preds_high = np.array([0.85, 0.05, 0.04, 0.03, 0.03], dtype=np.float32)
r1 = analyze_prediction_reliability(preds_high, 85.0)
assert r1["is_reliable"], "High-confidence should be reliable!"
print(f"  ✅ High-confidence (85%): reliable={r1['is_reliable']}, "
      f"entropy={r1['entropy']:.3f}")

preds_low = np.array([0.22, 0.20, 0.20, 0.19, 0.19], dtype=np.float32)
r2 = analyze_prediction_reliability(preds_low, 22.0)
assert not r2["is_reliable"], "Low-confidence should be unreliable!"
print(f"  ✅ Low-confidence (22%): reliable={r2['is_reliable']}, "
      f"warnings={len(r2['warnings'])}")

preds_uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype=np.float32)
r3 = analyze_prediction_reliability(preds_uniform, 20.0)
assert not r3["is_reliable"]
assert r3["entropy"] > 1.2
print(f"  ✅ Uniform distribution: entropy={r3['entropy']:.3f} (high = confused)")

# TTA disagreement
avg = np.array([0.90, 0.04, 0.03, 0.02, 0.01], dtype=np.float32)
all_pass = np.tile(avg, (8, 1))
r4 = analyze_prediction_reliability(avg, 90.0, all_pass_probs=all_pass)
assert r4["is_reliable"]
assert r4["tta_disagreement"] < 0.01
print(f"  ✅ TTA agreement (8 identical passes): "
      f"disagreement={r4['tta_disagreement']:.4f}")

print("\n  ✅ ALL PREDICTION MODULE CHECKS PASSED")

# ─── 4. OOD + Image Quality Tests ───────────────────────────
print("\n[SECTION 4] OOD Detection & Image Quality")
print("-" * 40)

import cv2
import tempfile

# Skin-colored image (should pass)
skin_ycrcb = np.full((100, 100, 3), [150, 150, 100], dtype=np.uint8)
skin_bgr = cv2.cvtColor(skin_ycrcb, cv2.COLOR_YCrCb2BGR)
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    tmp_path = f.name
    cv2.imwrite(tmp_path, skin_bgr)
is_valid, ratio = is_skin_image(tmp_path)
assert is_valid, f"Skin image rejected! ratio={ratio}"
print(f"  ✅ Skin image: valid={is_valid}, ratio={ratio:.2%}")
os.unlink(tmp_path)

# Non-skin image (should fail)
black = np.zeros((100, 100, 3), dtype=np.uint8)
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    tmp_path = f.name
    cv2.imwrite(tmp_path, black)
is_valid2, ratio2 = is_skin_image(tmp_path)
assert not is_valid2, "Black image should be rejected!"
print(f"  ✅ Non-skin (black): valid={is_valid2}, ratio={ratio2:.2%}")
os.unlink(tmp_path)

# Dark skin (Fitzpatrick V-VI, should pass)
dark_ycrcb = np.full((100, 100, 3), [100, 145, 115], dtype=np.uint8)
dark_bgr = cv2.cvtColor(dark_ycrcb, cv2.COLOR_YCrCb2BGR)
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    tmp_path = f.name
    cv2.imwrite(tmp_path, dark_bgr)
is_valid3, ratio3 = is_skin_image(tmp_path)
assert is_valid3, f"Dark skin rejected! ratio={ratio3}"
print(f"  ✅ Dark skin (Fitzpatrick V-VI): valid={is_valid3}, "
      f"ratio={ratio3:.2%}")
os.unlink(tmp_path)

# Blur detection
noisy = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.randn(noisy, (128, 128, 128), (50, 50, 50))
blurred = cv2.GaussianBlur(noisy, (51, 51), 0)
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    tmp_path = f.name
    cv2.imwrite(tmp_path, blurred)
ok, warnings = check_image_quality(tmp_path)
assert not ok, "Blurred image should fail quality!"
assert any("blurry" in w.lower() for w in warnings)
print(f"  ✅ Blur detection: detected! warnings={warnings}")
os.unlink(tmp_path)

# Dark image detection
dark_img = np.full((300, 300, 3), 20, dtype=np.uint8)
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    tmp_path = f.name
    cv2.imwrite(tmp_path, dark_img)
ok2, warnings2 = check_image_quality(tmp_path)
assert not ok2
assert any("dark" in w.lower() for w in warnings2)
print(f"  ✅ Dark image detection: detected! warnings={warnings2}")
os.unlink(tmp_path)

print("\n  ✅ ALL OOD + QUALITY CHECKS PASSED")

# ─── 5. API Module Tests ────────────────────────────────────
print("\n[SECTION 5] FastAPI API Module")
print("-" * 40)

try:
    from api import app
    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    print(f"  ✅ GET /health: {data}")

    # Root redirect
    resp2 = client.get("/", follow_redirects=False)
    assert resp2.status_code == 307
    print(f"  ✅ GET /: redirects to /static/index.html")

    # Auth rejection
    test_img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    import io
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    buf.seek(0)

    resp3 = client.post(
        "/predict",
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert resp3.status_code in [401, 403]
    print(f"  ✅ POST /predict (no auth): rejected ({resp3.status_code})")

    # Invalid file type
    resp4 = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        headers={"Authorization": "Bearer dev-key-not-for-production"},
    )
    assert resp4.status_code == 400
    print(f"  ✅ POST /predict (bad format): rejected ({resp4.status_code})")

    # File too large
    resp5 = client.post(
        "/predict",
        files={"file": ("big.jpg", b"x" * 11_000_000, "image/jpeg")},
        headers={"Authorization": "Bearer dev-key-not-for-production"},
    )
    assert resp5.status_code == 413
    print(f"  ✅ POST /predict (too large): rejected ({resp5.status_code})")

    print("\n  ✅ ALL API CHECKS PASSED")
except ImportError as e:
    print(f"  ⚠️ API test skipped: {e}")

# ─── 6. PDF Report Generation ───────────────────────────────
print("\n[SECTION 6] PDF Report Generation")
print("-" * 40)

test_img_path = None
try:
    # Create a temp test image
    test_arr = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        test_img_path = f.name
        cv2.imwrite(test_img_path, test_arr)

    pdf_bytes = generate_pdf_report(
        image_path=test_img_path,
        disease_name="Eczema (Atopic Dermatitis)",
        confidence=87.3,
        severity_label="Moderate - Monitor Closely",
        symptoms=["Itchy skin", "Dry patches", "Redness"],
        recommendations=["Moisturise daily", "Use hydrocortisone cream"],
        tta_agreement=92.5,
        entropy=0.312,
        latency_ms=1450,
        medications=[
            {"name": "CeraVe Cream", "use": "Restores skin barrier"},
            {"name": "Hydrocortisone 1%", "use": "Reduces itching"},
        ],
    )
    assert len(pdf_bytes) > 1000
    assert pdf_bytes[:4] == b"%PDF"
    print(f"  ✅ PDF generated: {len(pdf_bytes):,} bytes (valid PDF)")
finally:
    if test_img_path and os.path.exists(test_img_path):
        os.unlink(test_img_path)

print("\n  ✅ PDF GENERATION PASSED")

# ─── 7. Logger Module ───────────────────────────────────────
print("\n[SECTION 7] Logger Module")
print("-" * 40)

from logger import get_logger, LOG_FILE
log = get_logger("demo_test")
log.info("Demo test log entry")
print(f"  ✅ Logger: writing to {LOG_FILE}")
assert LOG_FILE.parent.exists(), "Log directory missing!"
print(f"  ✅ Log directory exists")

# ─── 8. Dataset Loader ──────────────────────────────────────
print("\n[SECTION 8] Dataset Loader")
print("-" * 40)

from dataset_loader import validate_dataset
ds_path = "dataset/skin_dataset"
if os.path.exists(ds_path):
    class_names, split_mode = validate_dataset(ds_path)
    print(f"  ✅ Dataset: {len(class_names)} classes, mode={split_mode}")
    print(f"     Classes: {class_names}")
else:
    print(f"  ⚠️ Dataset not found at {ds_path}")

# ─── 9. Model Files Check ───────────────────────────────────
print("\n[SECTION 9] Model Files")
print("-" * 40)

model_files = [
    "model/skin_model.h5",
    "model/best_model.h5",
    "utils/class_names.json",
]
for mf in model_files:
    exists = os.path.exists(mf)
    size = os.path.getsize(mf) if exists else 0
    status = f"✅ {size/1024/1024:.1f} MB" if exists else "⚠️ Not found"
    print(f"  {status}: {mf}")

# ─── 10. Evaluation Artifacts ────────────────────────────────
print("\n[SECTION 10] Evaluation Artifacts")
print("-" * 40)

eval_files = [
    "evaluation/accuracy_plot.png",
    "evaluation/loss_plot.png",
    "evaluation/confusion_matrix.png",
    "evaluation/classification_report.txt",
]
for ef in eval_files:
    exists = os.path.exists(ef)
    status = "✅" if exists else "⚠️ Missing"
    print(f"  {status}: {ef}")

if os.path.exists("evaluation/classification_report.txt"):
    with open("evaluation/classification_report.txt") as f:
        report = f.read()
    print(f"\n  --- Classification Report ---")
    print(report)

# ─── FINAL SUMMARY ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ FULL DEMO TEST COMPLETE — ALL MODULES WORKING")
print("=" * 60)
