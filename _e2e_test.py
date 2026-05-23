"""
End-to-End Integration Test for All 5 Enhancements.
Tests prediction pipeline, Grad-CAM, PDF report, model comparison, and ROC import.
"""
import os
import sys
import time

print("=" * 60)
print("  END-TO-END INTEGRATION TEST")
print("=" * 60)

# ── Test 1: Model Loading ──────────────────────────────────────
print("\n[TEST 1] Model Loading...")
from predict import load_model_cached
model, class_names = load_model_cached()
print(f"  Model loaded: {type(model).__name__}")
print(f"  Classes: {class_names}")
assert len(class_names) == 5, f"Expected 5 classes, got {len(class_names)}"
print("  PASS")

# ── Test 2: Prediction with TTA ────────────────────────────────
print("\n[TEST 2] Prediction with TTA...")
test_image = r"dataset\skin_dataset\test\acne\acne_0001.jpg"
assert os.path.exists(test_image), f"Test image not found: {test_image}"

from predict import predict_single_image, analyze_prediction_reliability
start = time.time()
top_results, avg_preds, all_pass_probs = predict_single_image(test_image, use_tta=True)
elapsed = time.time() - start

disease = top_results[0]["disease"]
confidence = top_results[0]["confidence"]
tta_agreement = top_results[0].get("tta_agreement", 0)
latency = top_results[0].get("latency_ms", 0)

print(f"  Disease: {disease}")
print(f"  Confidence: {confidence:.1f}%")
print(f"  TTA Agreement: {tta_agreement}%")
print(f"  Latency: {latency:.0f}ms (wall: {elapsed:.1f}s)")

reliability = analyze_prediction_reliability(avg_preds, confidence, all_pass_probs=all_pass_probs)
print(f"  Reliable: {reliability['is_reliable']}")
print(f"  Entropy: {reliability['entropy']:.4f}")
print("  PASS")

# ── Test 3: Grad-CAM Heatmap (Enhancement #1) ─────────────────
print("\n[TEST 3] Grad-CAM Heatmap Generation...")
import numpy as np
from preprocess import preprocess_single_image
from explainability import generate_gradcam_overlay

img_array = preprocess_single_image(test_image)
pred_idx = int(np.argmax(avg_preds))
overlay = generate_gradcam_overlay(model, img_array, pred_idx)

assert overlay is not None, "Grad-CAM returned None"
assert overlay.shape[0] > 0 and overlay.shape[1] > 0, "Grad-CAM returned empty image"
print(f"  Heatmap shape: {overlay.shape}")
print("  PASS")

# ── Test 4: PDF Report (Enhancement #3) ───────────────────────
print("\n[TEST 4] PDF Report Generation...")
import tempfile
import cv2
from utils.report_generator import generate_pdf_report
from utils.disease_info import get_disease_info, get_severity
from utils.medication_map import MEDICATION_MAP

info = get_disease_info(disease)
severity_label, _ = get_severity(confidence, disease)
medications = MEDICATION_MAP.get(disease.strip().lower(), [])

# Save heatmap temp
heatmap_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
cv2.imwrite(heatmap_tmp.name, overlay)
heatmap_tmp.close()

pdf_bytes = generate_pdf_report(
    image_path=test_image,
    disease_name=info.get("display_name", disease),
    confidence=confidence,
    severity_label=severity_label,
    symptoms=info.get("symptoms", []),
    recommendations=info.get("recommendations", []),
    heatmap_path=heatmap_tmp.name,
    tta_agreement=tta_agreement,
    entropy=reliability["entropy"],
    latency_ms=latency,
    medications=medications if medications else None,
)
os.unlink(heatmap_tmp.name)

print(f"  PDF size: {len(pdf_bytes)} bytes")
assert len(pdf_bytes) > 1000, "PDF too small"

# Save PDF for manual inspection
pdf_path = os.path.join("evaluation", "test_report.pdf")
os.makedirs("evaluation", exist_ok=True)
with open(pdf_path, "wb") as f:
    f.write(pdf_bytes)
print(f"  Saved to: {pdf_path}")
print("  PASS")

# ── Test 5: Model Comparison (Enhancement #5) ─────────────────
print("\n[TEST 5] Model Comparison Data...")
from utils.model_comparison import get_comparison_table_rows, get_recommendation, MODEL_COMPARISON

rows = get_comparison_table_rows()
assert len(rows) == 10, f"Expected 10 metrics, got {len(rows)}"
assert "Custom CNN" in MODEL_COMPARISON
assert "MobileNetV2 (Transfer Learning)" in MODEL_COMPARISON

rec = get_recommendation()
assert "MobileNetV2" in rec

print(f"  Metrics: {len(rows)}")
print(f"  Models: {list(MODEL_COMPARISON.keys())}")
print("  PASS")

# ── Test 6: ROC/AUC Import (Enhancement #2) ───────────────────
print("\n[TEST 6] ROC/AUC Functions...")
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Quick validation with dummy data
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
y_scores = np.random.rand(10, 3)
y_bin = label_binarize(y_true, classes=[0, 1, 2])
fpr, tpr, _ = roc_curve(y_bin[:, 0], y_scores[:, 0])
roc_auc = auc(fpr, tpr)
print(f"  ROC/AUC import: OK")
print(f"  Test AUC (random): {roc_auc:.3f}")

# Verify the function exists in train_model
import importlib
train_mod = importlib.import_module("train_model")
assert hasattr(train_mod, "plot_roc_curves"), "plot_roc_curves not found in train_model"
print(f"  plot_roc_curves() in train_model: Found")
print("  PASS")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL 6 TESTS PASSED")
print("=" * 60)
print(f"\n  Disease detected: {disease}")
print(f"  Confidence: {confidence:.1f}%")
print(f"  Grad-CAM: OK ({overlay.shape})")
print(f"  PDF Report: OK ({len(pdf_bytes)} bytes)")
print(f"  Model Comparison: OK ({len(rows)} metrics)")
print(f"  ROC/AUC: OK (import + function found)")
print(f"  Report saved: {pdf_path}")
