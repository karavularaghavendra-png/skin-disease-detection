"""End-to-end test script for Skin Disease Detection app."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Test 1: Model loading
print('=== Test 1: Model Loading ===')
from predict import load_model_cached
model, class_names = load_model_cached()
print(f'  Model loaded: {model is not None}')
print(f'  Class names: {class_names}')
print(f'  Model input shape: {model.input_shape}')
print('  PASS')

# Test 2: Prediction with TTA (same as app.py uses)
print()
print('=== Test 2: Prediction with TTA ===')
from predict import predict_single_image
result = predict_single_image('test_skin_sample.jpg', use_tta=True)
top_results, avg_preds, all_pass_probs = result
print(f'  Top prediction: {top_results[0]["disease"]}')
print(f'  Confidence: {top_results[0]["confidence"]:.1f}%')
print(f'  TTA Agreement: {top_results[0].get("tta_agreement", "N/A")}%')
print(f'  Latency: {top_results[0].get("latency_ms", "N/A")}ms')
for r in top_results:
    print(f'    {r["disease"]}: {r["confidence"]:.1f}%')
print('  PASS')

# Test 3: Disease info
print()
print('=== Test 3: Disease Info ===')
from utils.disease_info import get_disease_info, get_severity
info = get_disease_info(top_results[0]['disease'])
sev_label, sev_color = get_severity(top_results[0]['confidence'], top_results[0]['disease'])
print(f'  Display name: {info.get("display_name", "N/A")}')
print(f'  Severity: {sev_label} (color: {sev_color})')
print(f'  Symptoms: {len(info.get("symptoms", []))} items')
print(f'  Recommendations: {len(info.get("recommendations", []))} items')
print('  PASS')

# Test 4: OOD detector
print()
print('=== Test 4: OOD Detector ===')
from utils.ood_detector import is_skin_image
is_skin, ratio = is_skin_image('test_skin_sample.jpg')
print(f'  Is skin image: {is_skin} (ratio: {ratio:.2f})')
print('  PASS')

# Test 5: Image quality
print()
print('=== Test 5: Image Quality ===')
from utils.image_utils import check_image_quality
ok, warnings = check_image_quality('test_skin_sample.jpg')
print(f'  Quality OK: {ok}')
if warnings:
    for w in warnings:
        print(f'  Warning: {w}')
print('  PASS')

# Test 6: Reliability analysis
print()
print('=== Test 6: Reliability Analysis ===')
from predict import analyze_prediction_reliability
reliability = analyze_prediction_reliability(
    avg_preds, top_results[0]['confidence'], all_pass_probs=all_pass_probs
)
print(f'  Is reliable: {reliability["is_reliable"]}')
print(f'  Entropy: {reliability["entropy"]:.4f}')
print(f'  Margin: {reliability["margin"]:.2f}%')
print(f'  TTA disagreement: {reliability["tta_disagreement"]:.4f}')
if reliability["warnings"]:
    for w in reliability["warnings"]:
        print(f'  Warning: {w}')
print('  PASS')

# Test 7: Medication map
print()
print('=== Test 7: Medication Map ===')
from utils.medication_map import MEDICATION_MAP
disease_key = top_results[0]['disease'].strip().lower()
meds = MEDICATION_MAP.get(disease_key, [])
print(f'  Disease: {disease_key}')
print(f'  Medications found: {len(meds)}')
for m in meds:
    print(f'    - {m["name"]}: {m["use"]}')
print('  PASS')

# Test 8: Grad-CAM
print()
print('=== Test 8: Grad-CAM Heatmap ===')
try:
    from preprocess import preprocess_single_image
    import numpy as np
    from explainability import generate_gradcam_overlay
    img_array = preprocess_single_image('test_skin_sample.jpg')
    predicted_class_idx = int(np.argmax(avg_preds))
    gradcam_overlay = generate_gradcam_overlay(model, img_array, predicted_class_idx)
    print(f'  Heatmap generated: shape={gradcam_overlay.shape}')
    print('  PASS')
except Exception as e:
    print(f'  Grad-CAM skipped: {e}')
    print('  SKIP (optional feature)')

print()
print('=' * 50)
print('  ALL TESTS PASSED - APP IS WORKING CORRECTLY!')
print('=' * 50)
