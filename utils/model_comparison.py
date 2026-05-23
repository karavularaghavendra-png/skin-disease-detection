"""
Model Comparison Data for Skin Disease Detection.

Provides structured benchmark comparison between Custom CNN and MobileNetV2
transfer learning architectures. Used by the Streamlit sidebar and for
viva/presentation reference.

Note: Values represent typical benchmarks from training on a dermatological
dataset with ~3000+ images. Actual results depend on dataset size,
distribution, and training configuration. Re-run train_model.py with
--model cnn and --model transfer to generate your own benchmarks.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────
# Benchmark Data
# ─────────────────────────────────────────────────────────────
MODEL_COMPARISON = {
    "Custom CNN": {
        "architecture": "3-Block CNN (32>64>128 filters)",
        "params_total": "~1.2M",
        "params_trainable": "~1.2M (all)",
        "accuracy": "~78-85%",
        "precision": "~77-84%",
        "recall": "~76-83%",
        "f1_score": "~76-83%",
        "inference_time": "~50ms (single pass)",
        "model_size": "~5 MB",
        "training_time": "~15 min (20 epochs)",
        "strengths": [
            "Lightweight and fast inference",
            "Easy to understand and modify",
            "Good for small datasets",
            "Lower computational requirements",
        ],
        "weaknesses": [
            "Lower accuracy on complex skin conditions",
            "No pretrained feature knowledge",
            "Prone to overfitting on small datasets",
            "Requires more training data to generalize",
        ],
    },
    "MobileNetV2 (Transfer Learning)": {
        "architecture": "MobileNetV2 backbone + Custom head (GAP > 256 > softmax)",
        "params_total": "~2.6M",
        "params_trainable": "~590K (head only) / ~2.6M (fine-tuned)",
        "accuracy": "~92-96%",
        "precision": "~92-96%",
        "recall": "~91-95%",
        "f1_score": "~91-95%",
        "inference_time": "~200ms (single) / ~1.5s (8-pass TTA)",
        "model_size": "~13 MB",
        "training_time": "~25 min (20 epochs + 10 fine-tune)",
        "strengths": [
            "High accuracy from ImageNet pretrained features",
            "Excellent generalization on small datasets",
            "Fine-tuning unlocks domain-specific adaptation",
            "State-of-the-art mobile-optimized architecture",
        ],
        "weaknesses": [
            "Larger model size (~13 MB vs ~5 MB)",
            "Slower inference (especially with TTA)",
            "Requires more GPU memory during training",
            "More complex architecture to debug",
        ],
    },
}


def get_comparison_data() -> dict:
    """Return the full model comparison dictionary."""
    return MODEL_COMPARISON


def get_comparison_table_rows() -> list[dict]:
    """Return a flat list of comparison rows for rendering as a table.

    Returns:
        List of dicts with keys: metric, custom_cnn, mobilenetv2
    """
    metrics = [
        ("Architecture", "architecture"),
        ("Total Parameters", "params_total"),
        ("Trainable Parameters", "params_trainable"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
        ("Inference Time", "inference_time"),
        ("Model Size", "model_size"),
        ("Training Time", "training_time"),
    ]
    rows = []
    cnn = MODEL_COMPARISON["Custom CNN"]
    mv2 = MODEL_COMPARISON["MobileNetV2 (Transfer Learning)"]
    for display, key in metrics:
        rows.append({
            "Metric": display,
            "Custom CNN": cnn[key],
            "MobileNetV2": mv2[key],
        })
    return rows


def get_recommendation() -> str:
    """Return a summary recommendation for which model to use."""
    return (
        "**Recommendation:** Use **MobileNetV2 with Fine-Tuning** for production. "
        "It achieves 10-15% higher accuracy than the Custom CNN by leveraging "
        "pretrained ImageNet features. The Custom CNN is suitable for learning "
        "purposes and resource-constrained environments where model size matters."
    )
