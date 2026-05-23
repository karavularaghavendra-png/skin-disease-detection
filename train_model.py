"""
Training Pipeline for Skin Disease Detection.

Usage:
    python train_model.py --dataset dataset/skin_dataset
    python train_model.py --dataset dataset/skin_dataset --model cnn
    python train_model.py --dataset dataset/skin_dataset --model transfer --epochs 25
    python train_model.py --dataset dataset/skin_dataset --fine-tune --fine-tune-epochs 10
"""

import os
import sys
import json
import argparse

# ── Suppress TF 2.15 internal deprecation warnings ──
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import matplotlib
matplotlib.use("Agg")                # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf

# ── GPU memory growth (prevents OOM on shared GPUs) ──────────
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    GlobalAveragePooling2D, BatchNormalization,
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from dataset_loader import load_and_split
from preprocess import SkinDiseaseDataGenerator

# ─────────────────────────────────────────────────────────────
# Output Paths
# ─────────────────────────────────────────────────────────────
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.h5")
CLASS_NAMES_PATH = os.path.join("utils", "class_names.json")
EVAL_DIR = "evaluation"


# ─────────────────────────────────────────────────────────────
# Model Architectures
# ─────────────────────────────────────────────────────────────
def build_custom_cnn(num_classes: int, input_shape=(224, 224, 3)):
    """Builds a custom 3-block CNN with BatchNorm and Dropout."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="Custom_CNN",
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_transfer_model(num_classes: int, input_shape=(224, 224, 3)):
    """Builds a MobileNetV2-based transfer-learning model."""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False  # Freeze base

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out, name="MobileNetV2_Transfer")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ─────────────────────────────────────────────────────────────
# Fine-Tuning (unfreeze top MobileNetV2 layers)
# ─────────────────────────────────────────────────────────────
def fine_tune_model(model, train_gen, val_gen, epochs=10, unfreeze_layers=20):
    """
    Unfreeze the last `unfreeze_layers` of the base MobileNetV2 and
    train with a very low learning rate to adapt pretrained features
    to the skin disease domain.

    Typically boosts accuracy by 2–5% over frozen-base training.
    """
    # Find the MobileNetV2 base (first layer in a functional Model)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        print("  [WARNING] Could not find base model for fine-tuning. Skipping.")
        return None

    # Unfreeze last N layers
    base_model.trainable = True
    total_layers = len(base_model.layers)
    for layer in base_model.layers[:total_layers - unfreeze_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n[FINE-TUNE] Unfroze {trainable_count}/{total_layers} base layers")

    # Recompile with very low LR to avoid catastrophic forgetting
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=3,
            restore_best_weights=True, verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=2, min_lr=1e-7, verbose=1,
        ),
    ]

    print(f"\n[FINE-TUNE] Training for up to {epochs} epochs (lr=1e-5) ...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


# ─────────────────────────────────────────────────────────────
# Evaluation & Plotting
# ─────────────────────────────────────────────────────────────
def plot_training_history(history, save_dir=EVAL_DIR, prefix=""):
    """Saves accuracy and loss curves as PNG files."""
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_{prefix}" if prefix else ""

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"accuracy_plot{tag}.png"), dpi=150)
    plt.close()

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"loss_plot{tag}.png"), dpi=150)
    plt.close()

    print(f"  [PLOT] Accuracy/loss plots saved to {save_dir}/")


def plot_roc_curves(y_true_labels, y_pred_probs, class_names, save_dir=EVAL_DIR):
    """Generate multi-class One-vs-Rest ROC curves with per-class AUC.

    Saves a publication-quality ROC plot to evaluation/roc_curve.png.
    This is a standard metric for medical AI — shows the tradeoff
    between sensitivity (true positive rate) and specificity.

    Args:
        y_true_labels: List/array of true class indices.
        y_pred_probs:  Array of shape (n_samples, n_classes) with predicted probabilities.
        class_names:   List of class name strings.
        save_dir:      Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_classes = len(class_names)

    # Binarize true labels for One-vs-Rest
    y_true_bin = label_binarize(y_true_labels, classes=list(range(n_classes)))
    y_scores = np.array(y_pred_probs)

    plt.figure(figsize=(10, 8))

    # Colour palette for curves
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    # Per-class ROC
    all_auc = []
    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        all_auc.append(roc_auc)
        plt.plot(
            fpr, tpr, color=color, linewidth=2,
            label=f"{cls_name.capitalize()} (AUC = {roc_auc:.3f})",
        )

    # Micro-average ROC (overall)
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(
        fpr_micro, tpr_micro, color="navy", linewidth=2.5, linestyle="--",
        label=f"Micro-average (AUC = {roc_auc_micro:.3f})",
    )

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], color="grey", linewidth=1, linestyle=":")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Multi-Class ROC Curve (One-vs-Rest)", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Save AUC scores to text file
    auc_path = os.path.join(save_dir, "auc_scores.txt")
    with open(auc_path, "w") as f:
        f.write("Per-Class AUC Scores (One-vs-Rest)\n")
        f.write("=" * 45 + "\n\n")
        for cls_name, cls_auc in zip(class_names, all_auc):
            f.write(f"  {cls_name.capitalize():<20} AUC = {cls_auc:.4f}\n")
        f.write(f"\n  {'Micro-average':<20} AUC = {roc_auc_micro:.4f}\n")
        macro_auc = np.mean(all_auc)
        f.write(f"  {'Macro-average':<20} AUC = {macro_auc:.4f}\n")

    print(f"  [SAVED] ROC curve saved to {roc_path}")
    print(f"  [SAVED] AUC scores saved to {auc_path}")
    print(f"  [INFO]  Micro-average AUC = {roc_auc_micro:.4f}")


def evaluate_model(model, test_gen, class_names, save_dir=EVAL_DIR):
    """
    Evaluates the model on the test set and generates:
    - Classification report (accuracy, precision, recall, F1)
    - Confusion matrix heatmap
    - ROC curves with per-class AUC scores
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION (Test Set)")
    print("=" * 55)

    y_true, y_pred_probs = [], []
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred_probs.extend(preds)

    y_pred = np.argmax(y_pred_probs, axis=1)

    # ── Classification Report ──
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    print("\n" + report)

    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)
    print(f"  [SAVED] Report saved to {report_path}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  [SAVED] Confusion matrix saved to {cm_path}")

    # ── ROC Curves + AUC Scores ──
    print("\n  Generating ROC curves and AUC scores...")
    plot_roc_curves(y_true, y_pred_probs, class_names, save_dir)


# ─────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train the Skin Disease Detection model.")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the dataset directory (e.g. dataset/skin_dataset)",
    )
    parser.add_argument(
        "--model", type=str, choices=["cnn", "transfer"], default="transfer",
        help="Model architecture: 'cnn' (Custom CNN) or 'transfer' (MobileNetV2). Default: transfer",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Maximum training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--fine-tune", action="store_true",
        help="After initial training, unfreeze top MobileNetV2 layers and fine-tune (transfer only)",
    )
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=10,
        help="Epochs for fine-tuning phase (default: 10)",
    )
    parser.add_argument(
        "--fine-tune-layers", type=int, default=20,
        help="Number of base model layers to unfreeze during fine-tuning (default: 20)",
    )
    args = parser.parse_args()

    # ── 1. Load & Split Dataset ──
    print("\n[INFO] Loading dataset ...")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_split(args.dataset)
    num_classes = len(class_names)

    # ── 2. Save class names ──
    os.makedirs(os.path.dirname(CLASS_NAMES_PATH), exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"  [SAVED] Class names saved to {CLASS_NAMES_PATH}")

    # ── 3. Create Data Generators ──
    print("\n[INFO] Initialising data generators ...")
    train_gen = SkinDiseaseDataGenerator(X_train, y_train, num_classes, batch_size=args.batch_size, is_training=True)
    val_gen   = SkinDiseaseDataGenerator(X_val,   y_val,   num_classes, batch_size=args.batch_size, is_training=False)
    test_gen  = SkinDiseaseDataGenerator(X_test,  y_test,  num_classes, batch_size=args.batch_size, is_training=False)

    # ── 4. Build Model ──
    if args.model == "cnn":
        model = build_custom_cnn(num_classes)
        print("\n[MODEL] Architecture: Custom CNN")
    else:
        model = build_transfer_model(num_classes)
        print("\n[MODEL] Architecture: MobileNetV2 Transfer Learning")

    model.summary()

    # ── 5. Callbacks ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    # ── 6. Train (frozen base) ──
    print(f"\n[TRAIN] Starting training for up to {args.epochs} epochs ...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── 6b. Plot frozen-base training ──
    plot_training_history(history, prefix="frozen")

    # ── 7. Fine-Tune (optional) ──
    if args.fine_tune and args.model == "transfer":
        ft_history = fine_tune_model(
            model, train_gen, val_gen,
            epochs=args.fine_tune_epochs,
            unfreeze_layers=args.fine_tune_layers,
        )
        if ft_history is not None:
            plot_training_history(ft_history, prefix="finetune")
    elif args.fine_tune and args.model == "cnn":
        print("\n[WARNING] --fine-tune is only supported with --model transfer. Skipping.")

    # ── 8. Evaluate ──
    print("\n[EVAL] Generating evaluation metrics ...")
    evaluate_model(model, test_gen, class_names)

    # ── 9. Done ──
    print("\n" + "=" * 55)
    print("  [OK] TRAINING COMPLETE")
    print("=" * 55)
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Evaluation in:  {EVAL_DIR}/")
    if args.fine_tune and args.model == "transfer":
        print(f"  Fine-tuning:    {args.fine_tune_layers} layers unfrozen, {args.fine_tune_epochs} epochs")
    print()
    print("  Next step:")
    print("    streamlit run app.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
