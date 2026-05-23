"""Convert the Keras .h5 model to TFLite with integer quantisation.

Usage:
    python scripts/convert_to_tflite.py

Outputs:
    model/skin_model.tflite   (~3 MB, 4x smaller than .h5)
    model/skin_model_fp16.tflite  (float16 variant for edge devices)
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).parent.parent / "model"
CANDIDATES = ["best_model.h5", "skin_model.h5", "skin_model.keras"]


def find_model() -> Path:
    for name in CANDIDATES:
        p = MODEL_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No model file found in {MODEL_DIR}. "
        f"Expected one of: {CANDIDATES}"
    )


def convert():
    import tensorflow as tf

    # Enable memory growth to avoid OOM errors
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    model_path = find_model()
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # ── Standard dynamic-range quantisation ──────────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    t0 = time.time()
    tflite_model = converter.convert()
    print(f"Conversion done in {time.time()-t0:.1f}s")

    out_std = MODEL_DIR / "skin_model.tflite"
    out_std.write_bytes(tflite_model)
    print(f"Saved → {out_std}  ({len(tflite_model)/1e6:.2f} MB)")

    # ── Float16 quantisation (for edge devices) ───────────────
    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_fp16.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter_fp16.convert()

    out_fp16 = MODEL_DIR / "skin_model_fp16.tflite"
    out_fp16.write_bytes(tflite_fp16)
    print(f"Saved → {out_fp16}  ({len(tflite_fp16)/1e6:.2f} MB)")

    # ── Benchmark inference speed ─────────────────────────────
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(in_details[0]["index"], dummy)

    N = 20
    t0 = time.time()
    for _ in range(N):
        interpreter.invoke()
    avg_ms = (time.time() - t0) / N * 1000
    print(f"TFLite avg inference: {avg_ms:.1f} ms  ({1000/avg_ms:.0f} fps)")


if __name__ == "__main__":
    convert()
