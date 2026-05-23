"""Grad-CAM++ explainability for the skin disease classifier.

Usage::
    from explainability import generate_gradcam_overlay
    overlay_bgr = generate_gradcam_overlay(model, img_array, class_idx)
    # overlay_bgr is (H, W, 3) uint8 — convert for st.image:
    import cv2
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    st.image(overlay_rgb, caption="Model attention heatmap")
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def generate_gradcam_overlay(
    model,
    image_array: "np.ndarray",
    class_idx: int,
    alpha: float = 0.45,
) -> "np.ndarray":
    """Generate a Grad-CAM++ heatmap blended over the input image.

    Args:
        model:        Loaded Keras/TF model.
        image_array:  Preprocessed image shape (1, 224, 224, 3) float32.
        class_idx:    Predicted class index.
        alpha:        Heatmap blend weight (0 = original, 1 = full heatmap).

    Returns:
        BGR uint8 numpy array (224, 224, 3).
    """
    import cv2

    original_bgr = cv2.cvtColor(
        np.uint8(np.clip(image_array[0], 0, 1) * 255),
        cv2.COLOR_RGB2BGR,
    )

    try:
        from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
        from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
        from tf_keras_vis.utils.scores import CategoricalScore

        gradcam = GradcamPlusPlus(
            model,
            model_modifier=ReplaceToLinear(),
            clone=True,
        )
        cam = gradcam(CategoricalScore(class_idx), image_array, penultimate_layer=-1)
        heatmap_colour = cv2.applyColorMap(np.uint8(255 * cam[0]), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            original_bgr,
            1.0 - alpha,
            heatmap_colour,
            alpha,
            0,
        )
        log.info("Grad-CAM overlay generated for class %d", class_idx)
        return overlay

    except ImportError:
        log.warning("tf-keras-vis not installed. pip install tf-keras-vis")
        return original_bgr
    except Exception as exc:
        log.warning("Grad-CAM failed (%s) — returning original image", exc)
        return original_bgr
