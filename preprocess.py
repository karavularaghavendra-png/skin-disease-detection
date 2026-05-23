"""
Memory-efficient preprocess.py — Real-World Robust Edition
Loads images from file paths in batches — no RAM overflow!

Augmentation pipeline designed to close the domain gap between
clinical/curated training images and real-world smartphone photos.
"""

import os
import cv2
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

try:
    from tensorflow.keras.utils import Sequence
except ImportError:
    # Fallback for environments without TensorFlow (e.g. testing)
    class Sequence:
        """Minimal stub: real Sequence is provided by TF at runtime."""
        def __getitem__(self, idx): raise NotImplementedError
        def __len__(self): raise NotImplementedError
        def on_epoch_end(self): pass

IMG_SIZE = (224, 224)


def preprocess_single_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


class SkinDiseaseDataGenerator(Sequence):
    """
    Loads images from file paths in batches.
    Works with path lists from dataset_loader.

    Augmentation pipeline (10 stages) bridges the gap between
    curated clinical images and real-world smartphone captures.
    """

    def __init__(self, X, y, num_classes=5, batch_size=32,
                 augment=False, is_training=False):
        self.X           = X            # list of file paths
        self.y           = y            # list of int labels
        self.num_classes = num_classes
        self.batch_size  = batch_size
        self.augment     = augment or is_training
        self.indices     = list(range(len(self.X)))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_batch, y_batch = [], []
        for i in batch_idx:
            path = self.X[i]
            img  = cv2.imread(str(path))
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0

            if self.augment:
                img = self._augment(img)

            X_batch.append(img)
            y_batch.append(self.y[i])

        X_batch = np.array(X_batch, dtype=np.float32)

        # One-hot encode
        y_onehot = np.zeros((len(y_batch), self.num_classes), dtype=np.float32)
        for i, label in enumerate(y_batch):
            y_onehot[i, int(label)] = 1.0

        return X_batch, y_onehot

    # ─────────────────────────────────────────────────────────
    # Augmentation Pipeline — 10 stages for real-world robustness
    # ─────────────────────────────────────────────────────────
    def _augment(self, img):
        """
        Apply augmentations that simulate real-world image capture:

        Clinical image issues:
        1. Horizontal flip — mirror symmetry
        2. Brightness — simulate poor/varied lighting (0.6–1.4)
        3. Rotation — angled captures (±30°)
        4. Color jitter — color temperature shifts (hue/sat/contrast)

        Real-world gap closers:
        5. Gaussian noise — sensor noise from smartphone cameras
        6. Gaussian blur — out-of-focus / motion blur
        7. JPEG compression — artifacts from messaging apps (WhatsApp, etc.)
        8. Random zoom/crop — varied distances, framing

        Regularization:
        9.  Random cutout — rectangular occlusion
        10. Elastic deformation — simulates skin stretching
        """
        # 1. Horizontal flip (50% chance)
        if np.random.random() > 0.5:
            img = np.fliplr(img)

        # 2. Brightness adjustment (wider range: 0.6–1.4 for real-world lighting)
        factor = np.random.uniform(0.6, 1.4)
        img = np.clip(img * factor, 0, 1)

        # 3. Rotation (±30° — smartphones capture at all angles)
        if np.random.random() > 0.4:
            angle = np.random.uniform(-30, 30)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 4. Color jitter (40% chance — color temperature varies by lighting)
        if np.random.random() > 0.6:
            img = self._color_jitter(img)

        # 5. Gaussian noise (25% chance — sensor noise from phone cameras)
        if np.random.random() > 0.75:
            img = self._gaussian_noise(img)

        # 6. Gaussian blur (20% chance — simulates out-of-focus)
        if np.random.random() > 0.8:
            img = self._gaussian_blur(img)

        # 7. JPEG compression (15% chance — WhatsApp/social media artifacts)
        if np.random.random() > 0.85:
            img = self._jpeg_compression(img)

        # 8. Random zoom/crop (20% chance — varied distances)
        if np.random.random() > 0.8:
            img = self._random_zoom_crop(img)

        # 9. Random cutout (20% chance) — rectangular occlusion
        if np.random.random() > 0.8:
            img = self._random_cutout(img)

        # 10. Elastic deformation (15% chance) — simulates skin stretching
        if np.random.random() > 0.85:
            img = self._elastic_deformation(img)

        return img

    @staticmethod
    def _color_jitter(img, hue_range=0.08, sat_range=0.4, contrast_range=0.4):
        """Randomly adjust hue, saturation, and contrast.
        More aggressive than before to handle varied color temperatures.
        """
        # Convert to HSV for hue/saturation manipulation
        img_uint8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue shift
        hsv[:, :, 0] += np.random.uniform(-hue_range, hue_range) * 180
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)

        # Saturation shift
        sat_factor = np.random.uniform(1 - sat_range, 1 + sat_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

        img_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        img = img_uint8.astype(np.float32) / 255.0

        # Contrast adjustment
        contrast_factor = np.random.uniform(1 - contrast_range, 1 + contrast_range)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * contrast_factor + mean, 0, 1)

        return img

    @staticmethod
    def _gaussian_noise(img, std_range=(0.01, 0.05)):
        """Add Gaussian noise — simulates smartphone camera sensor noise."""
        std = np.random.uniform(*std_range)
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img + noise, 0, 1).astype(np.float32)

    @staticmethod
    def _gaussian_blur(img, kernel_range=(3, 7)):
        """Apply Gaussian blur — simulates out-of-focus / motion blur."""
        k = np.random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def _jpeg_compression(img, quality_range=(30, 70)):
        """Simulate JPEG compression artifacts (WhatsApp, social media)."""
        quality = np.random.randint(*quality_range)
        img_uint8 = (img * 255).astype(np.uint8)
        # Encode as JPEG with low quality, then decode back
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return decoded.astype(np.float32) / 255.0

    @staticmethod
    def _random_zoom_crop(img, zoom_range=(0.75, 1.0)):
        """Random zoom/crop — simulates images taken from different distances."""
        h, w = img.shape[:2]
        scale = np.random.uniform(*zoom_range)
        new_h, new_w = int(h * scale), int(w * scale)

        # Random crop position
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        cropped = img[top:top + new_h, left:left + new_w]
        # Resize back to original size
        return cv2.resize(cropped, (w, h)).astype(np.float32)

    @staticmethod
    def _random_cutout(img, max_holes=2, max_size=40):
        """Apply random rectangular cutout (fills with mean pixel value)."""
        h, w = img.shape[:2]
        fill_value = np.mean(img, axis=(0, 1))

        n_holes = np.random.randint(1, max_holes + 1)
        for _ in range(n_holes):
            cut_h = np.random.randint(10, max_size)
            cut_w = np.random.randint(10, max_size)
            y = np.random.randint(0, h - cut_h)
            x = np.random.randint(0, w - cut_w)
            img[y:y + cut_h, x:x + cut_w, :] = fill_value

        return img

    @staticmethod
    def _elastic_deformation(img, alpha=20, sigma=4):
        """
        Apply elastic deformation to simulate skin stretching/warping.
        Lower alpha/sigma = subtle; higher = more aggressive.
        """
        h, w = img.shape[:2]
        dx = gaussian_filter(np.random.randn(h, w) * alpha, sigma)
        dy = gaussian_filter(np.random.randn(h, w) * alpha, sigma)

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        map_x = (x_coords + dx).astype(np.float32)
        map_y = (y_coords + dy).astype(np.float32)

        # Apply per channel
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            result[:, :, c] = map_coordinates(
                img[:, :, c], [map_y, map_x], order=1, mode="reflect"
            )

        return np.clip(result, 0, 1).astype(np.float32)

    def on_epoch_end(self):
        if self.augment:  # Only shuffle training data
            np.random.shuffle(self.indices)
