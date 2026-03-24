"""
Brand Vision Prediction Pipeline.
Orchestrates YOLOv8 logo detection + EfficientNet-B0 classification
using ONNX Runtime only (no PyTorch at inference time).
"""

import os
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
HF_REPO_ID = os.getenv("HF_WEIGHTS_REPO", "YOUR_USERNAME/brand-vision-weights")


def ensure_weights() -> None:
    """Download ONNX weight files from HF Hub if not already present locally."""
    WEIGHTS_DIR.mkdir(exist_ok=True)

    required_files = [
        "logo_detector.onnx",
        "category_classifier.onnx",
        "category_classifier_labels.json",
        "brand_classifier.onnx",
        "brand_classifier_labels.json",
    ]

    for filename in required_files:
        dest = WEIGHTS_DIR / filename
        if not dest.exists():
            print(f"[Weights] Downloading {filename} from {HF_REPO_ID}...")
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(WEIGHTS_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"[Weights] Downloaded: {filename}")
        else:
            print(f"[Weights] Found cached: {filename}")


# Download weights at module import time (runs once on startup)
ensure_weights()


class BrandVisionPipeline:
    """Full prediction pipeline: detection → classification → response assembly."""

    def __init__(self) -> None:
        # Load ONNX inference sessions
        self.category_sess = ort.InferenceSession(
            str(WEIGHTS_DIR / "category_classifier.onnx")
        )
        self.brand_sess = ort.InferenceSession(
            str(WEIGHTS_DIR / "brand_classifier.onnx")
        )
        self.detector_sess = ort.InferenceSession(
            str(WEIGHTS_DIR / "logo_detector.onnx")
        )

        # Load class label lists
        with open(WEIGHTS_DIR / "category_classifier_labels.json") as f:
            self.category_labels: list[str] = json.load(f)
        with open(WEIGHTS_DIR / "brand_classifier_labels.json") as f:
            self.brand_labels: list[str] = json.load(f)

        # ImageNet normalisation constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print("BrandVisionPipeline initialised ✓")

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_efficientnet(self, img: Image.Image) -> np.ndarray:
        """Resize + normalise a PIL image for EfficientNet-B0 (224×224)."""
        img_rgb = img.convert("RGB").resize((224, 224))
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(arr, 0)  # (1, 3, 224, 224)

    def _preprocess_yolo(self, img: Image.Image, size: int = 640) -> np.ndarray:
        """Letterbox-resize + normalise for YOLOv8 ONNX (640×640)."""
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size
        scale = size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img_rgb.resize((new_w, new_h))

        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(arr, 0), scale, pad_x, pad_y  # (1, 3, 640, 640)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _detect_logo(
        self, img: Image.Image
    ) -> tuple[list[int] | None, Image.Image]:
        """Run YOLOv8 ONNX logo detection; return (bbox, logo_crop)."""
        inp, scale, pad_x, pad_y = self._preprocess_yolo(img)
        raw = self.detector_sess.run(None, {"images": inp})[0]  # (1, 5, N)

        if raw.shape[-1] == 0:
            return None, img

        # raw shape: (1, 5, N) → (N, 5)  [cx, cy, w, h, conf]
        preds = raw[0].T
        best_idx = preds[:, 4].argmax()
        conf = float(preds[best_idx, 4])
        if conf < 0.25:
            return None, img

        cx, cy, bw, bh = preds[best_idx, :4]
        # De-letterbox: remove padding then rescale back to original coords
        img_w, img_h = img.size
        x1 = int(((cx - bw / 2) - pad_x) / scale)
        y1 = int(((cy - bh / 2) - pad_y) / scale)
        x2 = int(((cx + bw / 2) - pad_x) / scale)
        y2 = int(((cy + bh / 2) - pad_y) / scale)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # 10% padding on crop
        px = int((x2 - x1) * 0.1)
        py = int((y2 - y1) * 0.1)
        crop = img.crop(
            (max(0, x1 - px), max(0, y1 - py), min(img_w, x2 + px), min(img_h, y2 + py))
        )
        return [x1, y1, x2, y2], crop

    def _classify(self, sess: ort.InferenceSession, img: Image.Image) -> np.ndarray:
        """Run EfficientNet ONNX session; return softmax probabilities."""
        inp = self._preprocess_efficientnet(img)
        logits = sess.run(None, {"image": inp})[0][0]
        return self._softmax(logits)

    def _get_dominant_color(self, img_np: np.ndarray) -> str:
        """K-means (k=1) dominant colour, mapped to a colour name."""
        pixels = img_np.reshape(-1, 3).astype(np.float32)
        idx = np.random.choice(len(pixels), min(1000, len(pixels)), replace=False)
        sample = pixels[idx]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(sample, 1, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        r, g, b = centers[0].astype(int)

        color_map = {
            "black":  (r < 60  and g < 60  and b < 60),
            "white":  (r > 200 and g > 200 and b > 200),
            "red":    (r > 150 and g < 80  and b < 80),
            "blue":   (b > 150 and r < 80  and g < 80),
            "green":  (g > 150 and r < 80  and b < 80),
            "yellow": (r > 200 and g > 200 and b < 80),
            "orange": (r > 200 and g > 100 and b < 50),
            "pink":   (r > 200 and b > 150 and g < 150),
            "gray":   (abs(int(r) - int(g)) < 30 and abs(int(g) - int(b)) < 30),
        }
        for name, condition in color_map.items():
            if condition:
                return name
        return "multi-color"

    # ------------------------------------------------------------------
    # Public predict method
    # ------------------------------------------------------------------

    def predict(self, img: Image.Image) -> dict:
        """
        Run the full Brand Vision pipeline on a PIL image.

        Returns a dict with brand, category, confidence scores,
        attributes, bounding box, and top-3 alternatives.
        """
        img_np = np.array(img.convert("RGB"))

        # Step 1: Logo detection
        bbox, logo_crop = self._detect_logo(img)

        # Step 2: Category classification (full image)
        cat_probs = self._classify(self.category_sess, img)
        cat_idx = int(cat_probs.argmax())

        # Step 3: Brand classification (logo crop or full image)
        brand_probs = self._classify(self.brand_sess, logo_crop)
        brand_idx = int(brand_probs.argmax())

        # Step 4: Dominant colour attribute
        dominant_color = self._get_dominant_color(img_np)

        def top3(probs: np.ndarray, labels: list[str], key: str) -> list[dict]:
            return [
                {key: labels[i], "confidence": float(round(probs[i], 4))}
                for i in probs.argsort()[-3:][::-1]
            ]

        return {
            "brand": self.brand_labels[brand_idx],
            "brand_confidence": float(round(brand_probs[brand_idx], 4)),
            "category": self.category_labels[cat_idx],
            "category_confidence": float(round(cat_probs[cat_idx], 4)),
            "attributes": {"dominant_color": dominant_color},
            "logo_detected": bbox is not None,
            "bbox": bbox,
            "top3_brands": top3(brand_probs, self.brand_labels, "brand"),
            "top3_categories": top3(cat_probs, self.category_labels, "category"),
        }


# ------------------------------------------------------------------
# Singleton accessor — models loaded once at startup
# ------------------------------------------------------------------

_pipeline: BrandVisionPipeline | None = None


def get_pipeline() -> BrandVisionPipeline:
    """Return the shared pipeline instance, initialising it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = BrandVisionPipeline()
    return _pipeline
