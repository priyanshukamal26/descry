"""
EfficientNet-B0 ONNX Export Script.

Usage (run from brand-vision/ root with venv activated):
    python training/export_onnx.py

Prerequisites:
    - training/runs/category/best_model.pt  (from Phase 3 category training)
    - training/runs/brand/best_model.pt     (from Phase 3 brand training)
    - training/runs/logo_detector/v1/weights/best.pt  (from Phase 2 YOLO training)

Outputs written to backend/weights/:
    category_classifier.onnx
    category_classifier_labels.json
    brand_classifier.onnx
    brand_classifier_labels.json
    logo_detector.onnx  (exported via ultralytics YOLO API)
"""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


ROOT = Path(__file__).parent.parent
RUNS_DIR = ROOT / "training" / "runs"
WEIGHTS_DIR = ROOT / "backend" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def build_efficientnet(num_classes: int) -> nn.Module:
    """Rebuild EfficientNet-B0 with the custom classifier head."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def export_efficientnet(checkpoint_path: Path, output_onnx: Path) -> None:
    """Load a .pt checkpoint and export it to ONNX format."""
    print(f"\nExporting: {checkpoint_path.name} → {output_onnx.name}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    classes: list[str] = checkpoint["classes"]
    num_classes = len(classes)

    model = build_efficientnet(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(output_onnx),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ✓ ONNX saved: {output_onnx}")

    # Save class labels next to the ONNX file
    labels_path = output_onnx.with_name(output_onnx.stem + "_labels.json")
    with open(labels_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"  ✓ Labels saved: {labels_path} ({num_classes} classes)")


def export_yolo(pt_path: Path, output_onnx: Path) -> None:
    """Export YOLOv8 .pt weights to ONNX via the Ultralytics API."""
    print(f"\nExporting YOLOv8: {pt_path.name} → {output_onnx.name}")
    from ultralytics import YOLO  # imported here so torch is only needed locally

    model = YOLO(str(pt_path))
    model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)

    # Ultralytics saves it next to the .pt file
    auto_output = pt_path.with_suffix(".onnx")
    if auto_output.exists():
        shutil.copy(auto_output, output_onnx)
        print(f"  ✓ Copied to: {output_onnx}")
    else:
        print(f"  ⚠ Expected ONNX at {auto_output} — not found. Check ultralytics output.")


def verify_onnx(onnx_path: Path) -> None:
    """Quick sanity-check: load model and run a dummy forward pass."""
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape

    # Build a dummy input matching the declared shape (replace dynamic dims with 1)
    shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp_shape]
    dummy = np.random.randn(*shape).astype(np.float32)

    out = sess.run(None, {inp_name: dummy})
    print(f"  ✓ Verified {onnx_path.name}  output shape: {out[0].shape}")


if __name__ == "__main__":
    # --- EfficientNet: Category ---
    export_efficientnet(
        checkpoint_path=RUNS_DIR / "category" / "best_model.pt",
        output_onnx=WEIGHTS_DIR / "category_classifier.onnx",
    )
    verify_onnx(WEIGHTS_DIR / "category_classifier.onnx")

    # --- EfficientNet: Brand ---
    export_efficientnet(
        checkpoint_path=RUNS_DIR / "brand" / "best_model.pt",
        output_onnx=WEIGHTS_DIR / "brand_classifier.onnx",
    )
    verify_onnx(WEIGHTS_DIR / "brand_classifier.onnx")

    # --- YOLOv8 ---
    export_yolo(
        pt_path=RUNS_DIR / "logo_detector" / "v1" / "weights" / "best.pt",
        output_onnx=WEIGHTS_DIR / "logo_detector.onnx",
    )
    verify_onnx(WEIGHTS_DIR / "logo_detector.onnx")

    print("\n✅ All ONNX exports complete. Files written to backend/weights/")
    print("Next: upload to HF Hub — see TRAINING_GUIDE.md Part 3")
