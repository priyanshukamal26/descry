# 📋 Implementation Plan
## Brand Recognition & Visual Intelligence — EfficientNet + YOLOv8 Approach

> **Agent Instructions:** Execute each phase sequentially. Do not skip phases. Each phase has clear inputs, outputs, and acceptance criteria. All tooling is free. Hardware target is RTX 4050 6GB.

---

## Phase Overview

| Phase | Name | Duration (Est.) | Output |
|---|---|---|---|
| **0** | Environment Setup | 1–2 hours | Working local dev environment |
| **1** | Dataset Preparation | 3–5 hours | Clean, split datasets ready for training |
| **2** | Model Training — Logo Detector | 2–4 hours | YOLOv8-nano `.pt` weights |
| **3** | Model Training — Classifiers | 4–6 hours | Two EfficientNet-B0 `.pt` weights |
| **4** | Model Export & Optimization | 1 hour | ONNX files for both models |
| **5** | Backend Development | 4–6 hours | FastAPI app passing all tests |
| **6** | Frontend Development | 5–8 hours | React app fully functional locally |
| **7** | Integration Testing | 2–3 hours | End-to-end pipeline working |
| **8** | Deployment | 2–3 hours | Live public URL |

---

## Phase 0 — Environment Setup

### 0.1 Install Python Environment

```bash
# Python 3.10+ required
python --version   # Should be 3.10+

# Create virtual environment (project root)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 0.2 Install Backend Dependencies

Create `backend/requirements.txt`:

```txt
# Core ML
torch==2.2.0
torchvision==0.17.0
ultralytics==8.1.0          # YOLOv8

# ONNX export & runtime
onnx==1.15.0
onnxruntime==1.17.0

# API
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.9     # For file uploads

# Image processing
Pillow==10.2.0
opencv-python-headless==4.9.0.80
numpy==1.26.0

# Utilities
python-dotenv==1.0.0
pydantic==2.5.0
```

Install:
```bash
pip install -r backend/requirements.txt
```

### 0.3 Verify CUDA

```python
# Run this in Python to verify GPU is accessible
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
# Expected: CUDA available: True | GPU: NVIDIA GeForce RTX 4050 | VRAM: ~6.1 GB
```

### 0.4 Frontend Setup

```bash
cd frontend
npm create vite@latest . -- --template react
npm install
npm install tailwindcss @tailwindcss/vite
npm install axios react-dropzone
```

### 0.5 Version Control

```bash
git init
echo "venv/\n*.pt\n*.onnx\n__pycache__/\nnode_modules/\ndist/\n.env" > .gitignore
git add .
git commit -m "chore: initial project scaffold"
```

**✅ Phase 0 Done when:** `uvicorn main:app --reload` starts without errors and `npm run dev` opens in browser.

---

## Phase 1 — Dataset Preparation

### 1.1 Datasets to Download

#### Dataset A — LogoDet-3K (Logo Detection + Brand Classification)
- **URL:** https://github.com/Wangjing1551/LogoDet-3K-Dataset
- **Kaggle mirror:** Search "LogoDet-3K" on kaggle.com/datasets
- **Size:** ~2.8 GB
- **Content:** 158,652 images, 3,000 brand categories, bounding box annotations in YOLO format
- **Used for:** Training YOLOv8 logo detector + brand classifier

```bash
# If using Kaggle CLI
kaggle datasets download -d fantacher/logodet3k
unzip logodet3k.zip -d training/datasets/logodet3k/
```

#### Dataset B — Product-10K (Product Category Classification)
- **URL:** https://www.kaggle.com/datasets/xlteam/product10k
- **Size:** ~3.5 GB
- **Content:** 141,931 product images across 10,000 product categories
- **Used for:** Training EfficientNet product category classifier

```bash
kaggle datasets download -d xlteam/product10k
unzip product10k.zip -d training/datasets/product10k/
```

#### Dataset C — DeepFashion (Optional, for fashion-specific attributes)
- **URL:** http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
- **Use only if** the project scope includes fashion products heavily
- **Skip for MVP** — Product-10K covers clothing categories adequately

### 1.2 Data Preprocessing Script

Create `training/scripts/prepare_data.py`:

```python
"""
Data preparation script.
Splits datasets into train/val/test (70/20/10).
Outputs: YOLO-format annotations for detection, folder-per-class for classification.
"""

import os
import shutil
import random
from pathlib import Path

def split_classification_dataset(source_dir, output_dir, splits=(0.7, 0.2, 0.1)):
    """
    Expects source_dir with subfolders = class names.
    Outputs output_dir/train/, output_dir/val/, output_dir/test/
    """
    source = Path(source_dir)
    output = Path(output_dir)
    
    for class_dir in source.iterdir():
        if not class_dir.is_dir():
            continue
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        random.shuffle(images)
        
        n = len(images)
        train_end = int(n * splits[0])
        val_end = train_end + int(n * splits[1])
        
        subsets = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
        
        for subset, files in subsets.items():
            dest = output / subset / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)
    
    print(f"Dataset split complete → {output}")

if __name__ == "__main__":
    # Product category dataset
    split_classification_dataset(
        source_dir="datasets/product10k/train",
        output_dir="datasets/product_category_split"
    )
    print("Done.")
```

### 1.3 Class Selection Strategy

> **Important for beginners:** Do NOT use all 3,000 brand classes or 10,000 product categories. Start with a focused subset.

**Recommended Product Categories (20 classes for MVP):**
```
shoes, t-shirt, jacket, dress, handbag, watch, smartphone,
laptop, headphones, sunglasses, backpack, jeans, sneakers,
boots, cap/hat, tablet, camera, bottle, wallet, belt
```

**Recommended Brands (50 classes for MVP):**
```
Nike, Adidas, Apple, Samsung, Sony, Puma, Reebok, New Balance,
Under Armour, Levi's, H&M, Zara, Gucci, Louis Vuitton, Rolex,
Ray-Ban, Canon, Nikon, JBL, Bose, Dell, HP, Lenovo, Asus,
Converse, Vans, Timberland, Tommy Hilfiger, Calvin Klein, Ralph Lauren,
Lacoste, Polo, Versace, Armani, Balenciaga, Off-White, Supreme,
The North Face, Columbia, Patagonia, Oakley, Casio, Fossil,
Huawei, Xiaomi, OnePlus, Google, Amazon, Microsoft, Intel
```

### 1.4 Dataset Directory Structure After Preparation

```
training/datasets/
├── product_category_split/
│   ├── train/
│   │   ├── shoes/         (imgs...)
│   │   ├── smartphone/    (imgs...)
│   │   └── ...
│   ├── val/
│   └── test/
│
├── brand_logo_split/      (from LogoDet-3K, cropped logo regions)
│   ├── train/
│   │   ├── Nike/          (cropped logo imgs...)
│   │   ├── Adidas/        (...)
│   │   └── ...
│   ├── val/
│   └── test/
│
└── yolo_logo_detection/   (YOLO format for bounding box training)
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/         (.txt files with bbox coords)
        └── val/
```

**✅ Phase 1 Done when:** All three dataset splits exist, class counts are balanced, no corrupted images remain.

---

## Phase 2 — Model Training: YOLOv8 Logo Detector

### 2.1 Create YOLO Dataset Config

Create `training/datasets/yolo_logo_detection/logo_data.yaml`:

```yaml
path: /absolute/path/to/training/datasets/yolo_logo_detection
train: images/train
val: images/val

nc: 1               # Only 1 class: "logo" (we don't need to classify brand here)
names: ["logo"]     # YOLOv8 just needs to FIND the logo region
```

> **Design Decision:** YOLOv8 is trained to detect "is there a logo and where?" as a single class. The BRAND IDENTITY (Nike vs Adidas) is determined by the separate EfficientNet brand classifier in Phase 3. This simplifies the YOLO training significantly.

### 2.2 Training Notebook (Kaggle/Colab)

Create `training/02_train_yolo_logo.ipynb`:

```python
# Cell 1 — Install
!pip install ultralytics -q

# Cell 2 — Verify GPU
import torch
print(torch.cuda.is_available())

# Cell 3 — Train YOLOv8-nano
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # nano = smallest, fastest, fits in 6GB

results = model.train(
    data="path/to/logo_data.yaml",
    epochs=50,               # 50 epochs is enough for logo detection
    imgsz=640,               # Standard YOLO input size
    batch=16,                # Safe for 6GB VRAM; increase to 32 on Kaggle P100
    device=0,                # GPU 0
    project="logo_detector",
    name="yolov8n_logos_v1",
    pretrained=True,         # Start from COCO pretrained weights
    patience=10,             # Early stopping
    lr0=0.01,
    lrf=0.001,
    optimizer="AdamW",
    amp=True,                # Mixed precision — saves VRAM
    cache=True,              # Cache images in RAM for speed
    workers=4,
    save=True,
    save_period=10,          # Save checkpoint every 10 epochs
)

print("Training complete!")
print(f"Best weights: {results.save_dir}/weights/best.pt")
```

```python
# Cell 4 — Validate
model_best = YOLO("logo_detector/yolov8n_logos_v1/weights/best.pt")
metrics = model_best.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

```python
# Cell 5 — Test inference on sample image
results = model_best.predict("sample_product.jpg", conf=0.25)
results[0].show()
results[0].save("prediction_test.jpg")
```

### 2.3 Training on Local RTX 4050

```bash
# Run training locally (recommended for iteration)
cd training
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='datasets/yolo_logo_detection/logo_data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    amp=True,
    project='runs/logo_detector',
    name='v1'
)
"
```

Expected training time: **45–90 minutes on RTX 4050**

### 2.4 Expected Metrics

| Metric | Target | Acceptable |
|---|---|---|
| mAP50 | > 0.75 | > 0.60 |
| mAP50-95 | > 0.50 | > 0.35 |
| Precision | > 0.80 | > 0.65 |
| Recall | > 0.75 | > 0.60 |

**✅ Phase 2 Done when:** `best.pt` saved, mAP50 > 0.60 on validation set.

---

## Phase 3 — Model Training: EfficientNet Classifiers

Two separate EfficientNet-B0 models are trained:
- **Model A:** Product category classifier (what type of product is this?)
- **Model B:** Brand classifier (what brand is this? — runs on cropped logo region)

### 3.1 Training Script Template

Create `training/train_efficientnet.py`:

```python
"""
Generic EfficientNet-B0 fine-tuner.
Usage:
  python train_efficientnet.py --data_dir datasets/product_category_split --task category
  python train_efficientnet.py --data_dir datasets/brand_logo_split --task brand
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
import time
import os

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def build_model(num_classes):
    # Load pretrained EfficientNet-B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Freeze backbone layers (train only classifier head first)
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Data loaders
    train_ds = datasets.ImageFolder(f"{args.data_dir}/train", get_transforms(train=True))
    val_ds   = datasets.ImageFolder(f"{args.data_dir}/val",   get_transforms(train=False))
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    num_classes = len(train_ds.classes)
    print(f"Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")
    
    model = build_model(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()  # Mixed precision
    
    best_acc = 0.0
    os.makedirs(f"runs/{args.task}", exist_ok=True)
    
    # Phase 1: Train head only (5 epochs)
    print("Phase 1: Training classifier head only...")
    for epoch in range(5):
        _run_epoch(model, train_dl, criterion, optimizer, scaler, device, epoch, "train")
    
    # Phase 2: Unfreeze backbone and fine-tune all layers
    print("Phase 2: Fine-tuning full network...")
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
    
    for epoch in range(args.epochs):
        train_acc = _run_epoch(model, train_dl, criterion, optimizer, scaler, device, epoch, "train")
        val_acc   = _run_epoch(model, val_dl,   criterion, None,       None,   device, epoch, "val")
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": train_ds.classes
            }, f"runs/{args.task}/best_model.pt")
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")
    
    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Weights saved to: runs/{args.task}/best_model.pt")

def _run_epoch(model, loader, criterion, optimizer, scaler, device, epoch, phase):
    model.train() if phase == "train" else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.set_grad_enabled(phase == "train"):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if phase == "train":
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    acc = correct / total
    print(f"Epoch {epoch+1} [{phase}] Loss: {total_loss/len(loader):.4f} | Acc: {acc:.4f}")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--task", required=True, choices=["category", "brand"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)  # Safe for RTX 4050 6GB
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
```

### 3.2 Run Training

```bash
# Train product category classifier
python train_efficientnet.py \
  --data_dir datasets/product_category_split \
  --task category \
  --epochs 30 \
  --batch_size 32

# Train brand classifier (on cropped logo images)
python train_efficientnet.py \
  --data_dir datasets/brand_logo_split \
  --task brand \
  --epochs 30 \
  --batch_size 32
```

Expected training time per model: **2–3 hours on RTX 4050** (30 epochs, ~50 classes)

### 3.3 Expected Metrics

| Model | Target Top-1 Acc | Acceptable | Notes |
|---|---|---|---|
| Category Classifier | > 80% | > 65% | 20 classes, clear visual differences |
| Brand Classifier | > 75% | > 60% | 50 classes, logo regions only |

**✅ Phase 3 Done when:** Both `best_model.pt` files saved with acceptable accuracy.

---

## Phase 4 — ONNX Export & Optimization

ONNX export is required because Hugging Face Spaces runs on CPU. ONNX Runtime on CPU is **5–10x faster** than PyTorch CPU inference.

### 4.1 Export EfficientNet to ONNX

Create `training/export_onnx.py`:

```python
import torch
import torch.nn as nn
from torchvision import models
import json

def export_efficientnet(checkpoint_path, output_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]
    num_classes = len(classes)
    
    # Rebuild model architecture
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Dummy input
    dummy = torch.randn(1, 3, 224, 224)
    
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True
    )
    
    # Save class labels alongside the ONNX file
    labels_path = output_path.replace(".onnx", "_labels.json")
    with open(labels_path, "w") as f:
        json.dump(classes, f)
    
    print(f"Exported: {output_path}")
    print(f"Labels:   {labels_path}")
    print(f"Classes:  {num_classes}")

export_efficientnet("runs/category/best_model.pt", "backend/weights/category_classifier.onnx")
export_efficientnet("runs/brand/best_model.pt",    "backend/weights/brand_classifier.onnx")
```

### 4.2 Export YOLOv8 to ONNX

```python
from ultralytics import YOLO

model = YOLO("runs/logo_detector/v1/weights/best.pt")
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
# Outputs: runs/logo_detector/v1/weights/best.onnx
# Copy to: backend/weights/logo_detector.onnx
```

### 4.3 Verify ONNX Models

```python
import onnxruntime as ort
import numpy as np

# Test category model
sess = ort.InferenceSession("backend/weights/category_classifier.onnx")
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = sess.run(None, {"image": dummy})
print(f"Category model output shape: {output[0].shape}")  # Should be (1, 20)

# Test brand model
sess2 = ort.InferenceSession("backend/weights/brand_classifier.onnx")
output2 = sess2.run(None, {"image": dummy})
print(f"Brand model output shape: {output2[0].shape}")  # Should be (1, 50)
```

**✅ Phase 4 Done when:** All 3 ONNX files load successfully and produce correct output shapes.

---

## Phase 5 — Backend Development (FastAPI)

### 5.1 Prediction Pipeline

Create `backend/models/pipeline.py`:

```python
import numpy as np
import onnxruntime as ort
import json
from PIL import Image
import cv2
from ultralytics import YOLO
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

class BrandVisionPipeline:
    
    def __init__(self):
        # Load YOLO detector
        self.detector = YOLO(WEIGHTS_DIR / "logo_detector.onnx", task="detect")
        
        # Load EfficientNet ONNX sessions
        self.category_sess = ort.InferenceSession(str(WEIGHTS_DIR / "category_classifier.onnx"))
        self.brand_sess    = ort.InferenceSession(str(WEIGHTS_DIR / "brand_classifier.onnx"))
        
        # Load class labels
        with open(WEIGHTS_DIR / "category_classifier_labels.json") as f:
            self.category_labels = json.load(f)
        with open(WEIGHTS_DIR / "brand_classifier_labels.json") as f:
            self.brand_labels = json.load(f)
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print("Pipeline initialized ✓")
    
    def preprocess_for_efficientnet(self, img_pil: Image.Image) -> np.ndarray:
        """Resize + normalize PIL image for EfficientNet input."""
        img = img_pil.convert("RGB").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = arr.transpose(2, 0, 1)          # HWC → CHW
        return np.expand_dims(arr, 0)          # Add batch dim → (1,3,224,224)
    
    def softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()
    
    def predict(self, img_pil: Image.Image) -> dict:
        img_np = np.array(img_pil.convert("RGB"))
        
        # --- Step 1: Detect logo bounding box ---
        yolo_results = self.detector.predict(img_np, conf=0.25, verbose=False)
        
        bbox = None
        logo_crop = img_pil  # Default: use full image if no logo found
        
        if len(yolo_results[0].boxes) > 0:
            # Take the highest-confidence detection
            best_box = yolo_results[0].boxes[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            bbox = [x1, y1, x2, y2]
            
            # Crop logo region with 10% padding
            h, w = img_np.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1c = max(0, x1 - pad_x)
            y1c = max(0, y1 - pad_y)
            x2c = min(w, x2 + pad_x)
            y2c = min(h, y2 + pad_y)
            logo_crop = img_pil.crop((x1c, y1c, x2c, y2c))
        
        # --- Step 2: Classify product category (full image) ---
        cat_input = self.preprocess_for_efficientnet(img_pil)
        cat_logits = self.category_sess.run(None, {"image": cat_input})[0][0]
        cat_probs  = self.softmax(cat_logits)
        cat_idx    = int(cat_probs.argmax())
        
        # --- Step 3: Classify brand (from cropped logo region) ---
        brand_input = self.preprocess_for_efficientnet(logo_crop)
        brand_logits = self.brand_sess.run(None, {"image": brand_input})[0][0]
        brand_probs  = self.softmax(brand_logits)
        brand_idx    = int(brand_probs.argmax())
        
        # --- Step 4: Simple colour attribute (K-means dominant colour) ---
        color_name = self._get_dominant_color(img_np)
        
        return {
            "brand":               self.brand_labels[brand_idx],
            "brand_confidence":    float(round(brand_probs[brand_idx], 4)),
            "category":            self.category_labels[cat_idx],
            "category_confidence": float(round(cat_probs[cat_idx], 4)),
            "attributes": {
                "dominant_color": color_name
            },
            "logo_detected": bbox is not None,
            "bbox": bbox,
            "top3_brands": [
                {"brand": self.brand_labels[i], "confidence": float(round(brand_probs[i], 4))}
                for i in brand_probs.argsort()[-3:][::-1]
            ],
            "top3_categories": [
                {"category": self.category_labels[i], "confidence": float(round(cat_probs[i], 4))}
                for i in cat_probs.argsort()[-3:][::-1]
            ]
        }
    
    def _get_dominant_color(self, img_np: np.ndarray) -> str:
        """Returns dominant colour name using K-means (k=1)."""
        pixels = img_np.reshape(-1, 3).astype(np.float32)
        # Sample max 1000 pixels for speed
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
            "gray":   (abs(int(r)-int(g)) < 30 and abs(int(g)-int(b)) < 30),
        }
        for color_name, condition in color_map.items():
            if condition:
                return color_name
        return "multi-color"


# Singleton — load models once at startup
_pipeline = None

def get_pipeline() -> BrandVisionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = BrandVisionPipeline()
    return _pipeline
```

### 5.2 FastAPI App

Create `backend/main.py`:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
from models.pipeline import get_pipeline

app = FastAPI(
    title="Brand Vision API",
    description="Deep learning-based brand recognition and product classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Tighten in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Pre-load models at startup to avoid cold-start latency on first request."""
    get_pipeline()

@app.get("/")
def root():
    return {"status": "ok", "message": "Brand Vision API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, WebP)")
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")
    
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Ensure it is a valid image file.")
    
    pipeline = get_pipeline()
    
    try:
        result = pipeline.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    return JSONResponse(content=result)
```

### 5.3 Backend Tests

Create `backend/test_api.py`:

```python
"""Quick smoke tests for the API."""
import requests

BASE = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    print("✓ Health check passed")

def test_predict_with_sample():
    with open("test_images/nike_shoe.jpg", "rb") as f:
        r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    data = r.json()
    assert "brand" in data
    assert "category" in data
    assert "brand_confidence" in data
    assert 0 <= data["brand_confidence"] <= 1
    print(f"✓ Prediction: brand={data['brand']} ({data['brand_confidence']:.2%}), "
          f"category={data['category']} ({data['category_confidence']:.2%})")

if __name__ == "__main__":
    test_health()
    test_predict_with_sample()
    print("\nAll tests passed ✓")
```

**✅ Phase 5 Done when:** `test_api.py` passes all assertions, `/predict` returns valid JSON with brand + category.

---

## Phase 6 — Frontend Development (React)

### 6.1 API Client

Create `frontend/src/api/predict.js`:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Prediction failed");
  }

  return response.json();
}
```

### 6.2 Core Components

**`frontend/src/components/Uploader.jsx`** — Handles drag-and-drop + click-to-upload

```jsx
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

export default function Uploader({ onImageSelect, isLoading }) {
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    onImageSelect(file);
  }, [onImageSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    multiple: false,
    maxSize: 10 * 1024 * 1024,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer
        transition-all duration-200
        ${isDragActive
          ? "border-blue-400 bg-blue-50 scale-[1.01]"
          : "border-gray-300 bg-gray-50 hover:border-blue-300 hover:bg-blue-50/40"
        }
        ${isLoading ? "pointer-events-none opacity-60" : ""}
      `}
    >
      <input {...getInputProps()} />
      {preview ? (
        <img
          src={preview}
          alt="Preview"
          className="mx-auto max-h-64 rounded-xl object-contain"
        />
      ) : (
        <div className="py-8">
          <div className="text-5xl mb-4">📦</div>
          <p className="text-lg font-medium text-gray-700">
            {isDragActive ? "Drop it here!" : "Drag & drop a product image"}
          </p>
          <p className="text-sm text-gray-400 mt-1">or click to browse · JPG, PNG, WebP · max 10MB</p>
        </div>
      )}
    </div>
  );
}
```

**`frontend/src/components/ConfidenceBar.jsx`**

```jsx
export default function ConfidenceBar({ value, label, color = "blue" }) {
  const pct = Math.round(value * 100);
  const colorMap = {
    blue: "bg-blue-500",
    green: "bg-green-500",
    orange: "bg-orange-500",
  };

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-600 font-medium">{label}</span>
        <span className="font-bold text-gray-800">{pct}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${colorMap[color]}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
```

**`frontend/src/components/ResultCard.jsx`** — Displays prediction results

```jsx
import ConfidenceBar from "./ConfidenceBar";

const CATEGORY_ICONS = {
  "shoes": "👟", "smartphone": "📱", "laptop": "💻", "watch": "⌚",
  "t-shirt": "👕", "jacket": "🧥", "dress": "👗", "handbag": "👜",
  "headphones": "🎧", "sunglasses": "🕶️", "backpack": "🎒",
};

export default function ResultCard({ result }) {
  if (!result) return null;

  const icon = CATEGORY_ICONS[result.category?.toLowerCase()] || "🏷️";

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{icon}</span>
          <div>
            <h2 className="text-white font-bold text-xl">{result.brand}</h2>
            <p className="text-slate-300 text-sm">{result.category}</p>
          </div>
          {result.logo_detected && (
            <span className="ml-auto text-xs px-2 py-1 bg-green-500/20 text-green-300 rounded-full border border-green-500/30">
              Logo detected
            </span>
          )}
        </div>
      </div>

      {/* Confidence scores */}
      <div className="px-6 py-5 space-y-4">
        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Confidence Scores
          </p>
          <div className="space-y-3">
            <ConfidenceBar
              label={`Brand: ${result.brand}`}
              value={result.brand_confidence}
              color="blue"
            />
            <ConfidenceBar
              label={`Category: ${result.category}`}
              value={result.category_confidence}
              color="green"
            />
          </div>
        </div>

        {/* Attributes */}
        {result.attributes && (
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Attributes
            </p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(result.attributes).map(([key, val]) => (
                <span
                  key={key}
                  className="text-xs px-3 py-1 bg-slate-100 text-slate-600 rounded-full"
                >
                  {key}: <strong>{val}</strong>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Top 3 alternatives */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Top Brands
            </p>
            {result.top3_brands?.map((b, i) => (
              <div key={i} className="flex justify-between text-sm py-1 border-b border-gray-50">
                <span className={i === 0 ? "font-semibold text-blue-600" : "text-gray-500"}>
                  {b.brand}
                </span>
                <span className="text-gray-400">{Math.round(b.confidence * 100)}%</span>
              </div>
            ))}
          </div>
          <div>
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Top Categories
            </p>
            {result.top3_categories?.map((c, i) => (
              <div key={i} className="flex justify-between text-sm py-1 border-b border-gray-50">
                <span className={i === 0 ? "font-semibold text-green-600" : "text-gray-500"}>
                  {c.category}
                </span>
                <span className="text-gray-400">{Math.round(c.confidence * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
```

### 6.3 Main App

Create `frontend/src/App.jsx`:

```jsx
import { useState } from "react";
import Uploader from "./components/Uploader";
import ResultCard from "./components/ResultCard";
import { predictImage } from "./api/predict";

export default function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = async (file) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-slate-800 tracking-tight">
            Brand <span className="text-blue-600">Vision</span>
          </h1>
          <p className="text-gray-500 mt-2 text-lg">
            Upload a product image to identify its brand and category
          </p>
        </div>

        {/* Upload area */}
        <div className="mb-6">
          <Uploader onImageSelect={handleImageSelect} isLoading={isLoading} />
        </div>

        {/* Loading */}
        {isLoading && (
          <div className="text-center py-8">
            <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-500 mt-3">Analysing image...</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl px-5 py-4 text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* Result */}
        {result && <ResultCard result={result} />}
      </div>
    </div>
  );
}
```

**✅ Phase 6 Done when:** UI loads, image uploads work, result card displays correctly with all fields populated.

---

## Phase 7 — Integration Testing

### 7.1 End-to-End Test Checklist

```bash
# Start backend
cd backend && uvicorn main:app --reload --port 8000

# Start frontend
cd frontend && npm run dev
```

Test the following manually or via automated tests:

- [ ] Upload a Nike shoe image → Brand: "Nike", Category: "shoes"
- [ ] Upload an Apple iPhone image → Brand: "Apple", Category: "smartphone"
- [ ] Upload an image with no visible logo → `logo_detected: false`, brand confidence < 0.5
- [ ] Upload a non-image file → API returns 400 error, UI shows error message
- [ ] Upload a >10MB image → API returns 400 error, UI shows error message
- [ ] Upload a corrupted image → API returns 400 error, UI shows error message
- [ ] Verify bounding box coordinates are valid (within image dimensions)
- [ ] Verify confidence scores are always 0.0–1.0
- [ ] Verify top3 lists have exactly 3 items each

### 7.2 Performance Benchmark

```python
# Run this to measure inference time
import time, requests

times = []
with open("test_images/nike_shoe.jpg", "rb") as f:
    for i in range(10):
        start = time.time()
        r = requests.post("http://localhost:8000/predict",
                         files={"file": ("test.jpg", f, "image/jpeg")})
        times.append(time.time() - start)
        f.seek(0)

print(f"Avg latency: {sum(times)/len(times)*1000:.0f}ms")
print(f"P95 latency: {sorted(times)[int(0.95*len(times))]*1000:.0f}ms")
# Target: <500ms locally, <3000ms on HF Spaces (CPU)
```

**✅ Phase 7 Done when:** All checklist items pass, latency < 500ms locally.

---

## Phase 8 — Deployment

*See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for complete deployment instructions.*

**✅ Phase 8 Done when:** Public URL accessible, `/predict` responds correctly with a real product image.
