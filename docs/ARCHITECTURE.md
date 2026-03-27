# 🏗️ System Architecture
## Brand Recognition & Visual Intelligence Framework

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / BROWSER                               │
│              React SPA (Vercel) — drag-drop upload                  │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  POST /predict
                          │  multipart/form-data (image file)
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                                   │
│              (Hugging Face Spaces — free CPU)                        │
│                                                                      │
│   ┌──────────────┐   ┌─────────────────────┐   ┌────────────────┐  │
│   │  Uploader    │   │  Brand Vision        │   │  Response      │  │
│   │  Validator   │──▶│  Pipeline            │──▶│  Builder       │  │
│   │  (10MB limit)│   │  (pipeline.py)       │   │  (JSON)        │  │
│   └──────────────┘   └──────────┬──────────┘   └────────────────┘  │
│                                  │                                   │
│              ┌───────────────────┼───────────────────┐              │
│              ▼                   ▼                   ▼              │
│   ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐    │
│   │  YOLOv8-nano    │  │ EfficientNet-  │  │ EfficientNet-    │    │
│   │  (ONNX)         │  │ B0 Category    │  │ B0 Brand         │    │
│   │                 │  │ (ONNX)         │  │ (ONNX)           │    │
│   │  Input: full    │  │                │  │                  │    │
│   │  image 640×640  │  │ Input: full    │  │ Input: cropped   │    │
│   │                 │  │ image 224×224  │  │ logo 224×224     │    │
│   │  Output: bbox   │  │                │  │                  │    │
│   │  coords [logo]  │  │ Output: 20     │  │ Output: 50       │    │
│   │                 │  │ category probs │  │ brand probs      │    │
│   └────────┬────────┘  └───────┬────────┘  └────────┬─────────┘    │
│            │                   │                    │               │
│            └───────────────────▼────────────────────┘              │
│                    Unified JSON Response                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Architecture Details

### 2.1 YOLOv8-nano — Logo Detector

| Property | Value |
|---|---|
| Architecture | CSPDarknet + PANet + Decoupled Head |
| Parameters | ~3.2M |
| Input Size | 640×640 (RGB) |
| Output | Bounding boxes (x1,y1,x2,y2) + confidence |
| Training Classes | 1 (`logo`) — binary: logo present or not |
| VRAM Required (training) | ~2.5GB @ batch=16 |
| VRAM Required (inference) | < 500MB |
| Inference Speed (local) | ~15ms/image on RTX 4050 |
| Inference Speed (CPU) | ~180ms/image on HF Spaces |

**Why YOLOv8-nano?**
- Smallest YOLO variant — 3.2M params vs 25M for YOLOv8m
- One-line Ultralytics training API — minimal boilerplate for beginners
- COCO pretrained weights available for download
- Exports cleanly to ONNX with `model.export(format="onnx")`

**Anchor-Free Design:** YOLOv8 is anchor-free (unlike YOLOv5). It directly predicts box centre/size offsets — no need to define anchor boxes manually.

---

### 2.2 EfficientNet-B0 — Product Category Classifier

| Property | Value |
|---|---|
| Architecture | Inverted Residual + Depthwise Separable Convolutions |
| Parameters | ~5.3M (backbone) + small custom head |
| Input Size | 224×224 (RGB, ImageNet normalised) |
| Output | Softmax over 20 product categories |
| Pretrained Weights | ImageNet-1K (torchvision) |
| VRAM Required (training) | ~1.8GB @ batch=32 with AMP |
| VRAM Required (inference) | < 300MB |
| Inference Speed (local) | ~8ms/image |
| Inference Speed (CPU ONNX) | ~90ms/image |

**Custom Classifier Head:**
```
backbone (EfficientNet-B0 features)
    ↓
AdaptiveAvgPool2d → [1, 1280, 1, 1]
    ↓ flatten
[1, 1280]
    ↓ Dropout(0.3)
Linear(1280 → 20)
    ↓
Softmax (during inference) / CrossEntropy (during training)
```

**Training Strategy:**
1. **Phase 1 (5 epochs):** Freeze backbone. Train only the new classifier head with `lr=1e-3`
2. **Phase 2 (25 epochs):** Unfreeze all layers. Fine-tune end-to-end with `lr=1e-4`

This two-phase approach prevents the randomly-initialized head from destroying pretrained features.

---

### 2.3 EfficientNet-B0 — Brand Classifier

Identical architecture to the category classifier but:
- Output: Softmax over 50 brand classes
- Input: **Cropped logo region** extracted by YOLOv8 (not the full image)
- Falls back to full image if YOLOv8 detects no logo

---

## 3. Data Flow — Step by Step

```
Input: product.jpg (user upload)
    │
    ▼
[1] Validation
    • File type check (must be image/*)
    • Size check (< 10MB)
    • PIL.Image.open() — validate readable
    │
    ▼
[2] YOLOv8-nano Inference
    • Resize to 640×640 (letterbox padding)
    • Forward pass → raw detection tensors
    • NMS (Non-Max Suppression) → filtered boxes
    • Output: List of (x1, y1, x2, y2, conf, class_id)
    • Take highest-confidence box
    │
    ├── If box found → crop logo region (+ 10% padding)
    └── If no box   → use full image as fallback
    │
    ▼
[3] EfficientNet-B0 Category (parallel with step 4)
    • Input: full image, resized to 224×224
    • Normalise: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    • ONNX forward pass → logits [1, 20]
    • Softmax → probabilities
    • argmax → predicted class index → class name
    │
    ▼
[4] EfficientNet-B0 Brand
    • Input: logo crop (or full image), resized to 224×224
    • Same preprocessing as above
    • ONNX forward pass → logits [1, 50]
    • Softmax → probabilities
    • argmax → predicted brand index → brand name
    │
    ▼
[5] Attribute Extraction
    • Dominant colour: sample pixels → K-means (k=1) → map RGB to colour name
    │
    ▼
[6] Response Assembly
    {
      "brand": str,
      "brand_confidence": float,
      "category": str,
      "category_confidence": float,
      "attributes": { "dominant_color": str },
      "logo_detected": bool,
      "bbox": [x1, y1, x2, y2] | null,
      "top3_brands": [...],
      "top3_categories": [...]
    }
```

---

## 4. API Specification

### `POST /predict`

**Request:**
```
Content-Type: multipart/form-data
Body: file (image file, max 10MB)
```

**Response 200:**
```json
{
  "brand": "Nike",
  "brand_confidence": 0.9421,
  "category": "shoes",
  "category_confidence": 0.8873,
  "attributes": {
    "dominant_color": "black"
  },
  "logo_detected": true,
  "bbox": [142, 88, 310, 210],
  "top3_brands": [
    { "brand": "Nike",       "confidence": 0.9421 },
    { "brand": "Adidas",     "confidence": 0.0321 },
    { "brand": "Converse",   "confidence": 0.0142 }
  ],
  "top3_categories": [
    { "category": "shoes",   "confidence": 0.8873 },
    { "category": "boots",   "confidence": 0.0612 },
    { "category": "sneakers","confidence": 0.0512 }
  ]
}
```

**Response 400 — Invalid file:**
```json
{ "detail": "File must be an image (JPEG, PNG, WebP)" }
```

**Response 500 — Inference error:**
```json
{ "detail": "Prediction failed: <error message>" }
```

---

## 5. Training Architecture

### 5.1 Loss Functions

| Model | Loss Function | Why |
|---|---|---|
| YOLOv8 | DFL + BCE (built-in) | Distribution Focal Loss for box regression; BCE for objectness |
| EfficientNet Category | CrossEntropyLoss (label_smoothing=0.1) | Label smoothing prevents overconfident predictions |
| EfficientNet Brand | CrossEntropyLoss (label_smoothing=0.1) | Same reasoning |

### 5.2 Optimizers & Schedulers

| Model | Optimizer | LR Schedule |
|---|---|---|
| YOLOv8 | AdamW (built-in) | Cosine decay (built-in) |
| EfficientNet (head phase) | AdamW, lr=1e-3 | Fixed |
| EfficientNet (fine-tune phase) | AdamW, lr=1e-4 | CosineAnnealingLR |

### 5.3 Data Augmentation

**YOLOv8 (built-in mosaic augmentation):**
- Mosaic (4 images combined) — enabled by default
- Random horizontal flip
- HSV augmentation (hue, saturation, value)
- Scale jitter

**EfficientNet (torchvision transforms):**
```python
# Training
transforms.RandomResizedCrop(224)       # Random crop with scale variation
transforms.RandomHorizontalFlip()        # 50% chance
transforms.ColorJitter(0.2, 0.2, 0.2)  # Colour variation
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Validation / Inference
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```

---

## 6. Deployment Architecture

```
GitHub Repository
    │
    ├── /frontend  ──── Vercel (auto-deploy on push to main)
    │                   URL: https://descry-app.vercel.app
    │
    └── /backend   ──── Hugging Face Spaces
                        Space type: Docker
                        Hardware: CPU Basic (2 vCPU, 16GB RAM, free)
                        URL: https://huggingface.co/spaces/USERNAME/descry-api

Model Weights:
    HuggingFace Hub → https://huggingface.co/USERNAME/descry-weights
    - logo_detector.onnx        (~6MB)
    - category_classifier.onnx  (~16MB)
    - brand_classifier.onnx     (~16MB)
    - *_labels.json             (tiny)
```

### 6.1 Hugging Face Spaces `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download model weights from HuggingFace Hub at startup
RUN python -c "from huggingface_hub import hf_hub_download; print('HF Hub ready')"

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### 6.2 Weight Loading in Production

In `backend/models/pipeline.py`, add this to `__init__`:

```python
from huggingface_hub import hf_hub_download

REPO_ID = "YOUR_USERNAME/descry-weights"   # HuggingFace Hub repo

def _download_weights(self):
    """Downloads ONNX weights from HF Hub if not present locally."""
    files = [
        "logo_detector.onnx",
        "category_classifier.onnx",
        "category_classifier_labels.json",
        "brand_classifier.onnx",
        "brand_classifier_labels.json",
    ]
    for filename in files:
        dest = WEIGHTS_DIR / filename
        if not dest.exists():
            print(f"Downloading {filename} from HuggingFace Hub...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=str(WEIGHTS_DIR))
```

---

## 7. VRAM Budget (Local Development)

| Task | VRAM Used | Notes |
|---|---|---|
| YOLOv8-nano training (batch=16) | ~2.5 GB | With AMP enabled |
| EfficientNet-B0 training (batch=32) | ~1.8 GB | With AMP enabled |
| Both models inference | ~0.8 GB | Loaded simultaneously |
| Full pipeline (local) | ~1.2 GB | Comfortable within 6GB |
| Browser + OS | ~1.0 GB | Background usage |
| **Total during training** | **~3.5 GB** | Well within 6GB budget |

> **Note:** Never train both models simultaneously. Train one at a time.

---

## 8. Known Limitations & Mitigations

| Limitation | Mitigation |
|---|---|
| Low brand confidence on products with small/obscured logos | Return "Unknown" if brand_confidence < 0.40; show top3 alternatives |
| CPU inference on HF Spaces is slow (~2–3s) | Export to ONNX (5x faster than PyTorch CPU); add loading spinner in UI |
| 50-brand scope misses many real-world brands | Clearly state supported brands in UI; show "Not in database" message |
| Logo detector may fail on text-only logos | Fall back to classifying full image when no bbox detected |
| Small training dataset per class (LogoDet-3K) | Use heavy data augmentation; pretrained ImageNet weights mitigate this |
