# 🧠 Training Guide
## Datasets, Notebooks, and GPU Resources

---

## Overview

All training is done on **free compute resources**. Nothing costs money.

| Training Platform | GPU | Free Quota | Best Used For |
|---|---|---|---|
| **Kaggle Notebooks** | NVIDIA P100 16GB | 30 hours/week | Primary training runs |
| **Google Colab Free** | NVIDIA T4 15GB | ~12h session limit | Experiments, debugging |
| **Local — RTX 4050** | NVIDIA RTX 4050 6GB | Unlimited | Quick iterations, testing |

**Strategy:**
- Do all development and quick tests locally (RTX 4050)
- Push final training runs to Kaggle Notebooks (P100 — more VRAM, faster)
- Use Colab as backup when Kaggle quota is exhausted

---

## Part 1 — Datasets

### 1.1 Dataset Sources

#### Primary: LogoDet-3K

| Property | Value |
|---|---|
| Source | https://github.com/Wangjing1551/LogoDet-3K-Dataset |
| Kaggle Mirror | https://www.kaggle.com/datasets/fantacher/logodet3k (search on Kaggle) |
| Size | ~2.8 GB |
| Images | 158,652 |
| Brand Classes | 3,000 |
| Annotation Format | YOLO (`.txt` files with normalised bbox coords) |
| License | Free for research/academic use |

**Used for:**
- YOLOv8 logo detection (bounding box labels)

#### Primary: Logo-2K+

| Property | Value |
|---|---|
| Size | ~3.3 GB |
| Images | 167,140 |
| Brand Classes | 2,341 |
| Annotation Format | Folder Structure (RootCategory/SubClass/Image.jpg) |
| License | Free for research/academic use |

**Used for:**
- EfficientNet brand classifier (cropped logo images)

#### Secondary: Product-10K

| Property | Value |
|---|---|
| Source | https://www.kaggle.com/datasets/xlteam/product10k |
| Size | ~3.5 GB |
| Images | 141,931 |
| Product Classes | 10,000 (we use all valid classes with >100 images) |
| License | Free / Kaggle rules |

**Used for:**
- EfficientNet product category classifier

#### Kaggle CLI Setup (run once)

```bash
# Install Kaggle CLI
pip install kaggle

# Get API key from: kaggle.com → Account → API → Create New Token
# Save the downloaded kaggle.json to:
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Download datasets
kaggle datasets download -d fantacher/logodet3k -p training/datasets/
kaggle datasets download -d xlteam/product10k   -p training/datasets/

# Unzip
cd training/datasets
unzip logodet3k.zip  -d logodet3k/
unzip product10k.zip -d product10k/

# Logo-2K+ should be manually downloaded and unzipped to:
# training/datasets/logo2k/
```

---

### 1.2 Dataset Preparation Notebook

> **Run this on your local machine** (no GPU needed — it's just file operations)

`training/01_data_preparation.ipynb`:

```python
# The notebook combines LogoDet-3K (for YOLO) and Logo-2K+ (for classification)
# See the notebook for full source.
```
```

---

## Part 2 — Training Notebooks

### 2.1 YOLOv8 Logo Detector

**Recommended platform:** Kaggle Notebooks (P100 — fast training)

`training/02_train_yolo_logo.ipynb`:

```python
# Cell 1 — Setup (Kaggle: datasets already available at /kaggle/input/)
!pip install ultralytics -q
import os
os.makedirs("/kaggle/working/logo_detector", exist_ok=True)

# Cell 2 — Create dataset YAML
yaml_content = """
path: /kaggle/working/yolo_logo_detection
train: images/train
val: images/val
nc: 1
names: ['logo']
"""
with open("/kaggle/working/logo_data.yaml", "w") as f:
    f.write(yaml_content)

# Cell 3 — Upload or copy dataset to /kaggle/working/yolo_logo_detection/
# (Upload the prepared zip from local machine as a Kaggle dataset)

# Cell 4 — Train
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="/kaggle/working/logo_data.yaml",
    epochs=60,
    imgsz=640,
    batch=32,          # P100 has 16GB — can fit batch=32
    device=0,
    amp=True,
    patience=15,
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.001,
    warmup_epochs=3,
    mosaic=1.0,        # Enable mosaic augmentation
    flipud=0.0,
    fliplr=0.5,
    degrees=5.0,       # Slight rotation augmentation
    project="/kaggle/working/runs",
    name="logo_v1",
    save=True,
    save_period=10,
    cache="ram",       # Cache in RAM on Kaggle (fast)
    workers=4,
    verbose=True,
)

# Cell 5 — Validate
model_best = YOLO("/kaggle/working/runs/logo_v1/weights/best.pt")
metrics = model_best.val(data="/kaggle/working/logo_data.yaml")
print(f"mAP50:    {metrics.box.map50:.4f}  (target: >0.75)")
print(f"mAP50-95: {metrics.box.map:.4f}   (target: >0.50)")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")

# Cell 6 — Export to ONNX
model_best.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
# Output: /kaggle/working/runs/logo_v1/weights/best.onnx
print("ONNX export done ✓")
```

---

### 2.2 EfficientNet Category Classifier

`training/03_train_efficientnet_category.ipynb`:

```python
# Cell 1 — Imports & setup
import torch, torch.nn as nn, torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
import os, json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | CUDA: {torch.version.cuda}")

# Cell 2 — Transforms
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cell 3 — Datasets
DATA_DIR = "/kaggle/working/product_category_split"  # adjust path
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", train_tfm)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   val_tfm)

print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Cell 4 — Model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
for p in model.features.parameters():
    p.requires_grad = False  # Freeze backbone

in_feats = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_feats, len(train_ds.classes))
)
model = model.to(device)
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Cell 5 — Phase 1: Head warmup (5 epochs)
criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer  = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
scaler     = GradScaler()

best_acc = 0
os.makedirs("/kaggle/working/runs/category", exist_ok=True)

def run_epoch(model, loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if train:
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad(), autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
        
        total_loss += loss.item()
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    
    return total_loss / len(loader), correct / total

print("--- Phase 1: Training classifier head ---")
for epoch in range(5):
    tr_loss, tr_acc = run_epoch(model, train_dl, train=True)
    vl_loss, vl_acc = run_epoch(model, val_dl, train=False)
    print(f"Ep {epoch+1:02d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}")

# Cell 6 — Phase 2: Fine-tune full network (30 epochs)
for p in model.features.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

print("\n--- Phase 2: Fine-tuning full network ---")
for epoch in range(30):
    tr_loss, tr_acc = run_epoch(model, train_dl, train=True)
    vl_loss, vl_acc = run_epoch(model, val_dl, train=False)
    scheduler.step()
    
    print(f"Ep {epoch+1:02d} | tr_acc={tr_acc:.4f} | vl_acc={vl_acc:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")
    
    if vl_acc > best_acc:
        best_acc = vl_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": vl_acc,
            "classes": train_ds.classes,
        }, "/kaggle/working/runs/category/best_model.pt")
        print(f"  ✅ Saved best model (val_acc={vl_acc:.4f})")

print(f"\nBest validation accuracy: {best_acc:.4f}")

# Cell 7 — Export to ONNX
import torch.onnx

checkpoint = torch.load("/kaggle/working/runs/category/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

dummy = torch.randn(1, 3, 224, 224, device=device)
torch.onnx.export(
    model.cpu(), torch.randn(1, 3, 224, 224),
    "/kaggle/working/runs/category/category_classifier.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17
)

with open("/kaggle/working/runs/category/category_classifier_labels.json", "w") as f:
    json.dump(train_ds.classes, f)

print("ONNX export done ✓")
```

### 2.3 EfficientNet Brand Classifier

**Recommended platform:** Google Colab (T4 / L4 — faster iterations)

`training/04_train_efficientnet_brand.ipynb` — **identical to Category notebook**, only change:

```python
# Colab-specific Drive mount:
from google.colab import drive
drive.mount('/content/drive')

# Change these two lines:
DATA_DIR = "/content/drive/MyDrive/descry/brand_logo_split"  # ← brand data on Drive
SAVE_DIR = "/content/drive/MyDrive/descry/runs/brand"        # ← save location on Drive
```

Everything else is the same. Upload your `brand_logo_split` folder (or zip) to Google Drive before starting.

---

## Part 3 — Uploading Weights to Hugging Face Hub

```bash
# Install HuggingFace Hub CLI
pip install huggingface_hub

# Login (get token from: huggingface.co/settings/tokens)
huggingface-cli login

# Create a new model repository
huggingface-cli repo create brand-vision-weights --type model

# Upload all ONNX files and labels
python -c "
from huggingface_hub import HfApi
api = HfApi()
repo_id = 'YOUR_USERNAME/brand-vision-weights'

files_to_upload = [
    ('backend/weights/logo_detector.onnx',              'logo_detector.onnx'),
    ('backend/weights/category_classifier.onnx',        'category_classifier.onnx'),
    ('backend/weights/category_classifier_labels.json', 'category_classifier_labels.json'),
    ('backend/weights/brand_classifier.onnx',           'brand_classifier.onnx'),
    ('backend/weights/brand_classifier_labels.json',    'brand_classifier_labels.json'),
]

for local_path, remote_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type='model'
    )
    print(f'Uploaded: {remote_path}')

print('All weights uploaded ✓')
"
```

---

## Part 4 — Troubleshooting Common Training Issues

### CUDA Out of Memory
```python
# Reduce batch size
batch = 16   # Instead of 32

# Or enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or reduce image size (only for EfficientNet, not YOLO)
transforms.Resize(192)  # Instead of 224
```

### Training Loss Not Decreasing
```python
# Check 1: Learning rate too high — reduce by 10x
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Check 2: Verify data loading is correct
for imgs, labels in train_dl:
    print(imgs.shape, labels.shape, labels.unique())
    break

# Check 3: Check class balance
from collections import Counter
counts = Counter([label for _, label in train_ds.samples])
print(sorted(counts.items(), key=lambda x: x[1]))
```

### YOLO Not Finding Logos
```python
# Lower confidence threshold for prediction
results = model.predict(img, conf=0.15)   # Default is 0.25

# Check if dataset annotations are correct
from ultralytics.utils.plotting import Annotator
results = model.val(data="logo_data.yaml", plots=True)
# Check runs/val/val_batch0_pred.jpg for visual predictions
```

### Slow Kaggle Training
```python
# Enable disk cache instead of RAM cache if OOM:
cache="disk"   # instead of cache="ram"

# Reduce workers if Kaggle warns about them:
workers=2
```
