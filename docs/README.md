# 🔍 Descry Recognition & Visual Intelligence Framework
### Deep Learning-Based E-Commerce Product Classification System

---

## Project Overview

This system automatically identifies product brands (Nike, Adidas, Apple, Samsung, etc.) and classifies products into e-commerce categories (shoes, t-shirts, smartphones, watches, etc.) using computer vision and deep learning. Users or sellers upload product images, and the system performs:

- **Brand / Logo Recognition** — YOLOv8-nano detects and localises the brand logo region
- **Product Category Classification** — EfficientNet-B0 classifies the product type
- **Attribute Tagging** *(optional Phase 3)* — colour, pattern, material via lightweight classifiers
- **Confidence Scoring** — every prediction ships with a probability score

The output is a structured JSON payload rendered in a polished React frontend — helping marketplaces automate catalog tagging, reduce human errors, and improve search relevance.

---

## Key Constraints & Environment

| Constraint | Detail |
|---|---|
| Developer Hardware | ASUS TUF F16, Intel Core 5 210H, RTX 4050 6GB (140W Turbo) |
| VRAM Budget | 6 GB — all models must train/infer within this limit |
| Budget | $0.00 — every tool, service, and platform must be free |
| DL Experience Level | Beginner — no prior model building or deep learning experience |
| Prior Experience | API-powered AI websites (React/JS frontend, REST backends) |
| Training Compute | Local RTX 4050 + Kaggle Notebooks (30 hr/week free) + Google Colab (free T4) |

---

## Architecture Summary

```
User (Browser)
    │
    ▼
React Frontend  ──────────────────────────────────────
    │   (Vercel, free)
    │   Drag-drop upload → displays results card
    │
    ▼ REST API call (multipart/form-data)
FastAPI Backend  ─────────────────────────────────────
    │   (Hugging Face Spaces, free)
    │
    ├──▶ YOLOv8-nano          → detects logo bounding box, crops region
    │
    ├──▶ EfficientNet-B0      → classifies product category
    │        (fine-tuned)
    │
    └──▶ Brand Classifier     → identifies brand from cropped logo
             (fine-tuned EfficientNet-B0 head)
    │
    ▼
JSON Response  ───────────────────────────────────────
    {
      "brand": "Nike",
      "brand_confidence": 0.94,
      "category": "Running Shoes",
      "category_confidence": 0.91,
      "attributes": { "color": "black", "pattern": "solid" },
      "bbox": [x1, y1, x2, y2]
    }
```

---

## Repository Structure

```
descry/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── ARCHITECTURE.md
├── TRAINING_GUIDE.md
├── DEPLOYMENT_GUIDE.md
│
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── routes/
│   │   └── predict.py           # /predict endpoint
│   ├── models/
│   │   ├── detector.py          # YOLOv8 logo detector wrapper
│   │   ├── classifier.py        # EfficientNet classifiers wrapper
│   │   └── pipeline.py          # Orchestrates full prediction pipeline
│   ├── utils/
│   │   ├── image_utils.py       # Preprocessing helpers
│   │   └── response_builder.py  # Formats API response
│   ├── weights/                 # Model weight files (.pt / .onnx)
│   ├── requirements.txt
│   └── Dockerfile               # For Hugging Face Spaces
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Uploader.jsx     # Drag-and-drop image upload
│   │   │   ├── ResultCard.jsx   # Displays prediction results
│   │   │   ├── ConfidenceBar.jsx
│   │   │   └── BBoxOverlay.jsx  # Draws bounding box on image preview
│   │   ├── api/
│   │   │   └── predict.js       # API call to FastAPI backend
│   │   └── styles/
│   │       └── index.css
│   ├── package.json
│   └── vite.config.js
│
└── training/
    ├── 01_data_preparation.ipynb
    ├── 02_train_yolo_logo.ipynb
    ├── 03_train_efficientnet_category.ipynb
    ├── 04_train_efficientnet_brand.ipynb
    ├── 05_export_onnx.ipynb
    └── datasets/
        └── README.md            # Dataset download instructions
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Object Detection | YOLOv8-nano (Ultralytics) | Fastest YOLO, 3.2M params, one-line training API |
| Classification | EfficientNet-B0 (torchvision) | Best accuracy/size ratio for beginners, ~5.3M params |
| Backend Framework | FastAPI (Python) | Async, auto Swagger docs, easy file upload handling |
| Frontend | React + Vite + Tailwind CSS | Familiar to developer, fast build |
| Backend Hosting | Hugging Face Spaces | Free CPU inference (2 cores, 16GB RAM) |
| Frontend Hosting | Vercel | Free, auto-deploy from Git |
| Model Storage | Hugging Face Hub | Free 10GB per repo |
| Training (cloud) | Kaggle Notebooks | Free P100 GPU, 30hr/week |
| Training (local) | PyTorch + CUDA | RTX 4050 handles both models |

---

## Deliverables Checklist

- [ ] YOLOv8-nano trained on brand logo detection (LogoDet-3K dataset)
- [ ] EfficientNet-B0 fine-tuned for product category classification
- [ ] EfficientNet-B0 fine-tuned for brand classification (from cropped logos)
- [ ] FastAPI backend serving unified `/predict` endpoint
- [ ] React frontend with drag-drop upload and results card
- [ ] ONNX export of both models for fast CPU inference on HF Spaces
- [ ] Deployed and publicly accessible URL
- [ ] Hugging Face Hub model cards for both weights

---

## Quick Start (Local Dev)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/descry.git
cd descry

# 2. Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

*For detailed phase-by-phase instructions, see [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)*
*For model architecture details, see [ARCHITECTURE.md](./ARCHITECTURE.md)*
*For training notebooks and dataset setup, see [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)*
*For Hugging Face + Vercel deployment, see [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)*
