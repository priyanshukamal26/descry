# Model and Version Records

This document maintains a record of the models and software versions used in the Descry (formerly Brand Vision) project.

## Current Models

### 1. Logo Detector (YOLOv8)
- **Model Path:** `backend/weights/logo_detector.onnx`
- **Architecture:** YOLOv8 Nano (`yolov8n`)
- **Input Size:** 640x640 (Dynamic Batch)
- **Current Dataset:** LogoDet-3K
- **Output:** Bounding boxes for "logo" class.
- **Export Format:** ONNX (Opset 17)

### 2. Product Category Classifier (EfficientNet-B0)
- **Model Path:** `backend/weights/category_classifier.onnx`
- **Architecture:** EfficientNet-B0 (Custom Head)
- **Input Size:** 224x224 (Dynamic Batch)
- **Current Dataset:** Product-10K
- **Classes:** 13 categories (backpack, belt, cap, dress, handbag, jacket, jeans, shoes, sunglasses, t-shirt, wallet, watch)
- **Export Format:** ONNX (Opset 17)

### 3. Brand Classifier (EfficientNet-B0)
- **Model Path:** `backend/weights/brand_classifier.onnx`
- **Architecture:** EfficientNet-B0 (Custom Head)
- **Input Size:** 224x224 (Dynamic Batch)
- **Current Dataset:** LogoDet-3K (Cropped logos)
- **Classes:** 16 brands (ASUS, Apple, Armani, Bose, Canon, Casio, Converse, Lacoste, LouisVuitton, Oakley, RalphLauren, TheNorthFace, Timberland, TommyHilfiger, UnderArmour)
- **Export Format:** ONNX (Opset 17)

## Software Stack & Versions

- **Deep Learning Framework:** `ultralytics>=8.1.0` (YOLOv8), `torch`, `torchvision` (installed via CUDA wheels)
- **Inference & Export:** `onnx>=1.15.0`, `onnxruntime>=1.17.0`
- **Backend:** `FastAPI>=0.109.0`, `uvicorn[standard]>=0.27.0`
- **Frontend:** `React` (with Vite)
- **Utilities:** `Pillow>=10.2.0`, `opencv-python-headless>=4.9.0.80`, `numpy>=1.26.0`, `huggingface_hub>=0.20.0`

## Planned Changes
- **Dataset Upgrade:** Replacing `LogoDet-3K` with `Logo2K+` for improved brand recognition and detection accuracy.
