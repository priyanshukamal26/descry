# Descry: Project Overview

**Descry** (formerly Brand Vision) is an advanced Artificial Intelligence system designed to analyze images and identify both the **Product Category** (e.g., shoes, jackets, bags) and the **Brand Identity** (e.g., Nike, Apple, Louis Vuitton). 

It is built using state-of-the-art computer vision models deployed via a high-performance backend, wrapped in a premium, glassmorphic user interface.

---

## 1. Core Architecture & Tech Stack

### 🧠 The AI Engine (Deep Learning)
The system relies on a three-stage neural network pipeline exported to the **ONNX** format for ultra-fast CPU inference:
1. **Logo Detector (YOLOv8):** A real-time object detection model trained on **158,600+ images** from the *LogoDet-3K* dataset. Its sole job is to locate and draw bounding boxes around any logos within an image.
2. **Brand Classifier (EfficientNet-B0):** A lightweight convolutional neural network trained on **167,000+ images spanning 2,341 classes** from the *Logo-2K+* dataset. It looks at the logo (or the whole image) and determines the exact brand.
3. **Product Category Classifier (EfficientNet-B0):** A secondary classifier trained on a massive Fashion Dataset to categorize the physical item itself into general categories (e.g., Backpacks, Shirts, Watches).

### ⚙️ The Backend (FastAPI + Python)
- **Framework:** FastAPI provides a lightning-fast, asynchronous REST API.
- **Inference:** Uses `onnxruntime` to run the heavy deep learning models without requiring a dedicated GPU on the server, ensuring cost-effective deployment.
- **Processing:** Handles image resizing, normalization, and aggregation of confidence scores from all three models.

### 💻 The Frontend (React + Vite + Tailwind CSS)
- **Design Language:** Implements a cutting-edge "brutal-glass" aesthetic with dynamic micro-animations.
- **Responsiveness:** Instantly renders the backend's diagnostic JSON payload into a visual `ResultCard` showing confidence probabilities and bounding box metadata.

---

## 2. The Application Workflow

When a user interacts with the Descry platform, the following pipeline executes seamlessly within milliseconds:

1. **Upload Phase:**
   The user drags and drops an image into the React frontend UI. The image is previewed instantly and converted into a secure payload.
2. **Transmission:**
   The frontend asynchronously posts the image to the FastAPI backend (`/api/analyze`).
3. **Detection (Stage 1):**
   The backend feeds the image into the **YOLOv8 Detector**. The model scans the pixels and returns bounding box coordinates if a logo is detected.
4. **Classification (Stage 2):**
   - *Brand Identification:* If YOLO found a logo, the backend crops that exact region and feeds it into the **Brand Classifier** for highly accurate brand recognition. If no logo is found, the entire image is analyzed.
   - *Category Identification:* Simultaneously, the entire uncropped image is fed into the **Product Category Classifier**.
5. **Synthesis:**
   The backend aggregates the probabilities, filters for the highest confidence scores, and constructs a structured JSON response containing the Brand, Category, Top-3 Alternative Predictions, and YOLO metadata.
6. **Rendering:**
   The React frontend reads the JSON and triggers the `ResultCard` component. The UI populates dynamically, showing the user the exact Identity, Composition, and Confidence scores of their uploaded item.

---

## 3. Dataset Scale & Training Strategy

To achieve production-grade accuracy, Descry utilizes massive, real-world datasets:
- **LogoDet-3K** (YOLO bounding boxes)
- **Logo-2K+** (2,341 unique brand classifications)
- **Fashion Product Dataset** (Hundreds of item categories)

Because processing over 300,000 images is computationally expensive, the project utilizes a **cloud-training methodology**:
1. A local Python script (`01_data_preparation.ipynb`) organizes the hundreds of gigabytes of raw data into clean class folders.
2. The data is pushed to Google Drive.
3. **Google Colab (T4 / L4 GPUs)** is used to train the EfficientNet and YOLO models for hours, leveraging free cloud compute.
4. The trained weights are exported to `.onnx` and downloaded back to the local `backend/weights/` folder for immediate inference.
