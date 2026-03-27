# 🚀 Deployment Guide
## Hugging Face Spaces (Backend) + Vercel (Frontend)

> **All steps are 100% free. No credit card required anywhere.**

---

## Architecture Summary

```
Frontend (React) ──── Vercel ──── https://descry-app.vercel.app
                                         │
                                         │ POST /predict
                                         ▼
Backend (FastAPI) ─── HF Spaces ─── https://USERNAME-descry-api.hf.space
                                         │
                                         │ loads weights from
                                         ▼
Model Weights ──── HF Hub ──── https://huggingface.co/USERNAME/descry-weights
```

---

## Part 1 — Prepare Accounts (One-Time)

### 1.1 Create Free Accounts

| Service | Sign-Up URL | Used For |
|---|---|---|
| GitHub | https://github.com | Code repository + CI/CD |
| Hugging Face | https://huggingface.co | Model weights storage + API hosting |
| Vercel | https://vercel.com | React frontend hosting |
| Kaggle | https://kaggle.com | Model training notebooks |

### 1.2 Connect GitHub to Vercel

1. Go to https://vercel.com → Sign up with GitHub
2. Click "New Project" → Import your GitHub repo → Select `descry`
3. Set Root Directory to `frontend`
4. Vercel auto-detects Vite — click Deploy

---

## Part 2 — Backend Deployment (Hugging Face Spaces)

### 2.1 Create a New Space

1. Go to https://huggingface.co → Click your profile → New Space
2. Settings:
   - **Space name:** `descry-api`
   - **SDK:** `Docker`
   - **Hardware:** `CPU Basic` (2 vCPU, 16GB RAM) — **FREE**
   - **Visibility:** Public

### 2.2 Backend File Structure for HF Spaces

HF Spaces requires these files in the root of the Space repository:

```
descry-api (HF Space repo)
├── Dockerfile
├── requirements.txt
├── main.py
├── models/
│   ├── pipeline.py
│   ├── detector.py
│   └── classifier.py
└── utils/
    ├── image_utils.py
    └── response_builder.py
```

### 2.3 Dockerfile

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create weights directory
RUN mkdir -p weights

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health')" || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

### 2.4 HF Spaces `requirements.txt`

```txt
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.9
Pillow==10.2.0
numpy==1.26.0
opencv-python-headless==4.9.0.80
onnxruntime==1.17.0
ultralytics==8.1.0
huggingface_hub==0.20.0
python-dotenv==1.0.0
pydantic==2.5.0
```

> **Note:** Do NOT include `torch` or `torchvision` in the HF Spaces requirements — we use ONNX Runtime for inference, which is much lighter and runs well on CPU.
> 
> ONNX Runtime is ~50MB vs PyTorch ~750MB. HF Spaces CPU tier works well with ONNX.

### 2.5 Update `pipeline.py` for Production (Weight Auto-Download)

Add this to the top of `backend/models/pipeline.py`:

```python
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
HF_REPO_ID  = os.getenv("HF_WEIGHTS_REPO", "YOUR_USERNAME/descry-weights")

def ensure_weights():
    """Download model weights from HF Hub if not present locally."""
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
                local_dir_use_symlinks=False
            )
            print(f"[Weights] Downloaded: {filename}")
        else:
            print(f"[Weights] Found cached: {filename}")

# Call at module import time
ensure_weights()
```

### 2.6 Update `main.py` for HF Spaces

```python
# Add this at the top of main.py for HF Spaces compatibility:
import os

# HF Spaces sets PORT env var — fall back to 7860
PORT = int(os.getenv("PORT", 7860))

# CORS: allow your Vercel frontend URL
ALLOWED_ORIGINS = [
    "https://descry-app.vercel.app",      # Production frontend
    "https://*.vercel.app",                  # Vercel preview deployments
    "http://localhost:5173",                 # Local dev
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
```

### 2.7 Push Backend to HF Spaces

```bash
# Install Git LFS (required for large files)
git lfs install

# Clone your HF Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/descry-api
cd descry-api

# Copy backend files into the Space repo
cp -r ../descry/backend/* .

# Commit and push
git add .
git commit -m "feat: initial FastAPI deployment"
git push

# HF Spaces will automatically build the Docker container and deploy
# Watch build logs at: huggingface.co/spaces/YOUR_USERNAME/descry-api
```

### 2.8 Set Environment Variables in HF Spaces

1. Go to your Space → Settings → Repository Secrets
2. Add:
   - `HF_WEIGHTS_REPO` = `YOUR_USERNAME/descry-weights`

### 2.9 Verify Backend Deployment

```bash
# Test health endpoint
curl https://YOUR_USERNAME-descry-api.hf.space/health
# Expected: {"status":"healthy"}

# Test predict endpoint
curl -X POST https://YOUR_USERNAME-descry-api.hf.space/predict \
  -F "file=@test_images/nike_shoe.jpg"
# Expected: JSON with brand, category, confidence scores
```

---

## Part 3 — Frontend Deployment (Vercel)

### 3.1 Set Environment Variable in Vercel

1. Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add:
   - **Name:** `VITE_API_URL`
   - **Value:** `https://YOUR_USERNAME-descry-api.hf.space`
   - **Environment:** Production + Preview

### 3.2 `vite.config.js`

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Proxy API calls in local dev to avoid CORS issues
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      }
    }
  }
})
```

### 3.3 `vercel.json`

Create `frontend/vercel.json`:

```json
{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### 3.4 Auto-Deploy Setup

Every push to `main` branch triggers auto-deploy on Vercel:

```bash
cd descry/frontend
git add .
git commit -m "feat: update UI"
git push origin main
# Vercel auto-deploys in ~1 minute
```

---

## Part 4 — Post-Deployment Checklist

### 4.1 Full System Test

```bash
# Test 1: Backend health
curl https://YOUR_USERNAME-descry-api.hf.space/health

# Test 2: Full prediction via CLI
curl -X POST https://YOUR_USERNAME-descry-api.hf.space/predict \
  -F "file=@nike_shoe.jpg" | python -m json.tool

# Test 3: Frontend loads
open https://descry-app.vercel.app

# Test 4: End-to-end via browser
# - Go to https://descry-app.vercel.app
# - Upload a Nike shoe image
# - Verify result card shows brand + category
```

### 4.2 Performance Expectations on Free Tier

| Metric | Local (RTX 4050) | HF Spaces CPU |
|---|---|---|
| Cold start (first request) | ~2s | ~30–60s (model download) |
| Warm inference | ~80ms | ~2–4 seconds |
| Throughput | ~12 req/s | ~0.3 req/s |

> **HF Spaces free tier sleeps after 48 hours of inactivity.** First request after sleep triggers cold start. This is expected behaviour — acceptable for a demo/project.

### 4.3 Optional: Add Loading UX for Slow API

Update `frontend/src/App.jsx` to show estimated wait time:

```jsx
{isLoading && (
  <div className="text-center py-8">
    <div className="inline-block w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
    <p className="text-gray-500 mt-3 font-medium">Analysing image...</p>
    <p className="text-gray-400 text-sm mt-1">
      First request may take up to 30s (model loading)
    </p>
  </div>
)}
```

---

## Part 5 — Git Repository Structure

### 5.1 Recommended Branch Strategy

```
main          ← production (Vercel + HF Spaces auto-deploy)
  │
  ├── dev     ← active development
  │
  └── training ← training notebooks (large files, not deployed)
```

### 5.2 `.gitignore`

```
# Python
venv/
__pycache__/
*.pyc
*.pyo
.env

# Model weights (stored on HF Hub, not Git)
*.pt
*.pth
*.onnx
backend/weights/

# Training artifacts
training/datasets/
training/runs/
*.zip

# Frontend build
frontend/dist/
frontend/node_modules/

# Misc
.DS_Store
*.log
```

### 5.3 Final Repository on GitHub

```bash
# Initial push
cd descry
git remote add origin https://github.com/YOUR_USERNAME/descry.git
git push -u origin main

# Connect Vercel to GitHub repo:
# vercel.com → New Project → Import from GitHub → descry → frontend/ subfolder
```

---

## Part 6 — Troubleshooting Deployment

### HF Spaces Build Failing

```bash
# Check build logs: huggingface.co/spaces/USERNAME/descry-api → Logs
# Common issues:

# Issue 1: libGL not found
# Fix: Already handled in Dockerfile with libgl1-mesa-glx

# Issue 2: Port mismatch
# Fix: Ensure CMD uses port 7860 (HF Spaces requirement)

# Issue 3: Weights download timeout
# Fix: Add retry logic to ensure_weights():
from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def download_file(filename):
    hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=str(WEIGHTS_DIR))
```

### Vercel Build Failing

```bash
# Check: frontend/package.json has correct build script
"scripts": {
  "build": "vite build",
  "dev": "vite"
}

# Check: VITE_API_URL is set in Vercel environment variables
# Check: vite.config.js is correct
```

### CORS Errors in Browser

```python
# In main.py, make sure CORS middleware includes your exact Vercel URL:
allow_origins=[
    "https://descry-app.vercel.app",
    "https://descry-git-main-USERNAME.vercel.app",  # Git branch preview URL
    "http://localhost:5173",
]
```
