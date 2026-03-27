# 🤖 Agent Instructions for Antigravity
## Brand Vision — Complete Build Checklist

> This document is the primary instruction file for Antigravity.
> Follow every section in order. Do not skip steps.
> All decisions have been pre-made. Execute, do not redesign.

---

## Critical Constraints — Never Violate These

1. **$0 budget** — every tool, platform, and service must be free
2. **VRAM limit: 6GB** — never run two training jobs simultaneously
3. **ONNX for inference** — do NOT load PyTorch models in the deployed backend
4. **Batch sizes** — max 32 for EfficientNet-B0, max 16 for YOLOv8 locally
5. **Python 3.10+** — use f-strings, pathlib, type hints throughout
6. **No Docker Desktop** — use HF Spaces Docker-based deployment only
7. **React + Vite + Tailwind** — no other frontend framework

---

## File Reading Order (For Agent)

Read these documents in this order before writing any code:

1. `README.md` — project overview, repo structure, deliverables checklist
2. `ARCHITECTURE.md` — model specs, data flow, API spec, VRAM budget
3. `IMPLEMENTATION_PLAN.md` — phase-by-phase instructions with full code
4. `TRAINING_GUIDE.md` — dataset download, preparation, training notebooks
5. `DEPLOYMENT_GUIDE.md` — HF Spaces + Vercel deployment steps

---

## Master Build Checklist

### Phase 0 — Environment
- [ ] Python 3.10+ virtualenv created and activated
- [ ] `backend/requirements.txt` created with exact versions from IMPLEMENTATION_PLAN.md
- [ ] All packages installed without errors
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] GPU reported as `NVIDIA GeForce RTX 4050`
- [ ] Frontend: `npm create vite@latest` with React template done
- [ ] Frontend: `react-dropzone` and `axios` installed
- [ ] `.gitignore` created (see DEPLOYMENT_GUIDE.md §5.2)
- [ ] GitHub repository created and initial commit pushed

### Phase 1 — Datasets
- [ ] Kaggle CLI installed and configured with API key
- [ ] LogoDet-3K downloaded to `training/datasets/logodet3k/`
- [ ] Product-10K downloaded to `training/datasets/product10k/`
- [ ] Data preparation notebook (`01_data_preparation.ipynb`) runs without errors
- [ ] `brand_logo_split/` created with train/val/test subdirs per brand
- [ ] `product_category_split/` created with train/val/test subdirs per category
- [ ] `yolo_logo_detection/` created in YOLO format (images/ + labels/)
- [ ] `logo_data.yaml` created with correct paths
- [ ] Class counts logged — no class has fewer than 20 training samples

### Phase 2 — YOLOv8 Training
- [ ] `02_train_yolo_logo.ipynb` created
- [ ] YOLOv8 training completes (locally or on Kaggle)
- [ ] `best.pt` saved to `training/runs/logo_detector/v1/weights/`
- [ ] mAP50 ≥ 0.60 on validation set
- [ ] ONNX export: `best.onnx` generated
- [ ] ONNX verified: loads with `onnxruntime`, produces output shape `[1, 5, N]`

### Phase 3 — EfficientNet Training
- [ ] `03_train_efficientnet_category.ipynb` created
- [ ] Category model trains for 35 epochs (5 warmup + 30 fine-tune)
- [ ] `category/best_model.pt` saved
- [ ] Category val accuracy ≥ 0.65
- [ ] `04_train_efficientnet_brand.ipynb` created (copy of category notebook with path changes)
- [ ] Brand model trains for 35 epochs
- [ ] `brand/best_model.pt` saved
- [ ] Brand val accuracy ≥ 0.60

### Phase 4 — ONNX Export
- [ ] `training/export_onnx.py` created
- [ ] `category_classifier.onnx` exported and verified
- [ ] `category_classifier_labels.json` saved
- [ ] `brand_classifier.onnx` exported and verified
- [ ] `brand_classifier_labels.json` saved
- [ ] All 5 weight files copied to `backend/weights/`
- [ ] All ONNX files verified with onnxruntime (correct output shapes)

### Phase 5 — Backend
- [ ] `backend/models/pipeline.py` created (full pipeline with ONNX inference)
- [ ] `backend/main.py` created (FastAPI with CORS, /predict endpoint)
- [ ] `backend/models/__init__.py` created (empty)
- [ ] `uvicorn main:app --reload` starts without errors
- [ ] `/health` endpoint returns `{"status": "healthy"}`
- [ ] `/predict` returns valid JSON with Nike shoe test image
- [ ] All fields present: brand, brand_confidence, category, category_confidence, attributes, bbox, top3_brands, top3_categories
- [ ] Error handling: non-image returns 400, oversized returns 400
- [ ] `backend/test_api.py` created and all tests pass
- [ ] Inference latency < 500ms locally (warm, after model load)

### Phase 6 — Frontend
- [ ] `frontend/src/api/predict.js` created (uses `VITE_API_URL` env var)
- [ ] `frontend/src/components/Uploader.jsx` created (react-dropzone)
- [ ] `frontend/src/components/ConfidenceBar.jsx` created
- [ ] `frontend/src/components/ResultCard.jsx` created (brand, category, top3, attributes)
- [ ] `frontend/src/App.jsx` created (orchestrates all components)
- [ ] `frontend/vite.config.js` updated with proxy config
- [ ] `frontend/vercel.json` created
- [ ] `npm run dev` opens UI without errors
- [ ] Drag-drop image upload works
- [ ] Results card renders correctly with all fields
- [ ] Loading spinner shows during API call
- [ ] Error message shows when API returns error
- [ ] Responsive: works on mobile screen sizes

### Phase 7 — Integration Testing
- [ ] Backend running on :8000, frontend on :5173
- [ ] Nike shoe image → brand: Nike (or near-Nike brand), category: shoes
- [ ] Apple iPhone image → brand: Apple, category: smartphone
- [ ] CORS: browser fetch to localhost:8000 succeeds (no CORS errors in console)
- [ ] bounding box coordinates are valid integers
- [ ] All confidence scores are floats between 0.0 and 1.0
- [ ] top3_brands and top3_categories each have exactly 3 entries
- [ ] Non-image file upload → error message displayed (not a crash)

### Phase 8 — Deployment
- [ ] HuggingFace account created
- [ ] `descry-weights` model repo created on HF Hub
- [ ] All 5 ONNX + JSON weight files uploaded to HF Hub
- [ ] HF Space `descry-api` created (Docker, CPU Basic, Public)
- [ ] `backend/Dockerfile` created (see DEPLOYMENT_GUIDE.md §2.3)
- [ ] `pipeline.py` updated with `ensure_weights()` auto-download function
- [ ] `main.py` updated with production CORS origins
- [ ] Backend code pushed to HF Space repository
- [ ] HF Space Docker build succeeds (watch logs)
- [ ] `curl https://USERNAME-descry-api.hf.space/health` returns 200
- [ ] Vercel project created, linked to GitHub `frontend/` folder
- [ ] `VITE_API_URL` env var set in Vercel dashboard
- [ ] Frontend deployed successfully on Vercel
- [ ] Live URL accessible in browser
- [ ] End-to-end: upload image on live URL → prediction returned in UI

---

## Code Style Guidelines

```python
# Python
# - Use pathlib.Path for all file paths (no os.path)
# - Use type hints on all function signatures
# - Use dataclasses or Pydantic models for structured data
# - Keep functions under 30 lines; extract helpers aggressively
# - Add docstrings to all classes and public functions
# - Use f-strings, not .format() or %

# Example:
def preprocess_image(img: Image.Image, size: int = 224) -> np.ndarray:
    """Resize and normalise a PIL image for EfficientNet input."""
    ...
```

```jsx
// React / JavaScript
// - Functional components only (no class components)
// - useState / useCallback / useEffect for state
// - Tailwind utility classes for all styling (no inline styles except dynamic values)
// - PropTypes or JSDoc for component props
// - async/await for API calls (no .then() chains)
// - Handle loading, error, and success states for every API call
```

---

## What NOT to Do

- ❌ Do not use PyTorch for inference in the deployed backend (use ONNX Runtime)
- ❌ Do not commit `.pt` or `.onnx` files to Git (upload to HF Hub only)
- ❌ Do not use `os.path` — use `pathlib.Path`
- ❌ Do not use class components in React
- ❌ Do not use inline CSS — use Tailwind classes
- ❌ Do not train both models simultaneously on local GPU
- ❌ Do not hardcode the API URL in frontend code — use `VITE_API_URL`
- ❌ Do not use `allow_origins=["*"]` in production CORS config

---

## Quick Reference — Key File Locations

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI entry point |
| `backend/models/pipeline.py` | Full prediction pipeline |
| `backend/weights/` | ONNX model files (not in Git) |
| `frontend/src/App.jsx` | Main React app |
| `frontend/src/api/predict.js` | API client |
| `frontend/src/components/` | UI components |
| `training/datasets/` | Raw + processed datasets (not in Git) |
| `training/runs/` | Training outputs, weights (not in Git) |
| `backend/Dockerfile` | HF Spaces container definition |

---

## Estimated Timeline

| Phase | Hours | Platform |
|---|---|---|
| 0 — Environment Setup | 1–2h | Local |
| 1 — Dataset Prep | 3–5h | Local (no GPU needed) |
| 2 — YOLO Training | 1–2h | Local RTX 4050 |
| 3 — EfficientNet × 2 | 4–6h | Local RTX 4050 / Kaggle |
| 4 — ONNX Export | 0.5h | Local |
| 5 — Backend | 4–6h | Local |
| 6 — Frontend | 5–8h | Local |
| 7 — Integration | 2–3h | Local |
| 8 — Deployment | 2–3h | HF Spaces + Vercel |
| **Total** | **22–35h** | |

---

*All referenced code is in IMPLEMENTATION_PLAN.md. All deployment commands are in DEPLOYMENT_GUIDE.md. If something is ambiguous, prefer the most explicit, defensive implementation.*
