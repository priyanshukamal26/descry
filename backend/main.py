"""
Brand Vision FastAPI Application.
Entry point: uvicorn main:app --reload --port 8000
"""

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models.pipeline import get_pipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Brand Vision API",
    description="Deep learning-based brand recognition and product category classification.",
    version="1.0.0",
)

# CORS — development: allow all; production origins are tightened in env config.
ALLOWED_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: pre-load models so the first request is fast
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Pre-load ONNX models at startup to avoid cold-start on first request."""
    get_pipeline()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    return {"status": "ok", "message": "Brand Vision API is running"}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept a product image and return brand + category predictions.

    - **file**: image file (JPEG, PNG, or WebP), max 10 MB
    """
    # Validate MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, WebP).",
        )

    contents = await file.read()

    # Validate file size (10 MB hard limit)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Image too large. Maximum size is 10 MB.",
        )

    # Validate the file is a readable image
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # checks header integrity
        image = Image.open(io.BytesIO(contents))  # re-open after verify() closes it
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image. Ensure it is a valid JPEG, PNG, or WebP file.",
        )

    pipeline = get_pipeline()

    try:
        result = pipeline.predict(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return JSONResponse(content=result)
