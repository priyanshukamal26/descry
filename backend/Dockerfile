FROM python:3.10-slim

WORKDIR /app

# System dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (enables Docker layer caching)
COPY requirements.txt .

# Install Python dependencies (no torch/torchvision — ONNX Runtime only)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create weights directory (populated at runtime via HF Hub auto-download)
RUN mkdir -p weights

# Hugging Face Spaces requires port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
