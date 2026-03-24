"""Quick smoke tests for the Brand Vision API.

Usage (with backend running on :8000):
    python test_api.py
"""

import sys
from pathlib import Path

import requests

BASE = "http://localhost:8000"


def test_health() -> None:
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.json().get("status") == "healthy"
    print("✓ /health — OK")


def test_root() -> None:
    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200
    print("✓ / — OK")


def test_predict_with_sample() -> None:
    test_img = Path("test_images/nike_shoe.jpg")
    if not test_img.exists():
        print(f"⚠  Skipping predict test — {test_img} not found")
        return

    with test_img.open("rb") as f:
        r = requests.post(
            f"{BASE}/predict",
            files={"file": ("nike_shoe.jpg", f, "image/jpeg")},
            timeout=30,
        )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()

    required_keys = [
        "brand", "brand_confidence", "category", "category_confidence",
        "attributes", "logo_detected", "bbox", "top3_brands", "top3_categories",
    ]
    for key in required_keys:
        assert key in data, f"Missing key: {key}"

    assert 0.0 <= data["brand_confidence"] <= 1.0
    assert 0.0 <= data["category_confidence"] <= 1.0
    assert len(data["top3_brands"]) == 3
    assert len(data["top3_categories"]) == 3

    print(
        f"✓ /predict — brand={data['brand']} ({data['brand_confidence']:.0%})"
        f"  category={data['category']} ({data['category_confidence']:.0%})"
    )


def test_predict_non_image() -> None:
    r = requests.post(
        f"{BASE}/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    assert r.status_code == 400, f"Expected 400 for non-image, got {r.status_code}"
    print("✓ /predict non-image → 400")


def test_predict_oversized() -> None:
    big = b"x" * (11 * 1024 * 1024)  # 11 MB
    r = requests.post(
        f"{BASE}/predict",
        files={"file": ("big.jpg", big, "image/jpeg")},
        timeout=15,
    )
    assert r.status_code == 400, f"Expected 400 for oversized, got {r.status_code}"
    print("✓ /predict oversized → 400")


if __name__ == "__main__":
    try:
        test_health()
        test_root()
        test_predict_with_sample()
        test_predict_non_image()
        test_predict_oversized()
        print("\nAll tests passed ✓")
    except AssertionError as exc:
        print(f"\n✗ Test failed: {exc}", file=sys.stderr)
        sys.exit(1)
