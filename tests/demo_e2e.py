"""End-to-end demo script for indexing and search.

This script runs against FastAPI app directly with TestClient.
It also forces offline fallback embeddings to avoid model download blockers.

Run:
    python tests/demo_e2e.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

# Force deterministic fallback embeddings for fast, reliable local demo.
os.environ["FORCE_FAKE_EMBEDDINGS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app  # noqa: E402


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (256, 256), color=color)
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return buff.getvalue()


def main() -> None:
    client = TestClient(app)

    samples = [
        ("red.png", _png_bytes((220, 40, 40))),
        ("green.png", _png_bytes((40, 220, 40))),
        ("blue.png", _png_bytes((40, 40, 220))),
    ]

    t0 = time.perf_counter()
    files = [("files", (name, payload, "image/png")) for name, payload in samples]
    index_resp = client.post("/index/batch", files=files)
    assert index_resp.status_code == 200, index_resp.text

    bad_upload = client.post(
        "/index",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert bad_upload.status_code == 400, bad_upload.text

    empty_query = client.post("/search/text", params={"q": "   ", "top_k": 3})
    assert empty_query.status_code in (400, 422), empty_query.text

    text_queries = ["red object", "green scene", "blue color"]
    for query in text_queries:
        resp = client.post("/search/text", params={"q": query, "top_k": 3})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert isinstance(data.get("results"), list)

    query_img = _png_bytes((225, 50, 50))
    image_resp = client.post("/search/image", params={"top_k": 3}, files={"file": ("query.png", query_img, "image/png")})
    assert image_resp.status_code == 200, image_resp.text

    stats = client.get("/stats")
    assert stats.status_code == 200, stats.text
    indexed_count = int(stats.json().get("indexed_count", 0))
    assert indexed_count >= 3

    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"E2E flow exceeded 2s threshold: {elapsed:.2f}s"
    print("E2E demo passed")
    print(f"Indexed images count: {indexed_count}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
