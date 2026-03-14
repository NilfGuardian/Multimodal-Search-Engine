"""FastAPI backend for indexing and searching images."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.embedding import MultimodalEmbedder
from src.milvus_client import MilvusSearchClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
TEMP_DIR = PROJECT_ROOT / "temp"
MILVUS_DB_PATH = PROJECT_ROOT / "milvus_lite.db"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multimodal Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def get_embedder() -> MultimodalEmbedder:
    """Create and cache embedding service lazily."""
    return MultimodalEmbedder()


@lru_cache(maxsize=1)
def get_vector_db() -> MilvusSearchClient:
    """Create and cache vector DB service lazily."""
    return MilvusSearchClient(db_path=str(MILVUS_DB_PATH), collection_name="image_search")


def _validate_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    return suffix


def _safe_stem(filename: str) -> str:
    """Create filesystem-safe stem while preserving original filename intent."""
    stem = Path(filename).stem.strip().replace(" ", "_")
    stem = re.sub(r"[^A-Za-z0-9._-]", "", stem)
    return stem or "uploaded_image"


def _rerank_results_by_filename(query: str, results: List[dict]) -> List[dict]:
    """Blend vector score with lexical filename match for practical relevance."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t]
    if not tokens:
        return results

    reranked: List[dict] = []
    for item in results:
        image_path = str(item.get("image_path", ""))
        score = float(item.get("score", 0.0))
        name = Path(image_path).name.lower()

        lexical_hits = sum(1 for tok in tokens if tok in name)
        lexical_boost = 0.12 * lexical_hits
        blended = score + lexical_boost

        reranked.append({**item, "score": blended})

    reranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return reranked


def _discover_local_images() -> List[Path]:
    """Find supported image files already present in images directory."""
    files: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(IMAGES_DIR.glob(pattern))
    return sorted(set(files))


def _index_existing_images(force: bool = False) -> dict:
    """Populate vector DB from local images.

    If force is False, indexing is skipped when the DB already has entries.
    """
    embedder = get_embedder()
    vector_db = get_vector_db()

    existing_count = vector_db.count_images()
    if not force and existing_count > 0:
        return {"indexed_count": 0, "skipped": existing_count, "status": "already_populated"}

    image_files = _discover_local_images()
    indexed = 0
    failed: List[dict] = []

    for image_path in image_files:
        try:
            embedding = embedder.embed_image(image_path)
            vector_db.insert_image(str(image_path), embedding)
            indexed += 1
        except Exception as exc:  # noqa: BLE001
            failed.append({"image_path": str(image_path), "error": str(exc)})
            print(f"[api] Auto-index failed for {image_path}: {exc}")

    return {
        "indexed_count": indexed,
        "failed_count": len(failed),
        "failed": failed,
        "status": "indexed_from_images_dir",
    }


async def _save_upload(upload: UploadFile, destination_dir: Path) -> Path:
    """Persist uploaded file and return saved file path."""
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing filename")

    suffix = _validate_suffix(upload.filename)
    original_name = upload.filename
    save_name = f"{_safe_stem(original_name)}_{uuid4().hex[:8]}{suffix}"
    file_path = destination_dir / save_name

    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    file_path.write_bytes(data)
    return file_path


@app.post("/index")
async def index_single_image(file: UploadFile = File(...)) -> dict:
    """Upload and index a single image."""
    try:
        embedder = get_embedder()
        vector_db = get_vector_db()
        saved_path = await _save_upload(file, IMAGES_DIR)
        embedding = embedder.embed_image(saved_path)
        vector_db.insert_image(str(saved_path), embedding)
        return {
            "message": "Image indexed successfully",
            "image_path": str(saved_path),
            "original_filename": file.filename,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[api] /index failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/index/batch")
async def index_batch_images(files: List[UploadFile] = File(...)) -> dict:
    """Upload and index multiple images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    indexed = 0
    failed: List[dict] = []
    embedder = get_embedder()
    vector_db = get_vector_db()

    for upload in files:
        try:
            saved_path = await _save_upload(upload, IMAGES_DIR)
            embedding = embedder.embed_image(saved_path)
            vector_db.insert_image(str(saved_path), embedding)
            indexed += 1
        except Exception as exc:  # noqa: BLE001
            failed.append({"filename": upload.filename, "error": str(exc)})
            print(f"[api] batch item failed ({upload.filename}): {exc}")

    return {
        "message": "Batch indexing complete",
        "indexed_count": indexed,
        "failed_count": len(failed),
        "failed": failed,
    }


@app.post("/index/local")
async def index_local_images() -> dict:
    """Index all images currently present in ./images directory."""
    try:
        summary = _index_existing_images(force=True)
        return {"message": "Local indexing complete", **summary}
    except Exception as exc:  # noqa: BLE001
        print(f"[api] /index/local failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/text")
async def search_by_text(
    q: str = Query(..., min_length=1, description="Text query"),
    top_k: int = Query(5, ge=1, le=50),
) -> dict:
    """Search indexed images by text query."""
    try:
        _index_existing_images(force=False)
        embedder = get_embedder()
        vector_db = get_vector_db()
        query_embedding = embedder.embed_text(q)
        results = vector_db.search(query_embedding, top_k=top_k * 3)
        results = _rerank_results_by_filename(q, results)[:top_k]
        return {"query": q, "results": results}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[api] /search/text failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=50)) -> dict:
    """Search indexed images using an uploaded image query."""
    try:
        _index_existing_images(force=False)
        embedder = get_embedder()
        vector_db = get_vector_db()
        saved_path = await _save_upload(file, TEMP_DIR)
        query_embedding = embedder.embed_image(saved_path)
        results = vector_db.search(query_embedding, top_k=top_k)
        return {"query_image": str(saved_path), "results": results}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[api] /search/image failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/stats")
async def get_stats() -> dict:
    """Get number of indexed images."""
    try:
        vector_db = get_vector_db()
        return {"indexed_count": vector_db.count_images()}
    except Exception as exc:  # noqa: BLE001
        print(f"[api] /stats failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
