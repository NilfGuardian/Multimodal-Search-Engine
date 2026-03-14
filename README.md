# Multimodal Search Engine

A college project that supports image retrieval using either text queries or query images. The system uses CLIP embeddings and Milvus Lite for cosine similarity vector search.

## Features
- Text-to-image search using CLIP text embeddings
- Image-to-image search using CLIP image embeddings
- Milvus Lite local vector storage (no Docker required)
- FastAPI backend with indexing/search endpoints
- Streamlit frontend with two tabs: Search and Add Images

## Project Structure
```text
multimodal-search/
|-- images/
|-- temp/
|-- src/
|   |-- embedding.py
|   |-- milvus_client.py
|   |-- api.py
|   `-- ui.py
|-- tests/
|-- requirements.txt
|-- README.md
`-- PROJECT_CONTEXT.md
```

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

Windows quick setup:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generate starter dataset (50 images):

```powershell
.\.venv\Scripts\python.exe scripts/generate_sample_images.py
```

## Run Backend
```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Note on CLIP loading behavior:

- Default behavior is fast startup: app uses cached CLIP files only and falls back to deterministic embeddings if CLIP files are not already local.
- To allow online CLIP download/load from Hugging Face, set:

```powershell
set CLIP_DOWNLOAD_ALLOWED=1
```

## Run Frontend
In a second terminal:

```bash
streamlit run src/ui.py
```

Optional backend URL override:

```bash
set API_BASE_URL=http://127.0.0.1:8000
```

## API Endpoints
- `POST /index` - Upload and index one image
- `POST /index/batch` - Upload and index multiple images
- `POST /search/text?q=query&top_k=5` - Search using text
- `POST /search/image?top_k=5` - Search using query image upload
- `GET /stats` - Get indexed image count

## Quick Test Plan
1. Upload 5 to 10 images via `Add Images` tab.
2. Run at least 3 text queries.
3. Run image search with 2 different query images.
4. Test error cases: empty text query and invalid file type.
5. Confirm response time is acceptable for your machine.

Automated end-to-end check:

```powershell
.\.venv\Scripts\python.exe tests/demo_e2e.py
```

## Notes
- CLIP model: `openai/clip-vit-base-patch32` (ViT-B/32 family)
- Embeddings are normalized before storage/search.
- Milvus collection name: `image_search`
- HNSW index: `M=16`, `efConstruction=200`, query `ef=64`
- If CLIP or Milvus cannot initialize on a machine, deterministic fallbacks keep the app runnable.
