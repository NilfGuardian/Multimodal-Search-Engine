"""Streamlit frontend for multimodal image search."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Multimodal Search Engine", layout="wide")
st.title("Multimodal Search Engine")
st.caption("Search images using text or another image, and add new images to the index.")


def _backend_is_reachable(timeout: int = 5) -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=timeout)
        return response.ok
    except Exception:
        return False


def _resolve_python_executable(project_root: Path) -> str:
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)

    runtime_python = project_root / ".runtime_venv" / "Scripts" / "python.exe"
    if runtime_python.exists():
        return str(runtime_python)

    return sys.executable


def _try_start_backend() -> bool:
    project_root = Path(__file__).resolve().parents[1]
    python_exe = _resolve_python_executable(project_root)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    try:
        subprocess.Popen(
            [
                python_exe,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        return False

    for _ in range(20):
        if _backend_is_reachable(timeout=3):
            return True
        time.sleep(0.6)

    return False


def _ensure_backend() -> None:
    if _backend_is_reachable(timeout=3):
        st.session_state["backend_ready"] = True
        return

    if not st.session_state.get("backend_start_attempted"):
        st.session_state["backend_start_attempted"] = True
        with st.spinner("Backend is offline. Attempting to start it..."):
            if _try_start_backend():
                st.session_state["backend_ready"] = True
                st.success("Backend started successfully.")
                return

    st.session_state["backend_ready"] = False
    st.error(
        "Backend is not reachable at http://127.0.0.1:8000. "
        "Start it with: python -m uvicorn src.api:app --host 127.0.0.1 --port 8000"
    )


_ensure_backend()


def _request_stats() -> int | None:
    if not st.session_state.get("backend_ready", False):
        return None

    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=15)
        response.raise_for_status()
        return int(response.json().get("indexed_count", 0))
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Unable to fetch stats: {exc}")
        return None


def _display_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        st.info("No matching images found.")
        return

    def _safe_image_render(path: str) -> None:
        """Render image across Streamlit versions with compatible kwargs."""
        try:
            st.image(path, use_container_width=True)
        except TypeError:
            # Older Streamlit versions use use_column_width instead.
            st.image(path, use_column_width=True)

    cols = st.columns(3)
    for idx, item in enumerate(results):
        image_path = item.get("image_path")
        score = item.get("score")
        with cols[idx % 3]:
            if image_path:
                _safe_image_render(image_path)
            st.caption(f"Similarity score: {score:.4f}" if isinstance(score, (int, float)) else "Similarity score: N/A")
            st.code(str(image_path) if image_path else "Unknown path")


stats_count = _request_stats()
if stats_count is not None:
    st.metric("Indexed Images", stats_count)

search_tab, add_tab = st.tabs(["Search", "Add Images"])

with search_tab:
    st.subheader("Search")
    mode = st.radio("Search type", ["Text", "Image"], horizontal=True)

    if not st.session_state.get("backend_ready", False):
        st.info("Backend is offline. Start backend first, then run search.")
        st.stop()

    if mode == "Text":
        text_query = st.text_input("Enter text query", placeholder="example: red sports car")
        top_k_text = st.slider("Top K", min_value=1, max_value=20, value=5)

        if st.button("Run Text Search", type="primary"):
            if not text_query.strip():
                st.error("Please enter a non-empty text query.")
            else:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/search/text",
                        params={"q": text_query, "top_k": top_k_text},
                        timeout=30,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    _display_results(payload.get("results", []))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Text search failed: {exc}")

    else:
        uploaded_query_image = st.file_uploader(
            "Upload query image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )
        top_k_image = st.slider("Top K ", min_value=1, max_value=20, value=5)

        if st.button("Run Image Search", type="primary"):
            if not uploaded_query_image:
                st.error("Please upload an image for search.")
            else:
                try:
                    files = {
                        "file": (
                            uploaded_query_image.name,
                            uploaded_query_image.getvalue(),
                            uploaded_query_image.type or "application/octet-stream",
                        )
                    }
                    response = requests.post(
                        f"{API_BASE_URL}/search/image",
                        params={"top_k": top_k_image},
                        files=files,
                        timeout=60,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    _display_results(payload.get("results", []))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Image search failed: {exc}")

with add_tab:
    st.subheader("Add Images")

    if not st.session_state.get("backend_ready", False):
        st.info("Backend is offline. Start backend first, then index images.")
        st.stop()

    uploads = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if st.button("Index Uploaded Images"):
        if not uploads:
            st.error("Please upload at least one image.")
        else:
            progress = st.progress(0)
            total = len(uploads)

            files = [
                ("files", (up.name, up.getvalue(), up.type or "application/octet-stream"))
                for up in uploads
            ]

            try:
                response = requests.post(
                    f"{API_BASE_URL}/index/batch",
                    files=files,
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
                progress.progress(100)

                st.success(
                    f"Indexed {payload.get('indexed_count', 0)} / {total} images."
                )

                failed = payload.get("failed", [])
                if failed:
                    st.warning(f"{len(failed)} files failed to index.")
                    st.json(failed)

                updated_count = _request_stats()
                if updated_count is not None:
                    st.info(f"Total indexed images: {updated_count}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Batch indexing failed: {exc}")
