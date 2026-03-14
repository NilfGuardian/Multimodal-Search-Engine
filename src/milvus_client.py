"""Based on PROJECT_CONTEXT.md - Milvus client implementation.

Uses Milvus Lite for vector storage and cosine similarity search.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

try:
    import chromadb
except Exception:  # noqa: BLE001
    chromadb = None


class MilvusSearchClient:
    """Milvus vector search wrapper for image embeddings."""

    def __init__(
        self,
        db_path: str = "./milvus_lite.db",
        collection_name: str = "image_search",
        embedding_dim: int = 512,
    ) -> None:
        self.db_path = str(Path(db_path))
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection: Collection | None = None
        self.sqlite_conn: sqlite3.Connection | None = None
        self.chroma_client = None
        self.chroma_collection = None
        self._memory_store: List[Dict[str, Any]] = []
        self.backend = "milvus"

        try:
            self._connect()
            self._ensure_collection()
            self.backend = "milvus"
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Milvus unavailable, trying SQLite fallback: {exc}")
            if self._init_sqlite():
                self.backend = "sqlite"
            elif self._init_chroma():
                self.backend = "chroma"
            else:
                print("[milvus] No persistent backend available, falling back to in-memory vector store")
                self.backend = "memory"

    def _init_sqlite(self) -> bool:
        """Initialize persistent SQLite fallback backend."""
        try:
            db_file = Path(self.db_path).with_name("vector_store.sqlite3")
            db_file.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT UNIQUE NOT NULL,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
            self.sqlite_conn = conn
            print(f"[milvus] Using SQLite persistent backend at: {db_file}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Failed to initialize SQLite backend: {exc}")
            return False

    def _init_chroma(self) -> bool:
        """Initialize persistent ChromaDB fallback backend."""
        if chromadb is None:
            return False

        try:
            base = Path(self.db_path)
            chroma_dir = base.parent / "chroma_db"
            chroma_dir.mkdir(parents=True, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            print(f"[milvus] Using ChromaDB persistent backend at: {chroma_dir}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Failed to initialize ChromaDB: {exc}")
            return False

    def _connect(self) -> None:
        """Connect to local Milvus Lite database file."""
        try:
            connections.connect(alias="default", uri=self.db_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Connection failed, retrying once: {exc}")
            connections.connect(alias="default", uri=self.db_path)

    def _ensure_collection(self) -> None:
        """Create and load collection if it does not exist."""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
                ]
                schema = CollectionSchema(
                    fields=fields,
                    description="Image search collection",
                    enable_dynamic_field=False,
                )
                self.collection = Collection(name=self.collection_name, schema=schema)

            assert self.collection is not None
            if not self.collection.indexes:
                self.collection.create_index(
                    field_name="embedding",
                    index_params={
                        "index_type": "HNSW",
                        "metric_type": "COSINE",
                        "params": {"M": 16, "efConstruction": 200},
                    },
                    index_name="embedding_hnsw",
                )
            self.collection.load()
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Failed to initialize collection: {exc}")
            raise

    def _memory_cosine(self, v1: List[float], v2: List[float]) -> float:
        """Cosine similarity for in-memory fallback backend."""
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if math.isclose(denom, 0.0):
            return 0.0
        return float(np.dot(a, b) / denom)

    def insert_image(self, image_path: str, embedding: List[float]) -> int:
        """Insert one image embedding and return inserted row count."""
        try:
            if len(embedding) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}"
                )

            if self.backend == "chroma":
                if self.chroma_collection is None:
                    raise RuntimeError("Chroma collection is not initialized")
                image_id = hashlib.sha1(image_path.encode("utf-8")).hexdigest()
                self.chroma_collection.upsert(
                    ids=[image_id],
                    embeddings=[embedding],
                    metadatas=[{"image_path": image_path}],
                )
                return 1

            if self.backend == "sqlite":
                if self.sqlite_conn is None:
                    raise RuntimeError("SQLite connection is not initialized")
                embedding_json = json.dumps(np.asarray(embedding, dtype=np.float32).tolist())
                self.sqlite_conn.execute(
                    """
                    INSERT INTO embeddings (image_path, embedding_json)
                    VALUES (?, ?)
                    ON CONFLICT(image_path) DO UPDATE SET embedding_json=excluded.embedding_json
                    """,
                    (image_path, str(embedding_json)),
                )
                self.sqlite_conn.commit()
                return 1

            if self.backend == "memory":
                self._memory_store.append({"image_path": image_path, "embedding": list(embedding)})
                return 1

            if self.collection is None:
                raise RuntimeError("Milvus collection is not initialized")

            result = self.collection.insert([[image_path], [embedding]])
            self.collection.flush()
            inserted = len(result.primary_keys) if result is not None else 0
            return inserted
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Insert failed for {image_path}: {exc}")
            raise

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search nearest images by cosine similarity."""
        try:
            if len(query_embedding) != self.embedding_dim:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {len(query_embedding)}"
                )

            if self.backend == "chroma":
                if self.chroma_collection is None:
                    raise RuntimeError("Chroma collection is not initialized")
                response = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(1, top_k),
                    include=["metadatas", "distances"],
                )
                metadatas = response.get("metadatas", [[]])[0]
                distances = response.get("distances", [[]])[0]
                payload: List[Dict[str, Any]] = []
                for meta, distance in zip(metadatas, distances):
                    if not meta:
                        continue
                    similarity = 1.0 - float(distance)
                    payload.append({"image_path": meta.get("image_path"), "score": similarity})
                return payload

            if self.backend == "sqlite":
                if self.sqlite_conn is None:
                    raise RuntimeError("SQLite connection is not initialized")
                cursor = self.sqlite_conn.execute("SELECT image_path, embedding_json FROM embeddings")
                rows = cursor.fetchall()
                scored: List[Dict[str, Any]] = []
                for image_path, embedding_json in rows:
                    vector = np.asarray(json.loads(embedding_json), dtype=np.float32)
                    score = self._memory_cosine(query_embedding, vector.tolist())
                    scored.append({"image_path": image_path, "score": score})
                scored.sort(key=lambda x: x["score"], reverse=True)
                return scored[: max(1, top_k)]

            if self.backend == "memory":
                scored = [
                    {
                        "image_path": item["image_path"],
                        "score": self._memory_cosine(query_embedding, item["embedding"]),
                    }
                    for item in self._memory_store
                ]
                scored.sort(key=lambda x: x["score"], reverse=True)
                return scored[: max(1, top_k)]

            if self.collection is None:
                raise RuntimeError("Milvus collection is not initialized")

            self.collection.load()
            search_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=max(1, top_k),
                output_fields=["image_path"],
            )

            payload: List[Dict[str, Any]] = []
            for hit in search_results[0]:
                image_path = None
                if hasattr(hit, "entity") and hit.entity is not None:
                    image_path = hit.entity.get("image_path")
                payload.append(
                    {
                        "image_path": image_path,
                        "score": float(hit.score),
                    }
                )
            return payload
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Search failed: {exc}")
            raise

    def count_images(self) -> int:
        """Return number of indexed images."""
        try:
            if self.backend == "chroma":
                if self.chroma_collection is None:
                    return 0
                return int(self.chroma_collection.count())

            if self.backend == "sqlite":
                if self.sqlite_conn is None:
                    return 0
                cursor = self.sqlite_conn.execute("SELECT COUNT(*) FROM embeddings")
                row = cursor.fetchone()
                return int(row[0]) if row else 0

            if self.backend == "memory":
                return len(self._memory_store)

            if self.collection is None:
                raise RuntimeError("Milvus collection is not initialized")
            return self.collection.num_entities
        except Exception as exc:  # noqa: BLE001
            print(f"[milvus] Count failed: {exc}")
            raise
