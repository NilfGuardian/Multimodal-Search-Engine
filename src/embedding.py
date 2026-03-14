"""Following the multimodal search project in PROJECT_CONTEXT.md.

Embedding module for CLIP-based image and text vectors.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

ImageInput = Union[str, Path, Image.Image]


class MultimodalEmbedder:
    """Generate normalized CLIP embeddings for images and text.

    The model and processor are cached at the class level to avoid reloading.
    """

    _model: CLIPModel | None = None
    _processor: CLIPProcessor | None = None
    _device: torch.device | None = None
    _projection_matrix: np.ndarray | None = None

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", embedding_dim: int = 512) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._use_clip = False
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        """Load CLIP model and processor once per process."""
        if self.__class__._model is not None and self.__class__._processor is not None:
            self._use_clip = True
            return

        if os.getenv("FORCE_FAKE_EMBEDDINGS", "0") == "1":
            print("[embedder] FORCE_FAKE_EMBEDDINGS=1 set; using deterministic fallback embeddings")
            self._use_clip = False
            return

        allow_download = os.getenv("CLIP_DOWNLOAD_ALLOWED", "0") == "1"

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # By default, load only from local cache so requests don't block on large downloads.
            model = CLIPModel.from_pretrained(self.model_name, local_files_only=not allow_download)
            processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=not allow_download)
            model.to(device)
            model.eval()

            self.__class__._model = model
            self.__class__._processor = processor
            self.__class__._device = device
            self._use_clip = True
        except Exception as exc:  # noqa: BLE001
            # Fallback keeps the app functional in offline or constrained environments.
            print(f"[embedder] CLIP unavailable, using deterministic fallback embeddings: {exc}")
            self._use_clip = False

    @property
    def model(self) -> CLIPModel:
        assert self.__class__._model is not None
        return self.__class__._model

    @property
    def processor(self) -> CLIPProcessor:
        assert self.__class__._processor is not None
        return self.__class__._processor

    @property
    def device(self) -> torch.device:
        assert self.__class__._device is not None
        return self.__class__._device

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize vectors to unit length for cosine similarity."""
        return F.normalize(tensor, p=2, dim=-1)

    def _to_vector(self, tensor: torch.Tensor) -> List[float]:
        """Convert tensor embedding to plain Python float list."""
        return tensor.detach().cpu().flatten().tolist()

    def _load_image(self, image_input: ImageInput) -> Image.Image:
        """Load image from a file path or pass through PIL image.

        Raises:
            ValueError: If the image cannot be opened or decoded.
            TypeError: If image_input type is unsupported.
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise ValueError(f"Image file does not exist: {image_path}")
            try:
                with Image.open(image_path) as img:
                    return img.convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Corrupt or unsupported image file: {image_path}") from exc

        raise TypeError("image_input must be a file path or PIL.Image.Image")

    def _normalize_numpy(self, vector: np.ndarray) -> np.ndarray:
        """Normalize numpy vectors with numerical stability."""
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def _get_projection_matrix(self, input_dim: int) -> np.ndarray:
        """Create and cache deterministic projection matrix for fallback image embeddings."""
        matrix = self.__class__._projection_matrix
        if matrix is None or matrix.shape[0] != input_dim or matrix.shape[1] != self.embedding_dim:
            rng = np.random.default_rng(42)
            matrix = rng.standard_normal((input_dim, self.embedding_dim), dtype=np.float32)
            self.__class__._projection_matrix = matrix
        return matrix

    def _fallback_embed_text(self, text: str) -> List[float]:
        """Deterministic text embedding fallback for offline mode."""
        digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
        repeated = (digest * ((self.embedding_dim // len(digest)) + 1))[: self.embedding_dim]
        vector = np.frombuffer(repeated, dtype=np.uint8).astype(np.float32)
        vector = (vector - 127.5) / 127.5
        return self._normalize_numpy(vector).tolist()

    def _fallback_embed_image(self, image: Image.Image) -> List[float]:
        """Deterministic image embedding fallback based on projected pixel features."""
        resized = image.convert("RGB").resize((32, 32))
        arr = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
        projection = self._get_projection_matrix(arr.shape[0])
        vector = arr @ projection
        return self._normalize_numpy(vector.astype(np.float32)).tolist()

    @torch.no_grad()
    def embed_image(self, image_input: ImageInput) -> List[float]:
        """Generate a normalized 512-d embedding from an image input."""
        image = self._load_image(image_input)
        if not self._use_clip:
            return self._fallback_embed_image(image)

        model_inputs = self.processor(images=image, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        image_features = self.model.get_image_features(**model_inputs)
        normalized = self._normalize_tensor(image_features)
        return self._to_vector(normalized)

    @torch.no_grad()
    def embed_text(self, text: str) -> List[float]:
        """Generate a normalized 512-d embedding from a text query."""
        if not text or not text.strip():
            raise ValueError("Text query must not be empty")

        if not self._use_clip:
            return self._fallback_embed_text(text)

        model_inputs = self.processor(text=[text.strip()], return_tensors="pt", padding=True, truncation=True)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        text_features = self.model.get_text_features(**model_inputs)
        normalized = self._normalize_tensor(text_features)
        return self._to_vector(normalized)

    @torch.no_grad()
    def embed_images_batch(self, image_inputs: Sequence[ImageInput]) -> List[List[float]]:
        """Embed a batch of images, skipping unreadable inputs.

        Returns only successfully embedded images.
        """
        loaded_images: List[Image.Image] = []
        vectors: List[List[float]] = []

        for image_input in image_inputs:
            try:
                loaded_images.append(self._load_image(image_input))
            except Exception as exc:  # noqa: BLE001
                print(f"[embedder] Skipping image due to error: {exc}")

        if not loaded_images:
            return vectors

        if not self._use_clip:
            return [self._fallback_embed_image(img) for img in loaded_images]

        model_inputs = self.processor(images=loaded_images, return_tensors="pt")
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
        image_features = self.model.get_image_features(**model_inputs)
        normalized = self._normalize_tensor(image_features)
        for row in normalized:
            vectors.append(self._to_vector(row))
        return vectors
