from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class EmbeddingManager:
    """Generates embeddings with a SentenceTransformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )