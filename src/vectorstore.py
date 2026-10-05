import hashlib
import os
from typing import Any, List

import chromadb
import numpy as np

from .config import COLLECTION_NAME, VECTOR_DIR


def _clean_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool values, and no None."""
    clean = {}
    for key, value in meta.items():
        if value is None:
            continue
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


class VectorStore:
    """Persistent ChromaDB collection using cosine distance."""

    def __init__(self, collection_name: str = COLLECTION_NAME, persist_directory=VECTOR_DIR):
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_collection()

    def add_documents(self, documents: List[Any], embeddings: np.ndarray, batch_size: int = 500) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids, texts, metadatas, vectors = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            meta = _clean_metadata(dict(doc.metadata))
            meta["chunk_index"] = i
            raw = f"{meta.get('source', '')}|{meta.get('page', '')}|{i}|{doc.page_content}"
            ids.append(hashlib.sha1(raw.encode("utf-8")).hexdigest())
            texts.append(doc.page_content)
            metadatas.append(meta)
            vectors.append(emb.tolist())

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end],
            )
        print(f"Stored {len(ids)} chunks. Collection size: {self.count()}")