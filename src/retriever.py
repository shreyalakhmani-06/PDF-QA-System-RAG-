from typing import Any, Dict, List

from .config import SCORE_THRESHOLD, TOP_K
from .embeddings import EmbeddingManager
from .vectorstore import VectorStore


class RAGRetriever:
    """Embeds a query and fetches the most similar chunks."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD) -> List[Dict[str, Any]]:
        if self.vector_store.count() == 0:
            return []

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )
        if not results["documents"] or not results["documents"][0]:
            return []

        retrieved = []
        rows = zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0])
        for rank, (doc_id, content, metadata, distance) in enumerate(rows, start=1):
            similarity = 1 - distance
            if similarity >= score_threshold:
                retrieved.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "rank": rank,
                })
        return retrieved