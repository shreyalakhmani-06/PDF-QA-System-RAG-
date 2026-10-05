from .config import PDF_DIR, SCORE_THRESHOLD, TOP_K
from .embeddings import EmbeddingManager
from .generator import GeminiGenerator
from .ingest import build_index
from .retriever import RAGRetriever
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingManager()
        self.store = VectorStore()
        self.retriever = RAGRetriever(self.store, self.embedder)
        self.generator = GeminiGenerator()

    def build_index(self, pdf_dir=PDF_DIR, **kwargs):
        return build_index(self.embedder, self.store, pdf_dir, **kwargs)

    def ask(self, question: str, top_k: int = TOP_K, score_threshold: float = SCORE_THRESHOLD):
        docs = self.retriever.retrieve(question, top_k=top_k, score_threshold=score_threshold)
        return self.generator.answer(question, docs)