from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR
from .embeddings import EmbeddingManager
from .vectorstore import VectorStore


def load_pdfs(pdf_dir: Path = PDF_DIR) -> List:
    docs = []
    for pdf_path in sorted(Path(pdf_dir).glob("**/*.pdf")):
        docs.extend(PyMuPDFLoader(str(pdf_path)).load())
    return docs


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_index(embedder: EmbeddingManager, store: VectorStore, pdf_dir: Path = PDF_DIR,
                chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """Full rebuild: load -> split -> embed -> store. Returns (num_pages, num_chunks)."""
    pages = load_pdfs(pdf_dir)
    if not pages:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    chunks = split_documents(pages, chunk_size, chunk_overlap)
    embeddings = embedder.generate_embeddings([c.page_content for c in chunks], show_progress=True)

    store.reset()
    store.add_documents(chunks, embeddings)
    return len(pages), len(chunks)