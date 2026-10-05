import streamlit as st

from src.config import PDF_DIR, SCORE_THRESHOLD, TOP_K
from src.pipeline import RAGPipeline

st.set_page_config(page_title="PDF Q&A (RAG)", page_icon="📄")
st.title("📄 Chat with your PDFs")
st.caption("RAG pipeline: PyMuPDF → chunking → MiniLM embeddings → ChromaDB → Gemini")


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


try:
    pipeline = get_pipeline()
except ValueError as err:  # e.g. missing API key
    st.error(str(err))
    st.stop()

# ---------- Sidebar: upload + index ----------
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Index documents"):
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploads or []:
            (PDF_DIR / f.name).write_bytes(f.getbuffer())
        try:
            with st.spinner("Indexing... large PDFs can take a few minutes"):
                pages, chunks = pipeline.build_index()
            st.success(f"Indexed {pages} pages into {chunks} chunks")
        except FileNotFoundError as err:
            st.warning(str(err))

    st.metric("Chunks in index", pipeline.store.count())
    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, TOP_K)
    threshold = st.slider("Min similarity", 0.0, 0.9, SCORE_THRESHOLD, 0.05)

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for src in msg.get("sources", []):
            with st.expander(f"{src['label']}  ·  similarity {src['score']}"):
                st.write(src["snippet"] + "...")

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = pipeline.ask(question, top_k=top_k, score_threshold=threshold)
            except Exception as err:
                result = {"answer": f"Something went wrong: {err}", "sources": []}
        st.markdown(result["answer"])
        for src in result["sources"]:
            with st.expander(f"{src['label']}  ·  similarity {src['score']}"):
                st.write(src["snippet"] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )