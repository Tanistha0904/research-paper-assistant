"""Step 8: Streamlit UI.  Run with:  streamlit run app.py"""
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingest import chunk_pdf
from src.rag import answer_question
from src.vectorstore import VectorStore

load_dotenv()

# On Streamlit Cloud, copy secrets into environment variables so os.getenv() finds them
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # no secrets file locally, which is fine (.env is used instead)

st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
st.title("📄 Research Paper Assistant")
st.caption("Upload papers, ask questions, get answers with the paper name and page.")

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
    st.session_state.indexed = set()
    st.session_state.history = []

store: VectorStore = st.session_state.store

with st.sidebar:
    st.header("1. Upload papers")
    files = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Index papers", disabled=not files):
        with st.spinner("Reading, chunking and embedding..."):
            for f in files:
                if f.name in st.session_state.indexed:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / f.name          # keep original name for citations
                    path.write_bytes(f.getbuffer())
                    store.add(chunk_pdf(path))
                st.session_state.indexed.add(f.name)
        st.success(f"Indexed {len(store.chunks)} chunks from {len(store.sources())} paper(s).")

    st.header("2. Options")
    k = st.slider("Chunks to retrieve", 2, 10, 5)
    chosen = st.multiselect("Limit to papers (optional)", store.sources())

question = st.chat_input("Ask about your papers, e.g. 'What dataset did Paper 2 use?'")

for q, a, cites in st.session_state.history:
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        st.write(a)
        for c in cites:
            st.caption(f"Source: {c['source']} · Page: {c['page']}")

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            answer, cites = answer_question(store, question, k=k, only_sources=chosen or None)
        st.write(answer)
        for c in cites:
            with st.expander(f"Source: {c['source']} · Page: {c['page']}  (match {c['score']})"):
                st.write(c["snippet"] + "...")
    st.session_state.history.append((question, answer, cites))
