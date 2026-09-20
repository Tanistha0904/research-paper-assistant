# Research Paper Assistant (RAG)

Upload research-paper PDFs, ask questions, get answers with **paper name + page** citations.

PDF → text extraction (pypdf) → chunking (LangChain splitter) → embeddings (MiniLM) →
FAISS index → retriever → Gemini/OpenAI → answer + citation

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # add your GEMINI_API_KEY
streamlit run app.py
```
