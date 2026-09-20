"""Step 6 + 7: build the prompt from retrieved chunks, ask the LLM, return answer + citations."""
import os

from .vectorstore import VectorStore

SYSTEM_RULES = """You are a research assistant. Answer the question using ONLY the context below.
Each context block is labelled [Source: <paper>, Page: <n>].
- If the answer is not in the context, say: "I couldn't find this in the uploaded papers."
- Be concise and name the paper(s) your answer comes from.
- Never invent facts, numbers or dataset names."""


def build_prompt(question: str, hits) -> str:
    context = "\n\n".join(
        f"[Source: {c.source}, Page: {c.page}]\n{c.text}" for c, _ in hits
    )
    return f"{SYSTEM_RULES}\n\n=== CONTEXT ===\n{context}\n\n=== QUESTION ===\n{question}\n\n=== ANSWER ==="


def call_llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()

    if provider == "groq":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    if provider == "ollama":
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"]

    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    # default: Gemini
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
    )
    return resp.text


def answer_question(store: VectorStore, question: str, k: int = 5, only_sources=None):
    hits = store.search(question, k=k, only_sources=only_sources)
    if not hits:
        return "No documents indexed yet (or no match in the selected papers).", []
    answer = call_llm(build_prompt(question, hits))
    # de-duplicated citation list, in retrieval order
    seen, citations = set(), []
    for chunk, score in hits:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append({"source": chunk.source, "page": chunk.page, "score": round(score, 3),
                              "snippet": chunk.text[:300]})
    return answer, citations
