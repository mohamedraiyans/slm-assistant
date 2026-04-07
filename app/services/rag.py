import os
from pathlib import Path

DOCS_DIR = Path("/app/data/docs")

def load_docs():
    chunks = []
    for file in DOCS_DIR.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        # Split into chunks of ~500 chars with overlap
        words = text.split()
        size, overlap = 100, 20
        for i in range(0, len(words), size - overlap):
            chunk = " ".join(words[i:i + size])
            if chunk:
                chunks.append({"source": file.name, "text": chunk})
    return chunks

def search_docs(query: str, top_n: int = 3) -> str:
    chunks = load_docs()
    query_words = set(query.lower().split())

    scored = []
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split())
        score = len(query_words & chunk_words)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    if not top:
        return ""

    context = "\n\n".join(
        f"[{c['source']}]: {c['text']}" for _, c in top
    )
    return context