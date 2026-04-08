"""
RAGService — Retrieval-Augmented Generation using local .txt documents.

Documents are split into line-level chunks at load time.
Retrieval scores chunks by normalized keyword overlap, then passes
only the most relevant context to the language model.
"""

from pathlib import Path
from dataclasses import dataclass
import os

import httpx


@dataclass
class Document:
    filename: str
    content: str


@dataclass
class Chunk:
    filename: str
    text: str


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class DocumentLoader:
    """Loads plain-text documents from a directory."""

    def __init__(self, docs_dir: str = "data/docs"):
        self.docs_dir = Path(docs_dir)

    def load_all(self) -> list:
        if not self.docs_dir.exists():
            return []
        documents = []
        for path in sorted(self.docs_dir.glob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                documents.append(Document(filename=path.name, content=content))
        return documents


class DocumentChunker:
    """Splits documents into line-level chunks for fine-grained retrieval."""

    def chunk(self, documents: list) -> list:
        chunks = []
        for doc in documents:
            for line in doc.content.splitlines():
                line = line.strip()
                if line:
                    chunks.append(Chunk(filename=doc.filename, text=line))
        return chunks


def _normalize(text: str) -> set:
    """
    Lowercase and strip punctuation from each word so that
    'Location:' matches the query word 'location'.
    """
    return {word.strip(",:?!.") for word in text.lower().split()}


class DocumentRetriever:
    """
    Scores chunks by normalized keyword overlap with the query.

    Scoring: number of matching keywords divided by chunk length,
    so shorter and more focused chunks rank above long verbose ones
    with the same number of matching words.
    """

    def retrieve(self, query: str, chunks: list, top_k: int = 3) -> list:
        if not chunks:
            return []

        query_words = _normalize(query)
        scored = []
        for chunk in chunks:
            chunk_words = _normalize(chunk.text)
            overlap = len(query_words & chunk_words)
            if overlap == 0:
                continue
            score = overlap / (1 + len(chunk.text.split()) * 0.1)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return [r.chunk for r in scored[:top_k]]


class OllamaClient:
    """Thin HTTP client for the Ollama /api/generate endpoint."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()["response"].strip()


class RAGService:
    """
    Orchestrates chunking, retrieval, and answer generation.

    Given a question, it:
      1. Retrieves the most relevant chunks (not whole documents).
      2. Builds a focused, context-grounded prompt.
      3. Sends the prompt to the language model.
    """

    def __init__(self, loader=None, chunker=None, retriever=None, llm_client=None):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.retriever = retriever or DocumentRetriever()
        self.llm_client = llm_client or OllamaClient()
        self._chunks: list = []

    def load_documents(self) -> None:
        """Load and chunk documents from disk. Call once at startup."""
        documents = self.loader.load_all()
        self._chunks = self.chunker.chunk(documents)

    def generate_answer(self, question: str) -> str:
        relevant_chunks = self.retriever.retrieve(question, self._chunks)
        context = self._build_context(relevant_chunks)
        prompt = self._build_prompt(question, context)
        return self.llm_client.generate(prompt)

    def _build_context(self, chunks: list) -> str:
        if not chunks:
            return "No relevant information found."
        return "\n".join(
            f"- {chunk.text}" for chunk in chunks
        )

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "You are a factual assistant. "
            "You must answer using ONLY the facts listed below. "
            "Do not explain, do not speculate, do not add anything else. "
            "Just state the fact directly.\n\n"
            f"Facts:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer (one sentence, facts only):"
        )
