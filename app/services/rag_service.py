"""
RAGService — Retrieval-Augmented Generation using local .txt documents.

Documents are split into line-level chunks at load time.
Retrieval scores chunks by meaningful keyword overlap (stopwords excluded).
Top-scoring chunks plus their immediate neighbours are returned directly
as the answer — no LLM call needed for simple factual queries.

The OllamaClient is kept for future use when generative answers are needed,
but the default RAGService.generate_answer() returns retrieved facts directly.
"""

from pathlib import Path
from dataclasses import dataclass
import os
import json

import httpx


@dataclass
class Document:
    filename: str
    content: str


@dataclass
class Chunk:
    filename: str
    text: str
    index: int


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
                    chunks.append(Chunk(
                        filename=doc.filename,
                        text=line,
                        index=len(chunks),
                    ))
        return chunks


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "not",
}


def _normalize(text: str) -> set:
    """
    Lowercase, strip punctuation, and remove stopwords so that
    only meaningful keywords are used for scoring.
    Example: 'Location:' -> 'location', 'the'/'is'/'where' are excluded.
    """
    words = {word.strip(",:?!.()") for word in text.lower().split()}
    return words - _STOPWORDS


class DocumentRetriever:
    """
    Scores chunks by meaningful keyword overlap with the query.

    Stopwords are excluded so common words don't pollute scores.
    Each top match also pulls in its next line (neighbour) from the
    same file, so that key:value pairs like Company/Location are
    always retrieved together.
    """

    def retrieve(self, query: str, chunks: list, top_k: int = 2) -> list:
        if not chunks:
            return []

        query_words = _normalize(query)
        if not query_words:
            return []

        scored = []
        for chunk in chunks:
            chunk_words = _normalize(chunk.text)
            overlap = len(query_words & chunk_words)
            if overlap == 0:
                continue
            score = overlap / (1 + len(chunk.text.split()) * 0.1)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        top_chunks = [r.chunk for r in scored[:top_k]]

        result_indices = {c.index for c in top_chunks}
        for chunk in top_chunks:
            next_index = chunk.index + 1
            if next_index < len(chunks) and chunks[next_index].filename == chunk.filename:
                result_indices.add(next_index)

        return [chunks[i] for i in sorted(result_indices)]


class OllamaClient:
    """
    HTTP client for the Ollama /api/generate endpoint.
    Kept for future use when generative answers are needed.
    Uses streaming so the connection stays alive during slow CPU inference.
    """

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model or os.environ.get("OLLAMA_MODEL", "tinyllama")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        tokens = []
        with httpx.Client(timeout=httpx.Timeout(connect=60, read=600, write=60, pool=60)) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        tokens.append(chunk.get("response", ""))
                        if chunk.get("done"):
                            break
        return "".join(tokens).strip()


class RAGService:
    """
    Orchestrates chunking, retrieval, and answer generation.

    For factual queries, the retrieved chunks ARE the answer — no LLM
    call is needed. This gives instant responses regardless of hardware.

    The retrieved lines are formatted as a clean readable answer.
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
        """
        Retrieve relevant chunks and return them directly as the answer.
        No LLM call — instant response, works on any hardware.
        """
        relevant_chunks = self.retriever.retrieve(question, self._chunks)
        if not relevant_chunks:
            return "I don't have that information in the available documents."
        return "\n".join(chunk.text for chunk in relevant_chunks)