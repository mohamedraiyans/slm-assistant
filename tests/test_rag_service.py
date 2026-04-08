"""
Tests for RAGService components.

DocumentLoader, DocumentChunker, DocumentRetriever, and RAGService
are each tested in isolation. OllamaClient is never called.
"""

from app.services.rag_service import (
    _normalize,
    Document,
    Chunk,
    DocumentLoader,
    DocumentChunker,
    DocumentRetriever,
    RAGService,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class FakeLLMClient:
    def __init__(self, reply: str = "mocked answer"):
        self.reply = reply
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


def make_chunks(*texts: str) -> list:
    return [Chunk(filename="test.txt", text=t) for t in texts]


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

def test_loader_returns_empty_for_missing_directory(tmp_path):
    loader = DocumentLoader(docs_dir=str(tmp_path / "nonexistent"))
    assert loader.load_all() == []


def test_loader_reads_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.txt").write_text("foo bar", encoding="utf-8")
    docs = DocumentLoader(docs_dir=str(tmp_path)).load_all()
    contents = {d.content for d in docs}
    assert "hello world" in contents and "foo bar" in contents


def test_loader_ignores_non_txt_files(tmp_path):
    (tmp_path / "data.csv").write_text("col1,col2", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
    docs = DocumentLoader(docs_dir=str(tmp_path)).load_all()
    assert len(docs) == 1


def test_loader_skips_blank_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "real.txt").write_text("content", encoding="utf-8")
    docs = DocumentLoader(docs_dir=str(tmp_path)).load_all()
    assert len(docs) == 1


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

def test_chunker_splits_multiline_document():
    docs = [Document(filename="f.txt", content="line one\nline two\nline three")]
    chunks = DocumentChunker().chunk(docs)
    assert len(chunks) == 3
    assert chunks[0].text == "line one"
    assert chunks[2].text == "line three"


def test_chunker_skips_blank_lines():
    docs = [Document(filename="f.txt", content="hello\n\n\nworld")]
    chunks = DocumentChunker().chunk(docs)
    assert len(chunks) == 2


def test_chunker_preserves_source_filename():
    docs = [Document(filename="office.txt", content="WiFi: secret123")]
    chunks = DocumentChunker().chunk(docs)
    assert chunks[0].filename == "office.txt"


def test_chunker_returns_empty_for_no_documents():
    assert DocumentChunker().chunk([]) == []



# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_strips_trailing_punctuation():
    assert "location" in _normalize("Location:")

def test_normalize_lowercases_words():
    assert "wifi" in _normalize("WiFi")

def test_normalize_handles_mixed_punctuation():
    assert _normalize("Hello, world!") == {"hello", "world"}


# ---------------------------------------------------------------------------
# DocumentRetriever
# ---------------------------------------------------------------------------

def test_retriever_returns_empty_for_no_chunks():
    assert DocumentRetriever().retrieve("anything", []) == []


def test_retriever_finds_exact_matching_chunk():
    chunks = make_chunks(
        "Location: Kungsgatan 12, Stockholm, Sweden",
        "WiFi Password: Conversy2024",
        "Office Hours: Monday to Friday, 9:00 AM to 6:00 PM",
    )
    result = DocumentRetriever().retrieve("Where is the office location?", chunks)
    assert result[0].text == "Location: Kungsgatan 12, Stockholm, Sweden"


def test_retriever_excludes_zero_overlap_chunks():
    chunks = make_chunks("quantum physics", "black holes")
    result = DocumentRetriever().retrieve("coffee and tea", chunks)
    assert result == []


def test_retriever_prefers_shorter_focused_chunks():
    # Short precise chunk vs long chunk with same keyword count
    chunks = [
        Chunk(filename="f.txt", text="WiFi Password: Conversy2024"),
        Chunk(filename="f.txt", text="WiFi is available in the office and the password can be found on the notice board near the reception"),
    ]
    result = DocumentRetriever().retrieve("WiFi password", chunks)
    assert result[0].text == "WiFi Password: Conversy2024"


def test_retriever_respects_top_k():
    chunks = make_chunks("alpha beta", "alpha gamma", "alpha delta", "alpha epsilon")
    result = DocumentRetriever().retrieve("alpha", chunks, top_k=2)
    assert len(result) <= 2


# ---------------------------------------------------------------------------
# RAGService
# ---------------------------------------------------------------------------

def test_rag_passes_relevant_chunk_text_to_llm():
    fake_llm = FakeLLMClient()
    rag = RAGService(llm_client=fake_llm)
    rag._chunks = make_chunks("WiFi Password: Conversy2024")

    rag.generate_answer("WiFi password")

    assert "WiFi Password: Conversy2024" in fake_llm.last_prompt


def test_rag_returns_llm_reply():
    rag = RAGService(llm_client=FakeLLMClient(reply="Conversy2024"))
    rag._chunks = make_chunks("WiFi Password: Conversy2024")
    assert rag.generate_answer("WiFi") == "Conversy2024"


def test_rag_handles_no_chunks_gracefully():
    fake_llm = FakeLLMClient(reply="I don't have that information")
    rag = RAGService(llm_client=fake_llm)
    rag._chunks = []
    result = rag.generate_answer("anything")
    assert result == "I don't have that information"
    assert "No relevant information found" in fake_llm.last_prompt


def test_rag_load_documents_populates_chunks(tmp_path):
    (tmp_path / "facts.txt").write_text("line one\nline two", encoding="utf-8")
    rag = RAGService(
        loader=DocumentLoader(docs_dir=str(tmp_path)),
        llm_client=FakeLLMClient(),
    )
    rag.load_documents()
    assert len(rag._chunks) == 2
