from app.services.rag_service import RAGService

def test_rag_returns_string():
    rag = RAGService()
    result = rag.generate_answer("Hello")
    assert isinstance(result, str)
    assert "Hello" in result
