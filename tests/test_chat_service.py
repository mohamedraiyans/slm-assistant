from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService

def test_chat_flow():
    chat = ChatService(RAGService(), MemoryService())
    response = chat.handle_chat("Hi")
    assert isinstance(response, str)
