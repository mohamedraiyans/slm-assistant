from fastapi import APIRouter
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService
from app.services.memory_service import MemoryService

router = APIRouter()

rag_service = RAGService()
memory_service = MemoryService()
chat_service = ChatService(rag_service, memory_service)

@router.get("/")
def root():
    return {"message": "SLM Assistant is running"}

@router.post("/chat")
def chat(input_text: str):
    response = chat_service.handle_chat(input_text)
    return {"response": response}
