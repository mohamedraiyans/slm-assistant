from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="SLM Assistant",
    description="A local RAG-powered assistant backed by Ollama.",
    version="1.0.0",
)

app.include_router(router)
