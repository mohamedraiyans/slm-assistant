from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="SLM Assistant")

app.include_router(router)
