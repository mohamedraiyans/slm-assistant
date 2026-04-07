import requests
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.memory import add_message, get_history
from app.services.rag import search_docs

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI()

OLLAMA_URL = "http://ollama:11434/api/chat"
OLLAMA_MODEL = "tinyllama"

class Prompt(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "SLM Personal Assistant API"}

@app.post("/chat")
def chat(prompt: Prompt):
    # Search docs for relevant context
    context = search_docs(prompt.message)

    system_prompt = "You are a helpful personal assistant."
    if context:
        system_prompt += f"""

Use the following context from the knowledge base to answer the user's question.
If the answer is in the context, use it. If not, answer from your own knowledge.

Context:
{context}
"""

    history = get_history()
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["ai"]})
    messages.append({"role": "user", "content": prompt.message})

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "system": system_prompt
        }, timeout=120)

        reply = response.json()["message"]["content"]

    except Exception as e:
        reply = f"Error contacting Ollama: {str(e)}"

    add_message(prompt.message, reply)

    return {
        "response": reply,
        "history": get_history()
    }

@app.get("/ui", response_class=HTMLResponse)
def chat_ui(request: Request):
    return templates.TemplateResponse(request, "index.html")