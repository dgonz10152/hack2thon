import os

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama


def build_model() -> ChatOllama:
    kwargs: dict = {
        "model": os.getenv("OLLAMA_MODEL", "minimax-m3:cloud"),
        "reasoning": False,
    }
    if base_url := os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = base_url
    if api_key := os.getenv("OLLAMA_API_KEY"):
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    return ChatOllama(**kwargs)


model = build_model()
