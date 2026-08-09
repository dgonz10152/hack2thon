"""The Ollama chat model, plus the retry policy every model call runs under."""

import os

import httpx
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

# Ollama Cloud drops connections mid-response (`httpx.ReadError`, often with an
# empty message), which used to abort runs already 20+ minutes in. TransportError
# is the base for ReadError, ConnectError, the timeouts and RemoteProtocolError.
TRANSIENT_ERRORS = (httpx.TransportError, ConnectionError, TimeoutError)


def resilient(runnable):
    """Retry a model call on transient network failures.

    Applied to each bound object rather than to `model` itself: wrapping the
    chat model returns a RunnableRetry, which no longer has
    `with_structured_output`.
    """
    return runnable.with_retry(
        retry_if_exception_type=TRANSIENT_ERRORS,
        stop_after_attempt=int(os.getenv("MODEL_RETRIES", "3")),
        wait_exponential_jitter=True,
    )
