import json
from typing import AsyncIterator

import httpx

# pyrefly: ignore [missing-import]
from app.config import settings

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided context. "
    "Cite the source document and page number when you use information from it. "
    "If the answer cannot be found in the context, say so clearly."
)


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parts.append(
            f"[{i}] (Source: {meta['filename']}, page {meta['page']})\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def _payload(question: str, chunks: list[dict], stream: bool) -> dict:
    context = _build_context(chunks)
    payload: dict = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        "temperature": 0.2,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }


async def generate(question: str, chunks: list[dict]) -> str:
    """Call Groq (non-streaming) and return the full answer text."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(_GROQ_URL, json=_payload(question, chunks, False), headers=_headers())
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


async def stream_tokens(question: str, chunks: list[dict]) -> AsyncIterator[str | int]:
    """Yield answer tokens (str) then completion_tokens count (int) as the final item."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST", _GROQ_URL, json=_payload(question, chunks, True), headers=_headers()
        ) as response:
            response.raise_for_status()
            total_tokens = 0
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "usage" in data:
                    total_tokens = data["usage"].get("total_tokens", 0)
                    continue
                try:
                    delta = data["choices"][0]["delta"].get("content")
                except (KeyError, IndexError):
                    continue
                if delta:
                    yield delta
            yield total_tokens
