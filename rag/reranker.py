"""Cohere cross-encoder reranker client."""

import httpx

# pyrefly: ignore [missing-import]
from app.config import settings

_COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
_MODEL = "rerank-v3.5"


async def rerank(question: str, chunks: list[dict]) -> list[dict]:
    """Rerank chunks using Cohere cross-encoder; returns top_k_rerank results."""
    if not chunks:
        return []

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            _COHERE_RERANK_URL,
            headers={
                "Authorization": f"Bearer {settings.cohere_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _MODEL,
                "query": question,
                "documents": [c["text"] for c in chunks],
                "top_n": settings.top_k_rerank,
            },
        )
        response.raise_for_status()

    results = response.json()["results"]
    return [
        {**chunks[r["index"]], "score": r["relevance_score"]}
        for r in results
    ]
