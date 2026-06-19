import ollama
from typing import Literal

# 🎯 FIX: Completely removed 'from app.config import settings' to kill the loop!

async def embed_texts(
    texts: list[str],
    *,
    task: Literal["retrieval.passage", "retrieval.query"] = "retrieval.passage",
) -> list[list[float]]:
    """Embed a list of texts via local Ollama. Matches your friend's original function layout."""
    if not texts:
        return []

    embeddings = []
    try:
        for text in texts:
            # Route text embedding safely through your local llama3 model
            response = ollama.embeddings(model="llama3", prompt=text)
            embeddings.append(response["embedding"])
    except Exception as e:
        print(f"[ERROR] Local Ollama embedding generation failed: {str(e)}")
        # Fallback to a standard empty vector list if it hits an unexpected snag
        return [[0.0] * 4096 for _ in texts]

    return embeddings


def embed_query(text: str) -> list[float]:
    """Generate a vector embedding for a single user question search string."""
    try:
        response = ollama.embeddings(model="llama3", prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"[ERROR] Local Ollama query embedding failed: {str(e)}")
        return [0.0] * 4096