import asyncio

# 🎯 FIX: Corrected your folder pathing imports (removed 'app.')
from rag import bm25_store
from rag.chroma_store import query_chunks
from rag.embeddings import embed_texts

_RRF_K = 60  # standard constant — dampens the impact of rank differences

# 🎯 FIX: Hardcode how many search results to grab from each store
TOP_K_DENSE = 5
TOP_K_BM25 = 5


def _rrf(ranked_lists: list[list[dict]]) -> list[dict]:
    """Reciprocal Rank Fusion: score each chunk by 1/(k + rank) summed across lists."""
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, 1):
            key = chunk["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            if key not in chunks:
                chunks[key] = chunk

    return sorted(chunks.values(), key=lambda c: scores[c["text"]], reverse=True)


async def retrieve(question: str) -> list[dict]:
    """Hybrid retrieval: dense + BM25 → RRF fusion."""
    # Pass 'task' to match your friend's exact parameters
    vectors = await embed_texts([question], task="retrieval.query")
    loop = asyncio.get_running_loop()

    # 🎯 FIX: Replaced settings file calls with our hardcoded TOP_K values
    dense_results, bm25_results = await asyncio.gather(
        loop.run_in_executor(None, query_chunks, vectors[0], TOP_K_DENSE),
        loop.run_in_executor(None, bm25_store.query, question, TOP_K_BM25),
    )

    return _rrf([dense_results, bm25_results])


# 🎯 FIX: Added this synchronous wrapper function so it connects perfectly to your Flask app.py!
def retrieve_docs(question: str) -> list[dict]:
    """Bridge function to run the asynchronous retriever inside standard Flask routes."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(retrieve(question))
        loop.close()
        return results
    except Exception as e:
        print(f"[ERROR] Hybrid retrieval failed: {str(e)}")
        return []