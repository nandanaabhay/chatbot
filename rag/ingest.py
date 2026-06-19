import hashlib
from pathlib import Path
import asyncio  # 🎯 Added to bridge his async code with your standard Flask app

# 🎯 FIX: Corrected your folder pathing imports (removed 'app.')
from rag import bm25_store
from rag.chunker import chunk_pdf
from rag.chroma_store import add_chunks, find_by_hash
from rag.embeddings import embed_texts


def ingest_pdf(path: str) -> int:
    """Chunk, embed, and store a PDF. Returns the number of chunks indexed.

    Skips embedding if an identical file (same SHA-256) is already in the store.
    """
    # Convert string path to Path object safely
    path_obj = Path(path)
    
    content_hash = hashlib.sha256(path_obj.read_bytes()).hexdigest()

    existing = find_by_hash(content_hash)
    if existing > 0:
        return existing

    chunks = chunk_pdf(path_obj)
    if not chunks:
        return 0

    # 🎯 FIX: Wrapped his async embed call so your standard Flask app can execute it smoothly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    vectors = loop.run_until_complete(embed_texts([c.text for c in chunks]))
    loop.close()

    add_chunks(chunks, vectors, content_hash=content_hash)
    bm25_store.add_chunks(chunks)
    return len(chunks)