import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# 🎯 FIX: Removed 'from app.config import settings' to stop the crash!
from rag.chunker import Chunk


def _index_path() -> Path:
    # 🎯 FIX: Automatically saves your BM25 search file into an 'instance' folder in your chatbot project
    return Path("instance/bm25.pkl")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _load() -> tuple[BM25Okapi | None, list[dict]]:
    path = _index_path()
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None, []


def _save(index: BM25Okapi, corpus: list[dict]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((index, corpus), f)


def _build(corpus: list[dict]) -> BM25Okapi:
    return BM25Okapi([_tokenize(doc["text"]) for doc in corpus])


def add_chunks(chunks: list[Chunk]) -> None:
    _, corpus = _load()
    filenames = {c.filename for c in chunks}
    corpus = [d for d in corpus if d["filename"] not in filenames]
    corpus.extend(
        {"text": c.text, "filename": c.filename, "page": c.page, "section": c.section}
        for c in chunks
    )
    _save(_build(corpus), corpus)


def query(question: str, top_k: int) -> list[dict]:
    index, corpus = _load()
    if index is None or not corpus:
        return []

    scores = index.get_scores(_tokenize(question))
    ranked = sorted(
        ((i, float(scores[i])) for i in range(len(scores))),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        {
            "text": corpus[i]["text"],
            "metadata": {
                "filename": corpus[i]["filename"],
                "page": corpus[i]["page"],
                "section": corpus[i]["section"],
            },
            "score": score,
        }
        for i, score in ranked
    ]


def remove_document(filename: str) -> None:
    _, corpus = _load()
    corpus = [d for d in corpus if d["filename"] != filename]
    if corpus:
        _save(_build(corpus), corpus)
    else:
        path = _index_path()
        if path.exists():
            path.unlink()