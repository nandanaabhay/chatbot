"""Simple persistent cache for question → answer mapping, keyed by question hash."""

import hashlib
import json
import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
from app.config import settings

logger = logging.getLogger(__name__)


def _cache_path() -> Path:
    return Path(settings.chroma_persist_dir).parent / "question_cache.json"


def _key(question: str) -> str:
    return hashlib.sha256(question.lower().strip().encode()).hexdigest()


def get(question: str) -> dict | None:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get(_key(question))
    except Exception:
        logger.warning("question_cache read failed", exc_info=True)
        return None


def set(question: str, value: dict) -> None:
    path = _cache_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        data = {}
    data[_key(question)] = value
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        logger.warning("question_cache write failed", exc_info=True)
