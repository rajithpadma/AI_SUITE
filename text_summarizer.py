from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover
    pipeline = None  # type: ignore[assignment]

try:
    from . import get_models_dir
except ImportError:
    from __init__ import get_models_dir


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_pickle_load(path: Path, default):
    try:
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return default


def _safe_pickle_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


@dataclass
class SummarizerConfig:
    model_name: str = "facebook/bart-large-cnn"
    max_length: int = 225
    min_length: int = 70
    chunk_size_words: int = 512


class TextSummarizer:
    """
    Summarize long text by splitting into chunks and summarizing each chunk.
    Caches final summaries to a pickle file so repeated runs are faster.
    """

    def __init__(self, config: SummarizerConfig | None = None):
        self.config = config or SummarizerConfig()
        # Store cached summaries under ai_suite/models/
        self._cache_path = get_models_dir() / "summarizer_cache.pkl"
        self._cache = _safe_pickle_load(self._cache_path, default={})

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_pipeline(model_name: str):
        if torch is None or pipeline is None:  # pragma: no cover
            raise RuntimeError(
                "Transformers + Torch are required for summarization. Run: pip install torch torchvision transformers"
            )

        device = 0 if torch.cuda.is_available() else -1
        # pipeline caches models internally, but we still add our own summary cache.
        return pipeline("summarization", model=model_name, device=device)

    def _split_text(self, text: str, *, chunk_size_words: int) -> list[str]:
        # Word-based chunking (keeps implementation simple and stable for GitHub).
        # Note: this is not token-perfect, but works well in practice.
        words = text.split()
        size = max(50, int(chunk_size_words))
        return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]

    def summarize(
        self,
        text: str,
        *,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        chunk_size_words: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text is empty.")

        max_length = int(max_length or self.config.max_length)
        min_length = int(min_length or self.config.min_length)
        chunk_size_words = int(chunk_size_words or self.config.chunk_size_words)

        # Cache key based on content + parameters.
        key_src = "\n".join([_sha256_str(text), str(max_length), str(min_length), str(chunk_size_words), self.config.model_name])
        cache_key = _sha256_str(key_src)

        cached = self._cache.get(cache_key)
        if isinstance(cached, str) and cached.strip():
            return cached

        try:
            chunks = self._split_text(text, chunk_size_words=chunk_size_words)
            if not chunks:
                raise ValueError("Failed to split text into chunks.")

            summarizer = self._get_pipeline(self.config.model_name)
            summaries: list[str] = []

            for i, chunk in enumerate(chunks):
                if progress_callback:
                    progress_callback(f"Summarizing chunk {i + 1}/{len(chunks)}...")
                result = summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                )
                summaries.append(result[0]["summary_text"].strip())

            final_summary = " ".join([s for s in summaries if s]).strip()
            if not final_summary:
                final_summary = text[:max_length]  # fallback

            self._cache[cache_key] = final_summary
            _safe_pickle_dump(self._cache_path, self._cache)
            return final_summary
        finally:
            pass


def summarize_text(
    text: str,
    *,
    model_name: str = "facebook/bart-large-cnn",
    max_length: int = 225,
    min_length: int = 70,
    chunk_size_words: int = 512,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    summarizer = TextSummarizer(SummarizerConfig(model_name=model_name))
    return summarizer.summarize(
        text,
        max_length=max_length,
        min_length=min_length,
        chunk_size_words=chunk_size_words,
        progress_callback=progress_callback,
    )


if __name__ == "__main__":
    # Simple local smoke test (does not require Streamlit).
    sample = "Artificial intelligence is transforming many industries by enabling machines to perform tasks that normally require human intelligence."
    s = summarize_text(sample)
    print(s)

