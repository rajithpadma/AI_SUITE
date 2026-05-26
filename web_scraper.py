from __future__ import annotations

import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests
from bs4 import BeautifulSoup

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


@dataclass(frozen=True)
class ScraperConfig:
    user_agent: str = "Mozilla/5.0 (compatible; ai-suite/1.0)"
    timeout_s: int = 20
    min_text_chars: int = 200


class WebArticleScraper:
    """Fetch an article URL and return a cleaned (title, text) pair."""

    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        # Store scraper cache under ai_suite/models/
        self._cache_path = get_models_dir() / "web_scraper_cache.pkl"
        self._cache = _safe_pickle_load(self._cache_path, default={})

    def _clean_text(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s or "").strip()
        return s

    def fetch(self, url: str) -> Tuple[str, str]:
        url = (url or "").strip()
        if not url:
            raise ValueError("URL is empty.")

        cache_key = _sha256_str(url)
        cached = self._cache.get(cache_key)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[1]:
            return cached[0], cached[1]

        headers = {"User-Agent": self.config.user_agent}
        resp = requests.get(url, headers=headers, timeout=self.config.timeout_s)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])
        text = self._clean_text(text)

        # Fallback: try <article> tag if <p> extraction seems weak.
        if len(text) < self.config.min_text_chars:
            article = soup.find("article")
            if article:
                paras = [p.get_text(" ", strip=True) for p in article.find_all("p")]
                text2 = " ".join([p for p in paras if p])
                text2 = self._clean_text(text2)
                if len(text2) > len(text):
                    text = text2

        if len(text) < self.config.min_text_chars:
            raise ValueError("Could not extract enough text from the provided URL.")

        self._cache[cache_key] = (title, text)
        _safe_pickle_dump(self._cache_path, self._cache)
        return title, text


def fetch_article_text(url: str, *, timeout_s: int = 20) -> Tuple[str, str]:
    return WebArticleScraper(ScraperConfig(timeout_s=timeout_s)).fetch(url)


if __name__ == "__main__":
    t, tx = fetch_article_text("https://en.wikipedia.org/wiki/Artificial_intelligence", timeout_s=20)
    print("TITLE:", t)
    print("TEXT_PREVIEW:", tx[:300])

