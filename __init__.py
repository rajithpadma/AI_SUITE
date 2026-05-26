"""
ai_suite

A lightweight package that exposes 3 services:
- Text summarization
- Web article scraping (URL -> cleaned text)
- Speech-to-text + Neural Style Transfer
"""

from __future__ import annotations

from pathlib import Path


def get_cache_dir() -> Path:
    """Return a persistent cache directory inside the repo."""
    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_outputs_dir() -> Path:
    """Return a persistent outputs directory inside the repo."""
    outputs_dir = Path(__file__).resolve().parent / ".outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


__all__ = ["get_cache_dir", "get_outputs_dir"]

