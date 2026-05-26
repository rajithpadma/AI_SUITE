"""
ai_suite

A lightweight package that exposes 3 services:
- Text summarization
- Web article scraping (URL -> cleaned text)
- Speech-to-text + Neural Style Transfer
"""

from __future__ import annotations

from pathlib import Path


def get_models_dir() -> Path:
    """
    Directory where all model-related artifacts are stored.

    This includes:
    - Pickled caches for summarization / scraping / speech-to-text
    - Any future .pkl / .h5 models produced by the three AI modules
    """
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_cache_dir() -> Path:
    """
    Backwards-compatible alias used by older code.
    Internally we now store caches inside `models/`.
    """
    return get_models_dir()


def get_outputs_dir() -> Path:
    """Return a persistent outputs directory inside the repo."""
    outputs_dir = Path(__file__).resolve().parent / ".outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


__all__ = ["get_models_dir", "get_cache_dir", "get_outputs_dir"]


