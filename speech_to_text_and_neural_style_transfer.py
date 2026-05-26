from __future__ import annotations

import hashlib
import io
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
from PIL import Image

try:
    import speech_recognition as sr
except ModuleNotFoundError:  # pragma: no cover
    sr = None  # type: ignore[assignment]

try:
    from . import get_cache_dir, get_outputs_dir
except ImportError:
    from __init__ import get_cache_dir, get_outputs_dir


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class SpeechToText:
    """
    Transcribe an uploaded WAV audio file using `speech_recognition`.

    Notes for GitHub/Streamlit:
    - This approach uses the Google Web Speech API (requires internet).
    - For best compatibility, upload `.wav` files.
    """

    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        if sr is None:  # pragma: no cover
            raise RuntimeError("speech_recognition is required. Run: pip install SpeechRecognition")
        self._cache_path = get_cache_dir() / "speech_transcripts_cache.pkl"
        self._cache = _safe_pickle_load(self._cache_path, default={}) if cache_enabled else {}

    def transcribe_wav_bytes(
        self,
        wav_bytes: bytes,
        *,
        language: str = "en-US",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        if not wav_bytes:
            raise ValueError("Audio bytes are empty.")

        cache_key = _sha256_bytes(wav_bytes) + f"|{language}"
        if self.cache_enabled:
            cached = self._cache.get(cache_key)
            if isinstance(cached, str) and cached.strip():
                return cached

        if progress_callback:
            progress_callback("Saving uploaded audio...")

        recognizer = sr.Recognizer()

        # speech_recognition needs a real file handle for AudioFile.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)

            if progress_callback:
                progress_callback("Transcribing audio (Google)...")

            text = recognizer.recognize_google(audio, language=language)
            text = (text or "").strip()

            if not text:
                raise RuntimeError("Transcription returned empty text.")

            if self.cache_enabled:
                self._cache[cache_key] = text
                _safe_pickle_dump(self._cache_path, self._cache)
            return text
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@dataclass(frozen=True)
class NSTConfig:
    # Optimizer settings
    steps: int = 80
    lr: float = 0.01
    content_weight: float = 1e4
    style_weight: float = 1e2

    # Image settings
    max_size: int = 256  # longest side after resize

    # VGG19 layer indices used by the notebook you shared
    style_layers: Tuple[int, ...] = (0, 5, 10, 19, 28)
    content_layers: Tuple[int, ...] = (21,)

    # Reproducibility
    seed: int = 0

    # Init target image
    init_mode: str = "content"  # "content" or "random"
    random_noise_std: float = 0.02


class NeuralStyleTransfer:
    """
    Improved neural style transfer (robust + correct gradient flow).

    Compared to your notebook:
    - Adds the missing `loss.backward()` required for optimization.
    - Uses LBFGS with a proper closure for stable results.
    - Adds normalization/denormalization and clamps the target tensor.
    - Caches outputs by hashing inputs + parameters.
    """

    def __init__(self, config: NSTConfig | None = None):
        self.config = config or NSTConfig()
        if torch is None or models is None or transforms is None:  # pragma: no cover
            raise RuntimeError("Torch + torchvision are required for Neural Style Transfer.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vgg = self._load_vgg19_features()

        self._cache_index_path = get_cache_dir() / "nst_cache_index.pkl"
        self._cache_index = _safe_pickle_load(self._cache_index_path, default={})

        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _load_vgg19_features(self):
        # Torchvision changed `pretrained` -> `weights`; support both.
        try:
            weights = models.VGG19_Weights.DEFAULT  # type: ignore[attr-defined]
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features  # pragma: no cover

        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        return vgg.to(self.device)

    def _resize_preserve_aspect(self, img: Image.Image, max_size: int) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        if w >= h:
            new_w = max_size
            new_h = max(1, int(h * (max_size / w)))
        else:
            new_h = max_size
            new_w = max(1, int(w * (max_size / h)))
        return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    def _image_to_tensor(self, img: Image.Image, max_size: int) -> torch.Tensor:
        img = self._resize_preserve_aspect(img.convert("RGB"), max_size)
        t = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        t = (t - self._mean) / self._std
        return t

    def _tensor_to_image(self, t: torch.Tensor) -> Image.Image:
        t = t.detach().clamp(0, 1)
        t = t.squeeze(0).cpu()
        return transforms.ToPILImage()(t)

    def _gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        # features: (1, C, H, W) -> gram: (C, C)
        _, c, h, w = features.shape
        f = features.view(c, h * w)
        gram = torch.mm(f, f.t())
        return gram / (c * h * w)

    def _get_features(self, x: torch.Tensor, layers: Tuple[int, ...]) -> dict[int, torch.Tensor]:
        # vgg is a Sequential; iterate by index.
        out: dict[int, torch.Tensor] = {}
        wanted = set(layers)
        for idx, layer in enumerate(self._vgg):
            x = layer(x)
            if idx in wanted:
                out[idx] = x
            if wanted and len(out) == len(wanted):
                break
        return out

    def stylize(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        *,
        config: NSTConfig | None = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Image.Image:
        cfg = config or self.config

        content_bytes = _pil_to_bytes(content_image)
        style_bytes = _pil_to_bytes(style_image)

        key_src = "\n".join(
            [
                _sha256_bytes(content_bytes),
                _sha256_bytes(style_bytes),
                str(cfg.steps),
                str(cfg.lr),
                str(cfg.content_weight),
                str(cfg.style_weight),
                str(cfg.max_size),
                str(cfg.style_layers),
                str(cfg.content_layers),
                str(cfg.init_mode),
                str(cfg.random_noise_std),
            ]
        )
        cache_key = _sha256_bytes(key_src.encode("utf-8"))

        cached_path = self._cache_index.get(cache_key)
        if isinstance(cached_path, str) and Path(cached_path).exists():
            return Image.open(cached_path).convert("RGB")

        torch.manual_seed(cfg.seed)

        if progress_callback:
            progress_callback("Preparing images + VGG features...")

        content = self._image_to_tensor(content_image, cfg.max_size)
        style = self._image_to_tensor(style_image, cfg.max_size)

        # Precompute features/grams (small and stable cache entries).
        content_features = self._get_features(content, cfg.content_layers)
        style_features = self._get_features(style, cfg.style_layers)
        style_grams = {layer: self._gram_matrix(style_features[layer]) for layer in cfg.style_layers}

        if cfg.init_mode == "random":
            target = torch.randn_like(content) * cfg.random_noise_std
        else:
            target = content.clone()

        target = target.requires_grad_(True)

        # Clamp target so it stays in a valid normalized range.
        min_norm = (0.0 - self._mean) / self._std
        max_norm = (1.0 - self._mean) / self._std

        optimizer = torch.optim.LBFGS([target], lr=cfg.lr, max_iter=int(cfg.steps))

        iter_calls = 0
        last_report = -1

        def closure():
            nonlocal iter_calls, last_report
            iter_calls += 1

            with torch.no_grad():
                target.clamp_(min_norm, max_norm)

            optimizer.zero_grad()

            target_features = self._get_features(target, tuple(set(cfg.style_layers) | set(cfg.content_layers)))

            # Content loss
            content_layer = cfg.content_layers[0]
            c_loss = F.mse_loss(target_features[content_layer], content_features[content_layer])

            # Style loss
            s_loss = 0.0
            for layer in cfg.style_layers:
                t_gram = self._gram_matrix(target_features[layer])
                s_loss = s_loss + F.mse_loss(t_gram, style_grams[layer])

            loss = (cfg.content_weight * c_loss) + (cfg.style_weight * s_loss)
            loss.backward()

            # Rough progress reporting for LBFGS.
            if progress_callback and cfg.steps > 5:
                bucket = int(iter_calls * 10 / cfg.steps)
                if bucket != last_report and bucket % 2 == 0:
                    last_report = bucket
                    progress_callback(f"NST running... loss={loss.item():.4f}")

            return loss

        if progress_callback:
            progress_callback("Running NST optimization...")

        optimizer.step(closure)

        # Convert back to RGB for Streamlit.
        with torch.no_grad():
            target.clamp_(min_norm, max_norm)
            target_denorm = target * self._std + self._mean

        out_img = self._tensor_to_image(target_denorm)

        # Save output image + update cache index.
        out_path = get_outputs_dir() / f"nst_{cache_key}.png"
        out_img.save(out_path)
        self._cache_index[cache_key] = str(out_path)
        _safe_pickle_dump(self._cache_index_path, self._cache_index)

        return out_img


if __name__ == "__main__":
    # Minimal smoke test.
    # (Run with your own images/audio; this is not interactive.)
    print("ai_suite: SpeechToText + NeuralStyleTransfer ready")

