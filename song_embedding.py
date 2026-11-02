"""Embed an iTunes preview with MERT, without a system ffmpeg installation.

The public iTunes Search API supplies the preview URL.  PyAV decodes the
audio bytes in-process, so the script does not shell out to ffmpeg.
"""

from __future__ import annotations

import io
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

import av
import numpy as np
import requests
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MODEL_NAME = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24_000
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def normalize(text: str) -> str:
    """Make a title or artist name safe for a loose, human-style match."""
    text = text.lower().replace("&", "and")
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return " ".join(text.split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def find_itunes_song(title: str, artist: str) -> dict[str, Any]:
    """Return the best iTunes song result with an available preview.

    A strict post-search check prevents a similarly titled song from silently
    being embedded.  Some catalogue entries have no preview; callers should
    record that as a skipped row rather than substitute a different recording.
    """
    response = requests.get(
        ITUNES_SEARCH_URL,
        params={"term": f"{title} {artist}", "entity": "song", "limit": 25},
        timeout=30,
    )
    response.raise_for_status()
    candidates = [row for row in response.json().get("results", []) if row.get("previewUrl")]
    if not candidates:
        raise LookupError("No iTunes song result with an audio preview.")

    def score(row: dict[str, Any]) -> float:
        return 0.75 * similarity(title, row.get("trackName", "")) + 0.25 * similarity(
            artist, row.get("artistName", "")
        )

    best = max(candidates, key=score)
    if score(best) < 0.72:
        raise LookupError(
            "No confident iTunes match; best result was "
            f"{best.get('trackName')!r} by {best.get('artistName')!r}."
        )
    return best


def download_preview(preview_url: str) -> bytes:
    response = requests.get(preview_url, timeout=60)
    response.raise_for_status()
    return response.content


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode and resample arbitrary preview audio to mono 24 kHz float PCM."""
    container = av.open(io.BytesIO(audio_bytes))
    try:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().reshape(-1))
        if not chunks:
            raise ValueError("The preview contained no decoded audio samples.")
        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return audio, SAMPLE_RATE
    finally:
        container.close()


@lru_cache(maxsize=1)
def load_model() -> tuple[Wav2Vec2FeatureExtractor, AutoModel]:
    """Load MERT once per batch process (the first run downloads model weights)."""
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    return extractor, model


def generate_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return a single mean-pooled MERT embedding for one preview."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, received {sample_rate} Hz.")
    extractor, model = load_model()
    inputs = extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
    return hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def embed_song(title: str, artist: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Look up, download, decode, and embed a song preview."""
    match = find_itunes_song(title, artist)
    audio, sample_rate = load_audio_from_bytes(download_preview(match["previewUrl"]))
    return generate_embedding(audio, sample_rate), match
