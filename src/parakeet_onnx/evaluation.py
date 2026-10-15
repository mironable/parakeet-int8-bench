"""Shared helpers for data prep, benchmarking and reporting (dev dependencies only)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DATASET_LANG = {"librispeech": "en", "edacc": "en", "fleurs_de": "de"}


@dataclass
class Clip:
    id: str
    dataset: str
    audio: str
    text: str
    duration: float
    meta: dict

    @classmethod
    def from_json(cls, line: str) -> Clip:
        d = json.loads(line)
        return cls(d["id"], d["dataset"], d["audio"], d["text"], d["duration"], d.get("meta", {}))

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def read_manifest(path: Path) -> list[Clip]:
    return [Clip.from_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(path: Path, clips: Iterable[Clip]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for clip in clips:
            f.write(clip.to_json() + "\n")
            n += 1
    return n


def load_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import soxr

        audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
    return audio


@cache
def _normalizer(lang: str):
    if lang == "en":
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()
    from whisper_normalizer.basic import BasicTextNormalizer

    return BasicTextNormalizer()


def normalize(text: str, lang: str) -> str:
    return " ".join(_normalizer(lang)(text).split())


def wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus WER over pairs with a non-empty reference."""
    import jiwer

    pairs = [(r, h) for r, h in zip(refs, hyps, strict=True) if r]
    if not pairs:
        return float("nan")
    r, h = zip(*pairs, strict=True)
    return float(jiwer.wer(list(r), list(h)))


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
