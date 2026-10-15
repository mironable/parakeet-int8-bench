"""Log-mel features in plain NumPy, matching NeMo's AudioToMelSpectrogramPreprocessor (eval mode).

The mel filterbank and STFT window are not recomputed here: export_onnx.py copies the exact
tensors from the NeMo model into preprocessor.npz, so the only logic to get right is framing,
pre-emphasis, the power spectrum, log and per-feature normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STD_EPS = 1e-5  # NeMo's CONSTANT in normalize_batch


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    win_length: int
    preemph: float
    mag_power: float
    log_zero_guard: float
    normalize: str
    pad_value: float
    fb: np.ndarray  # (n_mels, n_fft // 2 + 1)
    window: np.ndarray  # (win_length,)

    @classmethod
    def load(cls, path: str | Path) -> FeatureConfig:
        z = np.load(path, allow_pickle=False)
        return cls(
            sample_rate=int(z["sample_rate"]),
            n_fft=int(z["n_fft"]),
            hop_length=int(z["hop_length"]),
            win_length=int(z["win_length"]),
            preemph=float(z["preemph"]),
            mag_power=float(z["mag_power"]),
            log_zero_guard=float(z["log_zero_guard"]),
            normalize=str(z["normalize"]),
            pad_value=float(z["pad_value"]),
            fb=z["fb"].astype(np.float32),
            window=z["window"].astype(np.float32),
        )


class LogMel:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        # torch.stft centres a short window inside n_fft with zeros on both sides.
        left = (cfg.n_fft - cfg.win_length) // 2
        padded = np.zeros(cfg.n_fft, dtype=np.float32)
        padded[left : left + cfg.win_length] = cfg.window
        self._window = padded

    def num_frames(self, num_samples: int) -> int:
        """Valid frame count, as NeMo's FilterbankFeatures.get_seq_len computes it.

        The STFT yields one frame more than this; NeMo masks that last frame to pad_value and
        leaves it out of the normalisation statistics.
        """
        c = self.cfg
        return (num_samples + 2 * (c.n_fft // 2) - c.n_fft) // c.hop_length

    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """audio: mono float32 at cfg.sample_rate. Returns ((n_mels, frames) float32, valid frame count)."""
        c = self.cfg
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {x.shape}")
        n_valid = self.num_frames(x.size)
        if n_valid < 1:
            raise ValueError(f"audio too short: {x.size} samples")

        if c.preemph:
            x = np.concatenate([x[:1], x[1:] - c.preemph * x[:-1]])

        # center=True with constant padding, as NeMo calls torch.stft
        half = c.n_fft // 2
        x = np.pad(x, (half, half))
        n_frames = 1 + (x.size - c.n_fft) // c.hop_length
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, c.n_fft),
            strides=(x.strides[0] * c.hop_length, x.strides[0]),
            writeable=False,
        )
        spec = np.fft.rfft(frames * self._window, n=c.n_fft, axis=-1)
        power = np.abs(spec).astype(np.float32)
        if c.mag_power != 1.0:
            power = power**c.mag_power

        mel = c.fb @ power.T  # (n_mels, frames)
        mel = np.log(mel + c.log_zero_guard)

        if c.normalize == "per_feature":
            valid = mel[:, :n_valid]
            mean = valid.mean(axis=1, keepdims=True)
            denom = max(n_valid - 1, 1)
            std = np.sqrt(((valid - mean) ** 2).sum(axis=1, keepdims=True) / denom)
            std = np.nan_to_num(std) + _STD_EPS
            mel = (mel - mean) / std
        elif c.normalize not in ("", "none", "None"):
            raise NotImplementedError(f"normalize={c.normalize!r}")

        mel[:, n_valid:] = c.pad_value
        return mel.astype(np.float32), n_valid
