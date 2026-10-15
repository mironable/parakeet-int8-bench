"""Parakeet TDT inference on ONNX Runtime: NumPy features, ONNX encoder, ONNX decoder+joint, greedy TDT.

No NeMo or PyTorch at runtime, so this is the path that would run on a phone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .decode import Hypothesis, greedy_tdt
from .features import FeatureConfig, LogMel

_ORT_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


@dataclass
class Result:
    text: str
    tokens: list[int]
    feat_ms: float
    enc_ms: float
    dec_ms: float
    decoder_steps: int

    @property
    def total_ms(self) -> float:
        return self.feat_ms + self.enc_ms + self.dec_ms


def _io(session: ort.InferenceSession, kind: str) -> dict[str, ort.NodeArg]:
    args = session.get_inputs() if kind == "in" else session.get_outputs()
    return {a.name: a for a in args}


def _pick(names: list[str], wanted: str, fallback_index: int) -> str:
    return wanted if wanted in names else names[fallback_index]


class OnnxParakeet:
    def __init__(self, model_dir: str | Path, assets_dir: str | Path, threads: int = 4):
        model_dir, assets_dir = Path(model_dir), Path(assets_dir)
        meta = json.loads((assets_dir / "model_config.json").read_text())
        self.blank_id: int = meta["blank_id"]
        self.durations: list[int] = meta["durations"]
        self.max_symbols: int = meta.get("max_symbols", 10)
        self.features = LogMel(FeatureConfig.load(assets_dir / "preprocessor.npz"))
        self._detok = _Detokenizer(assets_dir)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(model_dir / "encoder.onnx"), opts, providers=providers)
        self.decoder = ort.InferenceSession(str(model_dir / "decoder_joint.onnx"), opts, providers=providers)
        self._bind_io()

    def _bind_io(self) -> None:
        ein = _io(self.encoder, "in")
        ein_names = list(ein)
        self._enc_audio = _pick(ein_names, "audio_signal", 0)
        self._enc_len = _pick(ein_names, "length", 1)
        self._enc_len_dtype = _ORT_TO_NP[ein[self._enc_len].type]
        self._enc_audio_dtype = _ORT_TO_NP[ein[self._enc_audio].type]
        eout = list(_io(self.encoder, "out"))
        self._enc_out = _pick(eout, "outputs", 0)
        self._enc_out_len = _pick(eout, "encoded_lengths", 1)

        din = _io(self.decoder, "in")
        dnames = list(din)
        self._d_enc = _pick(dnames, "encoder_outputs", 0)
        self._d_tgt = _pick(dnames, "targets", 1)
        self._d_tgt_len = _pick(dnames, "target_length", 2)
        self._d_s1 = _pick(dnames, "input_states_1", 3)
        self._d_s2 = _pick(dnames, "input_states_2", 4)
        self._tgt_dtype = _ORT_TO_NP[din[self._d_tgt].type]
        self._tgt_len_dtype = _ORT_TO_NP[din[self._d_tgt_len].type]
        self._d_enc_dtype = _ORT_TO_NP[din[self._d_enc].type]
        dout = list(_io(self.decoder, "out"))
        self._d_logits = _pick(dout, "outputs", 0)
        self._d_os1 = _pick(dout, "output_states_1", 2)
        self._d_os2 = _pick(dout, "output_states_2", 3)

        def state_zeros(arg: ort.NodeArg) -> np.ndarray:
            layers, _, hidden = arg.shape  # (layers, batch, hidden); batch is symbolic
            return np.zeros((int(layers), 1, int(hidden)), dtype=_ORT_TO_NP[arg.type])

        self._init_state = (state_zeros(din[self._d_s1]), state_zeros(din[self._d_s2]))

    def encode(self, feats: np.ndarray, n_valid: int) -> np.ndarray:
        """feats: (n_mels, frames) with n_valid real frames -> encoder frames (T, D)."""
        out, out_len = self.encoder.run(
            [self._enc_out, self._enc_out_len],
            {
                self._enc_audio: feats[None].astype(self._enc_audio_dtype),
                self._enc_len: np.array([n_valid], dtype=self._enc_len_dtype),
            },
        )
        return out[0, :, : int(out_len[0])].T  # (D, T) -> (T, D)

    def _step(self, frame: np.ndarray, last_token: int, state: tuple[np.ndarray, np.ndarray]):
        logits, s1, s2 = self.decoder.run(
            [self._d_logits, self._d_os1, self._d_os2],
            {
                self._d_enc: frame[None, :, None].astype(self._d_enc_dtype),
                self._d_tgt: np.array([[last_token]], dtype=self._tgt_dtype),
                self._d_tgt_len: np.array([1], dtype=self._tgt_len_dtype),
                self._d_s1: state[0],
                self._d_s2: state[1],
            },
        )
        return logits.reshape(-1), (s1, s2)

    def decode(self, enc: np.ndarray) -> Hypothesis:
        return greedy_tdt(enc, self._step, self._init_state, self.blank_id, self.durations, self.max_symbols)

    def transcribe(self, audio: np.ndarray) -> Result:
        t0 = time.perf_counter()
        feats, n_valid = self.features(audio)
        t1 = time.perf_counter()
        enc = self.encode(feats, n_valid)
        t2 = time.perf_counter()
        hyp = self.decode(enc)
        t3 = time.perf_counter()
        return Result(
            text=self._detok(hyp.tokens),
            tokens=hyp.tokens,
            feat_ms=(t1 - t0) * 1e3,
            enc_ms=(t2 - t1) * 1e3,
            dec_ms=(t3 - t2) * 1e3,
            decoder_steps=hyp.steps,
        )


class _Detokenizer:
    """SentencePiece decode when available (what NeMo does), otherwise join pieces on '▁'."""

    def __init__(self, assets_dir: Path):
        self._sp = None
        spm_path = assets_dir / "tokenizer.model"
        if spm_path.exists():
            try:
                import sentencepiece as spm

                self._sp = spm.SentencePieceProcessor(model_file=str(spm_path))
            except ImportError:
                self._sp = None
        self._pieces = [
            line.rsplit(" ", 1)[0] for line in (assets_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        ]

    def __call__(self, ids: list[int]) -> str:
        if self._sp is not None:
            return self._sp.decode(ids)
        return "".join(self._pieces[i] for i in ids).replace("▁", " ").strip()
