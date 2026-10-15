"""Export helpers. Needs NeMo and PyTorch; nothing in the runtime path imports this module."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

_CHUNK = 64 * 2**20


def _all_tensors(model: onnx.ModelProto):
    try:
        from onnx.external_data_helper import _get_all_tensors

        return list(_get_all_tensors(model))
    except ImportError:
        return list(model.graph.initializer)


def consolidate(src: Path, dst: Path) -> None:
    """Write an exported graph to dst, gathering any external weights into one dst.data file.

    Over 2 GB the TorchScript exporter writes one file per tensor. Each source file is copied in
    chunks and deleted once its tensors are moved, so peak disk use stays near one copy of the
    weights. Graphs with inline weights are saved unchanged.
    """
    model = onnx.load(str(src), load_external_data=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[onnx.TensorProto]] = defaultdict(list)
    for tensor in _all_tensors(model):
        if uses_external_data(tensor):
            by_file[ExternalDataInfo(tensor).location].append(tensor)
    if not by_file:
        onnx.save_model(model, str(dst))
        return

    data_name = dst.name + ".data"
    with (dst.parent / data_name).open("wb") as out:
        for location, tensors in by_file.items():
            src_file = src.parent / location
            with src_file.open("rb") as f:
                for tensor in tensors:
                    info = ExternalDataInfo(tensor)
                    f.seek(info.offset or 0)
                    remaining = info.length if info.length else src_file.stat().st_size - (info.offset or 0)
                    offset = out.tell()
                    length = remaining
                    while remaining:
                        chunk = f.read(min(_CHUNK, remaining))
                        if not chunk:
                            raise OSError(f"{src_file} ended early while copying {tensor.name}")
                        out.write(chunk)
                        remaining -= len(chunk)
                    del tensor.external_data[:]
                    for key, value in (("location", data_name), ("offset", offset), ("length", length)):
                        entry = tensor.external_data.add()
                        entry.key, entry.value = key, str(value)
            src_file.unlink()
    onnx.save_model(model, str(dst))


def _log_zero_guard(featurizer) -> float:
    value = getattr(featurizer, "log_zero_guard_value", 2**-24)
    if isinstance(value, str):
        if value == "tiny":
            return float(np.finfo(np.float32).tiny)
        if value == "eps":
            return float(np.finfo(np.float32).eps)
        raise ValueError(f"unknown log_zero_guard_value {value!r}")
    return float(value)


def save_preprocessor(preprocessor, path: Path) -> None:
    """Copy NeMo's mel filterbank, window and STFT settings so NumPy can reproduce the features."""
    fz = preprocessor.featurizer
    if getattr(fz, "log_zero_guard_type", "add") != "add":
        raise NotImplementedError(f"log_zero_guard_type={fz.log_zero_guard_type}")
    if getattr(fz, "frame_splicing", 1) != 1:
        raise NotImplementedError("frame_splicing > 1")
    if getattr(fz, "exact_pad", False):
        raise NotImplementedError("exact_pad")
    if not getattr(fz, "log", True):
        raise NotImplementedError("log=False")
    fb = fz.fb.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sample_rate=np.int64(preprocessor._sample_rate),
        n_fft=np.int64(fz.n_fft),
        hop_length=np.int64(fz.hop_length),
        win_length=np.int64(fz.win_length),
        preemph=np.float64(fz.preemph or 0.0),
        mag_power=np.float64(fz.mag_power),
        log_zero_guard=np.float64(_log_zero_guard(fz)),
        normalize=np.str_(fz.normalize or "none"),
        pad_value=np.float64(fz.pad_value),
        fb=fb.reshape(fb.shape[-2], fb.shape[-1]).astype(np.float32),
        window=fz.window.detach().cpu().numpy().astype(np.float32),
    )


def save_model_config(path: Path, vocab_size: int, durations: list[int], max_symbols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "blank_id": vocab_size,
        "vocab_size": vocab_size,
        "durations": [int(d) for d in durations],
        "max_symbols": int(max_symbols),
    }
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def save_vocab(path: Path, pieces: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, piece in enumerate([*pieces, "<blk>"]):
            f.write(f"{piece} {i}\n")


def _tdt_durations(cfg) -> list[int]:
    for getter in (
        lambda: cfg.decoding.durations,
        lambda: cfg.model_defaults.tdt_durations,
        lambda: cfg.loss.tdt_kwargs.durations,
    ):
        try:
            value = getter()
        except (AttributeError, KeyError):
            continue
        if value:
            return [int(d) for d in value]
    raise RuntimeError("not a TDT model: no durations in the config")


def export_subnets(exportable, out_dir: Path) -> None:
    """Run NeMo's export (encoder + fused decoder/joint) and write encoder.onnx, decoder_joint.onnx."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        exportable.export(str(Path(tmp) / "model.onnx"))
        consolidate(Path(tmp) / "encoder-model.onnx", out_dir / "encoder.onnx")
        consolidate(Path(tmp) / "decoder_joint-model.onnx", out_dir / "decoder_joint.onnx")


def export_pretrained(name: str, out: Path) -> None:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(name, map_location="cpu")
    model.eval()

    assets = out / "assets"
    save_preprocessor(model.preprocessor, assets / "preprocessor.npz")
    pieces = list(model.tokenizer.vocab)
    save_vocab(assets / "vocab.txt", pieces)
    sp = getattr(model.tokenizer, "tokenizer", None)
    if sp is not None and hasattr(sp, "serialized_model_proto"):
        (assets / "tokenizer.model").write_bytes(sp.serialized_model_proto())
    durations = _tdt_durations(model.cfg)
    max_symbols = model.cfg.decoding.get("greedy", {}).get("max_symbols", 10) or 10
    save_model_config(assets / "model_config.json", len(pieces), durations, max_symbols)

    export_subnets(model, out / "fp32")

    expected = len(pieces) + 1 + len(durations)
    dec = onnx.load(str(out / "fp32" / "decoder_joint.onnx"), load_external_data=False)
    last = dec.graph.output[0].type.tensor_type.shape.dim[-1]
    if last.HasField("dim_value") and last.dim_value != expected:
        raise RuntimeError(f"decoder_joint emits {last.dim_value} logits, expected {expected}")
