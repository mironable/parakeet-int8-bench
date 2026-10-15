"""Parity tests against NeMo, using small randomly initialised modules so no checkpoint download is needed.

They pin the parts that are easy to get subtly wrong: the log-mel front end, the I/O contract of
NeMo's ONNX export, and the TDT greedy loop. The real-checkpoint check is scripts/check.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nemo")

from nemo.collections.asr.modules import (  # noqa: E402
    AudioToMelSpectrogramPreprocessor,
    ConformerEncoder,
    RNNTDecoder,
    RNNTJoint,
)
from nemo.collections.asr.modules.rnnt import RNNTDecoderJoint  # noqa: E402
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyTDTInfer  # noqa: E402

from parakeet_onnx import FeatureConfig, LogMel, OnnxParakeet  # noqa: E402
from parakeet_onnx.export import consolidate, save_model_config, save_preprocessor, save_vocab  # noqa: E402

# Front-end settings of nvidia/parakeet-tdt-0.6b-v3
PREPROCESSOR_CFG = dict(
    sample_rate=16000,
    normalize="per_feature",
    window_size=0.025,
    window_stride=0.01,
    window="hann",
    features=128,
    n_fft=512,
    log=True,
    frame_splicing=1,
    dither=1e-5,
    pad_to=0,
    pad_value=0.0,
)
DURATIONS = [0, 1, 2, 3, 4]
VOCAB = 40
ENC_DIM, PRED_DIM = 64, 48


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t)
    tone = sum(np.sin(2 * np.pi * k * np.cumsum(f0) / 16000) / k for k in range(1, 6))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t) ** 2
    audio = 0.1 * envelope * tone + 0.01 * rng.standard_normal(t.size)
    return audio.astype(np.float32)


@pytest.fixture(scope="module")
def preprocessor():
    pre = AudioToMelSpectrogramPreprocessor(**PREPROCESSOR_CFG)
    pre.eval()
    return pre


def _nemo_features(pre, audio: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        feats, _ = pre(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([audio.size]))
    return feats[0].numpy()


@pytest.mark.parametrize("seconds,seed", [(0.35, 0), (1.0, 1), (4.37, 2), (11.2, 3)])
def test_features_match_nemo(preprocessor, tmp_path: Path, seconds: float, seed: int):
    save_preprocessor(preprocessor, tmp_path / "preprocessor.npz")
    ours = LogMel(FeatureConfig.load(tmp_path / "preprocessor.npz"))
    audio = _speechlike(seconds, seed)
    ref = _nemo_features(preprocessor, audio)
    got, n_valid = ours(audio)
    assert got.shape == ref.shape
    assert n_valid == ours.num_frames(audio.size)
    np.testing.assert_allclose(got, ref, atol=2e-3, rtol=0)


def _tiny_model(seed: int):
    torch.manual_seed(seed)
    encoder = ConformerEncoder(
        feat_in=128,
        n_layers=2,
        d_model=ENC_DIM,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_channels=32,
        n_heads=4,
        conv_kernel_size=9,
        xscaling=True,
        dropout=0.0,
        dropout_pre_encoder=0.0,
        dropout_emb=0.0,
    )
    decoder = RNNTDecoder(prednet={"pred_hidden": PRED_DIM, "pred_rnn_layers": 2}, vocab_size=VOCAB)
    joint = RNNTJoint(
        jointnet={"encoder_hidden": ENC_DIM, "pred_hidden": PRED_DIM, "joint_hidden": 64, "activation": "relu"},
        num_classes=VOCAB,
        num_extra_outputs=len(DURATIONS),
    )
    # Random init leaves the duration head near uniform; sharpen it so all five durations occur.
    with torch.no_grad():
        for p in joint.parameters():
            p.mul_(3.0)
    for m in (encoder, decoder, joint):
        m.eval()
    return encoder, decoder, joint


def _reference_tokens(pre, encoder, decoder, joint, audio: np.ndarray) -> list[int]:
    greedy = GreedyTDTInfer(decoder, joint, blank_index=VOCAB, durations=DURATIONS, max_symbols_per_step=10)
    with torch.inference_mode():
        feats, flen = pre(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([audio.size]))
        enc, elen = encoder(audio_signal=feats, length=flen)
        (hyps,) = greedy(encoder_output=enc, encoded_lengths=elen)
    seq = hyps[0].y_sequence
    return [int(x) for x in (seq.tolist() if hasattr(seq, "tolist") else seq)]


def _export_tiny(pre, encoder, decoder, joint, root: Path) -> tuple[Path, Path]:
    tmp, model_dir, assets = root / "raw", root / "fp32", root / "assets"
    tmp.mkdir(parents=True)
    encoder.export(str(tmp / "encoder.onnx"))
    RNNTDecoderJoint(decoder, joint).export(str(tmp / "decoder_joint.onnx"))
    consolidate(tmp / "encoder.onnx", model_dir / "encoder.onnx")
    consolidate(tmp / "decoder_joint.onnx", model_dir / "decoder_joint.onnx")
    save_preprocessor(pre, assets / "preprocessor.npz")
    save_vocab(assets / "vocab.txt", [f"▁t{i}" for i in range(VOCAB)])
    save_model_config(assets / "model_config.json", VOCAB, DURATIONS, 10)
    return model_dir, assets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_onnx_pipeline_matches_nemo_greedy_tdt(preprocessor, tmp_path: Path, seed: int):
    encoder, decoder, joint = _tiny_model(seed)
    clips = [_speechlike(s, seed * 10 + i) for i, s in enumerate((0.6, 2.3, 6.1))]
    # Reference first: export flips export-only flags on the modules.
    refs = [_reference_tokens(preprocessor, encoder, decoder, joint, a) for a in clips]
    model_dir, assets = _export_tiny(preprocessor, encoder, decoder, joint, tmp_path)

    model = OnnxParakeet(model_dir, assets, threads=1)
    assert json.loads((assets / "model_config.json").read_text())["blank_id"] == VOCAB
    for audio, ref in zip(clips, refs, strict=True):
        assert model.transcribe(audio).tokens == ref
    assert any(refs), "all reference hypotheses empty; the test would not exercise decoding"


def test_quantized_model_runs(preprocessor, tmp_path: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    encoder, decoder, joint = _tiny_model(0)
    model_dir, assets = _export_tiny(preprocessor, encoder, decoder, joint, tmp_path)
    int8_dir = tmp_path / "int8"
    int8_dir.mkdir()
    for name in ("encoder.onnx", "decoder_joint.onnx"):
        quantize_dynamic(
            str(model_dir / name),
            str(int8_dir / name),
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=True,
            weight_type=QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},
        )
    result = OnnxParakeet(int8_dir, assets, threads=1).transcribe(_speechlike(2.0, 7))
    assert result.enc_ms > 0
