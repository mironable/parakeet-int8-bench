"""Check the exported ONNX path against NeMo on the real checkpoint before benchmarking.

1. NumPy log-mel features vs NeMo's preprocessor (max abs difference).
2. ONNX fp32 tokens vs NeMo greedy TDT tokens, clip by clip; they should be identical.
3. If models/int8 exists, how often int8 tokens still match fp32 (informational only).

Uses the first --n clips of each prepared manifest, or synthetic audio if none are prepared.
Exits non-zero if 1 or 2 fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from parakeet_onnx import OnnxParakeet
from parakeet_onnx.evaluation import load_audio, read_manifest
from parakeet_onnx.reference import TorchParakeet

FEATURE_ATOL = 2e-3
MIN_TOKEN_MATCH = 0.95  # a near-tie can flip one argmax; more than that means a bug


def _clips(data: Path, n: int) -> list[tuple[str, np.ndarray]]:
    out = []
    for manifest in sorted(data.glob("*.jsonl")):
        for clip in read_manifest(manifest)[:n]:
            out.append((clip.id, load_audio(clip.audio)))
    if not out:
        rng = np.random.default_rng(0)
        out = [(f"noise_{s}s", (0.05 * rng.standard_normal(s * 16000)).astype(np.float32)) for s in (2, 5)]
        print("no manifests found; using synthetic audio (token check is weak without speech)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--data", default="data/manifests")
    ap.add_argument("--n", type=int, default=8, help="clips per dataset")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    models = Path(args.models_dir)
    ref = TorchParakeet(args.model, threads=args.threads)
    fp32 = OnnxParakeet(models / "fp32", models / "assets", threads=args.threads)
    int8 = (
        OnnxParakeet(models / "int8", models / "assets", threads=args.threads) if (models / "int8").exists() else None
    )

    ok = True
    worst_feat = 0.0
    fp32_match = int8_match = 0
    clips = _clips(Path(args.data), args.n)
    for clip_id, audio in clips:
        ref_feats, ref_len = ref.features(audio)
        our_feats, our_len = fp32.features(audio)
        if ref_len != our_len or ref_feats.shape != our_feats.shape:
            print(f"FAIL {clip_id}: feature shape/length {our_feats.shape}/{our_len} vs {ref_feats.shape}/{ref_len}")
            ok = False
            continue
        worst_feat = max(worst_feat, float(np.abs(ref_feats - our_feats).max()))

        r = ref.transcribe(audio)
        f = fp32.transcribe(audio)
        fp32_match += f.tokens == r.tokens
        if f.tokens != r.tokens:
            print(f"  token mismatch {clip_id}\n    nemo: {r.text}\n    onnx: {f.text}")
        if int8 is not None:
            int8_match += int8.transcribe(audio).tokens == f.tokens

    n = len(clips)
    print(f"features: max |numpy - nemo| = {worst_feat:.2e} (limit {FEATURE_ATOL:.0e})")
    print(f"onnx fp32 tokens identical to nemo: {fp32_match}/{n}")
    if int8 is not None:
        print(f"onnx int8 tokens identical to fp32: {int8_match}/{n} (informational)")
    if worst_feat > FEATURE_ATOL:
        ok = False
    if n and fp32_match / n < MIN_TOKEN_MATCH:
        ok = False
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
