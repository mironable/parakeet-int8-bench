"""Transcribe audio files with the exported ONNX model; no PyTorch or NeMo needed.

python scripts/transcribe.py models/int8 clip1.wav clip2.flac
"""

from __future__ import annotations

import argparse
from pathlib import Path

from parakeet_onnx import OnnxParakeet
from parakeet_onnx.evaluation import load_audio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="models/fp32 or models/int8")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--assets", default="models/assets")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    model = OnnxParakeet(Path(args.weights), Path(args.assets), threads=args.threads)
    for path in args.audio:
        r = model.transcribe(load_audio(path))
        print(f"{path}\t{r.total_ms:.0f} ms\t{r.text}")


if __name__ == "__main__":
    main()
