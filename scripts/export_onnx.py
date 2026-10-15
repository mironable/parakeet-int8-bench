"""Export a NeMo Parakeet TDT checkpoint to ONNX plus the assets the NumPy/ONNX runtime needs.

Writes:
  models/fp32/encoder.onnx (+ encoder.onnx.data; the weights are over protobuf's 2 GB limit)
  models/fp32/decoder_joint.onnx
  models/assets/{preprocessor.npz, model_config.json, vocab.txt, tokenizer.model}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from parakeet_onnx.export import export_pretrained


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    export_pretrained(args.model, Path(args.out))
    print(f"exported {args.model} to {args.out}/fp32 and {args.out}/assets")


if __name__ == "__main__":
    main()
