"""Dynamic int8 quantization of the exported encoder and decoder+joint.

By default only MatMul/Gemm nodes with constant weights are quantized (weights to int8 per
channel, activations quantized on the fly at runtime). Attention score MatMuls between two
activations, the convolution modules and the audio front end stay in fp32.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(src: Path, dst: Path, per_channel: bool, ops: list[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        op_types_to_quantize=ops,
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
        use_external_data_format=False,  # int8 weights fit under 2 GB
    )
    print(f"{src} ({_mb(src)}) -> {dst} ({_mb(dst)})")


def _mb(path: Path) -> str:
    total = path.stat().st_size
    data = path.with_name(path.name + ".data")
    if data.exists():
        total += data.stat().st_size
    return f"{total / 2**20:.0f} MiB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="models/fp32")
    ap.add_argument("--dst", default="models/int8")
    ap.add_argument("--per-tensor", action="store_true", help="one scale per weight tensor instead of per channel")
    ap.add_argument("--include-conv", action="store_true", help="also quantize Conv (ConvInteger is slow on many CPUs)")
    args = ap.parse_args()
    ops = ["MatMul", "Gemm", *(["Conv"] if args.include_conv else [])]
    for name in ("encoder.onnx", "decoder_joint.onnx"):
        quantize(Path(args.src) / name, Path(args.dst) / name, per_channel=not args.per_tensor, ops=ops)


if __name__ == "__main__":
    main()
