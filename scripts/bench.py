"""Transcribe every manifest clip with one model variant, batch size 1, and log per-clip timings.

Run each variant in its own process (the Makefile does) so peak memory is per variant.
Writes results/<variant>/<dataset>.jsonl and results/<variant>/run.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from parakeet_onnx.evaluation import load_audio, read_manifest

VARIANTS = ("torch", "onnx-fp32", "onnx-int8")


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def _dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*.onnx*")) / 2**20


def _machine() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    if sys.platform == "darwin":
        try:
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
            info["memory_gb"] = (
                int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
                / 2**30
            )
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    else:
        info["cpu"] = platform.processor()
    for pkg in ("onnxruntime", "torch", "nemo_toolkit", "numpy"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return info


def load_variant(variant: str, model: str, models_dir: Path, threads: int):
    if variant == "torch":
        from parakeet_onnx.reference import TorchParakeet

        return TorchParakeet(model, threads=threads), None
    from parakeet_onnx import OnnxParakeet

    weights = models_dir / variant.removeprefix("onnx-")
    return OnnxParakeet(weights, models_dir / "assets", threads=threads), _dir_mb(weights)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--data", default="data/manifests")
    ap.add_argument("--datasets", nargs="+", default=["librispeech", "edacc", "fleurs_de"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out) / args.variant
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    model, size_mb = load_variant(args.variant, args.model, Path(args.models_dir), args.threads)
    load_s = time.perf_counter() - t_load

    manifests = {name: read_manifest(Path(args.data) / f"{name}.jsonl") for name in args.datasets}
    first = next(iter(manifests.values()))
    for clip in first[: args.warmup]:
        model.transcribe(load_audio(clip.audio))

    for name, clips in manifests.items():
        audio_s = proc_ms = 0.0
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i, clip in enumerate(clips, 1):
                audio = load_audio(clip.audio)
                r = model.transcribe(audio)
                audio_s += clip.duration
                proc_ms += r.total_ms
                row = {
                    "id": clip.id,
                    "dataset": name,
                    "duration": clip.duration,
                    "ref": clip.text,
                    "hyp": r.text,
                    "tokens": r.tokens,
                    "feat_ms": round(r.feat_ms, 3),
                    "enc_ms": round(r.enc_ms, 3),
                    "dec_ms": round(r.dec_ms, 3),
                    "total_ms": round(r.total_ms, 3),
                    "decoder_steps": r.decoder_steps,
                    "meta": clip.meta,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if i % 50 == 0 or i == len(clips):
                    print(f"[{args.variant}] {name} {i}/{len(clips)}  RTF {proc_ms / 1e3 / audio_s:.4f}", flush=True)

    run = {
        "variant": args.variant,
        "model": args.model,
        "threads": args.threads,
        "load_s": round(load_s, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "model_size_mb": round(size_mb, 1) if size_mb is not None else None,
        "machine": _machine(),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
