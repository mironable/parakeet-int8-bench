"""Summarise results/<variant>/*.jsonl into results/summary.json and results/results.md."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from parakeet_onnx.evaluation import DATASET_LANG, iter_jsonl, normalize, wer

VARIANTS = ("torch", "onnx-fp32", "onnx-int8")
LABEL = {"torch": "PyTorch fp32", "onnx-fp32": "ONNX fp32", "onnx-int8": "ONNX int8"}
MIN_GROUP = 20


def _load(results: Path) -> tuple[dict, dict]:
    rows: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    runs = {}
    for variant in VARIANTS:
        vdir = results / variant
        if not (vdir / "run.json").exists():
            continue
        runs[variant] = json.loads((vdir / "run.json").read_text())
        for path in sorted(vdir.glob("*.jsonl")):
            rows[path.stem][variant] = list(iter_jsonl(path))
    return rows, runs


def _clip_wer(ref: str, hyp: str) -> float:
    import jiwer

    return float(jiwer.wer(ref, hyp)) if ref else 0.0


def summarise(rows: dict, runs: dict) -> dict:
    summary: dict = {"runs": runs, "datasets": {}}
    for dataset, by_variant in rows.items():
        lang = DATASET_LANG.get(dataset, "en")
        ds: dict = {"clips": None, "audio_h": None, "variants": {}}
        base_ids = None
        for variant, rs in by_variant.items():
            rs = sorted(rs, key=lambda r: r["id"])
            ids = [r["id"] for r in rs]
            base_ids = base_ids or ids
            if ids != base_ids:
                raise SystemExit(f"{dataset}: {variant} covers different clips; rerun bench for all variants")
            refs = [normalize(r["ref"], lang) for r in rs]
            hyps = [normalize(r["hyp"], lang) for r in rs]
            total = np.array([r["total_ms"] for r in rs])
            enc = np.array([r["enc_ms"] for r in rs])
            audio_s = sum(r["duration"] for r in rs)
            ds["clips"], ds["audio_h"] = len(rs), round(audio_s / 3600, 3)
            ds["variants"][variant] = {
                "wer": round(100 * wer(refs, hyps), 2),
                "p50_ms": round(float(np.percentile(total, 50)), 1),
                "p95_ms": round(float(np.percentile(total, 95)), 1),
                "rtf": round(float(total.sum()) / 1e3 / audio_s, 4),
                "encoder_share": round(float(enc.sum() / total.sum()), 3),
                "_hyps": hyps,
                "_refs": refs,
                "_tokens": [r["tokens"] for r in rs],
                "_rows": rs,
            }
        _compare(ds, "onnx-fp32", "torch")
        _compare(ds, "onnx-int8", "onnx-fp32")
        summary["datasets"][dataset] = ds
    return summary


def _compare(ds: dict, variant: str, against: str) -> None:
    v, a = ds["variants"].get(variant), ds["variants"].get(against)
    if not v or not a:
        return
    same = sum(x == y for x, y in zip(v["_tokens"], a["_tokens"], strict=True))
    v[f"identical_to_{against}"] = round(same / len(v["_tokens"]), 3)
    v[f"wer_vs_{against}"] = round(100 * wer(a["_hyps"], v["_hyps"]), 2)


def _by_group(ds: dict, key: str) -> list[tuple[str, int, float, float]]:
    fp32, int8 = ds["variants"].get("onnx-fp32"), ds["variants"].get("onnx-int8")
    if not fp32 or not int8:
        return []
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(fp32["_rows"]):
        groups[str(row.get("meta", {}).get(key) or "unknown")].append(i)
    out = []
    for name, idx in groups.items():
        if len(idx) < MIN_GROUP:
            continue
        refs = [fp32["_refs"][i] for i in idx]
        out.append(
            (
                name,
                len(idx),
                100 * wer(refs, [fp32["_hyps"][i] for i in idx]),
                100 * wer(refs, [int8["_hyps"][i] for i in idx]),
            )
        )
    return sorted(out, key=lambda g: g[3] - g[2], reverse=True)


def _worst_int8(ds: dict, n: int) -> list[dict]:
    fp32, int8 = ds["variants"].get("onnx-fp32"), ds["variants"].get("onnx-int8")
    if not fp32 or not int8:
        return []
    items = []
    for i, ref in enumerate(fp32["_refs"]):
        delta = _clip_wer(ref, int8["_hyps"][i]) - _clip_wer(ref, fp32["_hyps"][i])
        if delta > 0:
            items.append((delta, i))
    items.sort(reverse=True)
    return [
        {"id": fp32["_rows"][i]["id"], "ref": fp32["_refs"][i], "fp32": fp32["_hyps"][i], "int8": int8["_hyps"][i]}
        for _, i in items[:n]
    ]


def to_markdown(summary: dict) -> str:
    runs = summary["runs"]
    lines = ["# Results", ""]
    machine = next(iter(runs.values()))["machine"] if runs else {}
    threads = next(iter(runs.values()))["threads"] if runs else "?"
    lines += [
        f"Machine: {machine.get('cpu', '?')}, {machine.get('memory_gb', '?')} GB, {machine.get('platform', '?')}. "
        f"Batch size 1, {threads} CPU threads, onnxruntime {machine.get('onnxruntime', '?')}.",
        "",
        "## Accuracy and latency",
        "",
        "| Dataset | Clips | Variant | WER % | p50 ms | p95 ms | RTF | Identical tokens | WER vs reference % |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, ds in summary["datasets"].items():
        for variant in VARIANTS:
            v = ds["variants"].get(variant)
            if not v:
                continue
            against = {"onnx-fp32": "torch", "onnx-int8": "onnx-fp32"}.get(variant)
            same = (
                f"{100 * v[f'identical_to_{against}']:.1f}% vs {LABEL[against]}"
                if against and f"identical_to_{against}" in v
                else "-"
            )
            delta = f"{v[f'wer_vs_{against}']:.2f}" if against and f"wer_vs_{against}" in v else "-"
            lines.append(
                f"| {dataset} | {ds['clips']} | {LABEL[variant]} | {v['wer']:.2f} | {v['p50_ms']:.0f} | "
                f"{v['p95_ms']:.0f} | {v['rtf']:.4f} | {same} | {delta} |"
            )
    lines += [
        "",
        "## Size and memory",
        "",
        "| Variant | Weights on disk MB | Peak RSS MB | Load s |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        r = runs.get(variant)
        if r:
            size = (
                r["model_size_mb"]
                if r["model_size_mb"] is not None
                else runs.get("onnx-fp32", {}).get("model_size_mb", "-")
            )
            lines.append(f"| {LABEL[variant]} | {size} | {r['peak_rss_mb']:.0f} | {r['load_s']:.1f} |")

    edacc = summary["datasets"].get("edacc")
    if edacc:
        groups = _by_group(edacc, "l1")
        if groups:
            lines += [
                "",
                f"## EdAcc by first language (groups with at least {MIN_GROUP} clips)",
                "",
                "| L1 | Clips | WER fp32 % | WER int8 % | Change |",
                "|---|---:|---:|---:|---:|",
            ]
            lines += [f"| {g} | {n} | {a:.2f} | {b:.2f} | {b - a:+.2f} |" for g, n, a, b in groups]

    for dataset, ds in summary["datasets"].items():
        worst = _worst_int8(ds, 5)
        if worst:
            lines += ["", f"## Where int8 changed the transcript most: {dataset}", ""]
            for w in worst:
                lines += [f"- {w['id']}", f"  - ref:  {w['ref']}", f"  - fp32: {w['fp32']}", f"  - int8: {w['int8']}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = Path(args.results)
    rows, runs = _load(results)
    if not runs:
        raise SystemExit("no results; run `make bench` first")
    summary = summarise(rows, runs)
    (results / "results.md").write_text(to_markdown(summary), encoding="utf-8")
    public = {
        "runs": summary["runs"],
        "datasets": {
            d: {
                **{k: v for k, v in ds.items() if k != "variants"},
                "variants": {
                    n: {k: x for k, x in v.items() if not k.startswith("_")} for n, v in ds["variants"].items()
                },
            }
            for d, ds in summary["datasets"].items()
        },
    }
    (results / "summary.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
    print((results / "results.md").read_text())


if __name__ == "__main__":
    main()
