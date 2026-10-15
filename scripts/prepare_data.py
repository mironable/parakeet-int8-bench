"""Download the public test sets and write 16 kHz manifests to data/manifests/<dataset>.jsonl.

  librispeech  LibriSpeech test-clean (read English; sanity check against published numbers)
  edacc        Edinburgh International Accents of English, first test shard (accented conversational English)
  fleurs_de    FLEURS German test split (read German)

Clips outside 1-30 s, or with an empty reference after normalisation, are dropped; then a
fixed-seed random subset of --limit clips per dataset is kept.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from parakeet_onnx.evaluation import DATASET_LANG, SAMPLE_RATE, Clip, normalize, write_manifest

MIN_S, MAX_S = 1.0, 30.0
LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://us.openslr.org/resources/12/test-clean.tar.gz",
]


def _download(urls: list[str], dst: Path) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"downloading {url}")
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)
            return dst
        except OSError as e:
            last_error = e
            print(f"  failed: {e}")
    raise RuntimeError(f"could not download {dst.name}") from last_error


def _keep(clip: Clip) -> bool:
    return MIN_S <= clip.duration <= MAX_S and bool(normalize(clip.text, DATASET_LANG[clip.dataset]))


def librispeech(root: Path) -> list[Clip]:
    tar_path = _download(LIBRISPEECH_URLS, root / "downloads" / "test-clean.tar.gz")
    base = root / "librispeech"
    if not (base / "LibriSpeech").exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(base, filter="data")
    clips = []
    for trans in sorted((base / "LibriSpeech" / "test-clean").rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            flac = trans.parent / f"{utt}.flac"
            info = sf.info(str(flac))
            clips.append(Clip(utt, "librispeech", str(flac), text, info.frames / info.samplerate, {}))
    return clips


def edacc(root: Path) -> list[Clip]:
    import pyarrow.parquet as pq
    import soxr
    from huggingface_hub import HfApi, hf_hub_download

    repo = "edinburghcstr/edacc"
    shard = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("data/test-"))[0]
    parquet = hf_hub_download(repo, shard, repo_type="dataset", cache_dir=str(root / "downloads" / "hf"))
    out_dir = root / "edacc" / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet, columns=["speaker", "text", "accent", "l1", "audio"])
    clips = []
    for i, row in enumerate(table.to_pylist()):
        text = (row["text"] or "").strip()
        if not text or "IGNORE_TIME_SEGMENT" in text:
            continue
        wav = out_dir / f"edacc_{i:05d}.wav"
        if not wav.exists():
            audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = soxr.resample(audio, sr, SAMPLE_RATE, quality="HQ").astype(np.float32)
            sf.write(str(wav), audio, SAMPLE_RATE, subtype="PCM_16")
        info = sf.info(str(wav))
        meta = {"speaker": row["speaker"], "accent": row["accent"], "l1": row["l1"]}
        clips.append(Clip(f"edacc_{i:05d}", "edacc", str(wav), text, info.frames / info.samplerate, meta))
    return clips


def fleurs_de(root: Path) -> list[Clip]:
    from huggingface_hub import hf_hub_download

    repo, cache = "google/fleurs", str(root / "downloads" / "hf")
    tsv = hf_hub_download(repo, "data/de_de/test.tsv", repo_type="dataset", cache_dir=cache)
    tar_path = hf_hub_download(repo, "data/de_de/audio/test.tar.gz", repo_type="dataset", cache_dir=cache)
    out_dir = root / "fleurs_de" / "wav"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".wav"):
                    member.name = Path(member.name).name
                    tar.extract(member, out_dir, filter="data")
    clips = []
    with open(tsv, encoding="utf-8") as f:
        # id, file_name, raw_transcription, transcription, phonemes, num_samples, gender (no header)
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            _id, file_name, raw, _norm, _ph, _n, gender = row[:7]
            wav = out_dir / file_name
            if not wav.exists():
                continue
            info = sf.info(str(wav))
            clip_id = f"fleurs_de_{Path(file_name).stem}"
            clips.append(Clip(clip_id, "fleurs_de", str(wav), raw, info.frames / info.samplerate, {"gender": gender}))
    return clips


BUILDERS = {"librispeech": librispeech, "edacc": edacc, "fleurs_de": fleurs_de}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--datasets", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    ap.add_argument("--limit", type=int, default=300, help="clips per dataset; 0 keeps all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    for name in args.datasets:
        clips = [c for c in BUILDERS[name](root) if _keep(c)]
        clips.sort(key=lambda c: c.id)
        if args.limit and len(clips) > args.limit:
            clips = sorted(random.Random(args.seed).sample(clips, args.limit), key=lambda c: c.id)
        n = write_manifest(root / "manifests" / f"{name}.jsonl", clips)
        hours = sum(c.duration for c in clips) / 3600
        print(f"{name}: {n} clips, {hours:.2f} h")


if __name__ == "__main__":
    main()
