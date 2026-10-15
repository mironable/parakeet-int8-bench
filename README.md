# parakeet-int8-bench

How much accuracy does 8-bit quantization cost [NVIDIA Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), and what does it buy on a laptop CPU?

This repo exports the model to ONNX and runs it without PyTorch: NumPy features, ONNX Runtime, and a greedy TDT decoder written from scratch. It then quantizes the weights to int8 and measures WER, latency and memory on read English, accented English and German.

## Results

Apple M1 Pro with 16 GB, CPU only, ONNX Runtime 1.31.0, batch size 1, 4 threads, 300 clips per test set. The full report, with more changed transcripts, is in [results/results.md](results/results.md), and the raw numbers are in [results/summary.json](results/summary.json).

| Dataset | Variant | WER % | p50 ms | p95 ms | RTF | Identical tokens | WER vs reference % |
|---|---|---:|---:|---:|---:|---:|---:|
| LibriSpeech | PyTorch fp32 | 1.93 | 788 | 1070 | 0.1119 | - | - |
| LibriSpeech | ONNX fp32 | 1.93 | 345 | 988 | 0.0586 | 100.0% vs PyTorch | 0.00 |
| LibriSpeech | ONNX int8 | 1.93 | 163 | 469 | 0.0279 | 88.0% vs ONNX fp32 | 0.27 |
| EdAcc | PyTorch fp32 | 14.82 | 765 | 1197 | 0.1218 | - | - |
| EdAcc | ONNX fp32 | 14.82 | 269 | 1111 | 0.0588 | 100.0% vs PyTorch | 0.00 |
| EdAcc | ONNX int8 | 14.73 | 133 | 542 | 0.0286 | 74.7% vs ONNX fp32 | 1.64 |
| FLEURS de | PyTorch fp32 | 6.31 | 1251 | 1525 | 0.0976 | - | - |
| FLEURS de | ONNX fp32 | 6.31 | 649 | 1276 | 0.0553 | 100.0% vs PyTorch | 0.00 |
| FLEURS de | ONNX int8 | 6.42 | 312 | 614 | 0.0267 | 81.7% vs ONNX fp32 | 0.94 |

| Variant | Weights on disk MB | Peak RSS MB | Load s |
|---|---:|---:|---:|
| PyTorch fp32 | 2432 | 5413 | 19.9 |
| ONNX fp32 | 2432 | 3070 | 5.1 |
| ONNX int8 | 892 | 1886 | 1.1 |

| EdAcc first language | Clips | WER fp32 % | WER int8 % | Change |
|---|---:|---:|---:|---:|
| Catalan | 85 | 19.70 | 19.88 | +0.18 |
| Scottish English | 70 | 15.77 | 15.51 | -0.26 |
| Spanish | 38 | 15.97 | 15.56 | -0.42 |
| Lithuanian | 28 | 9.05 | 8.62 | -0.43 |
| Sinhalese | 30 | 13.29 | 12.82 | -0.47 |

What the numbers show:

- The ONNX fp32 path emits the same tokens as NeMo on all 900 clips, so its WER is identical to PyTorch.
- Int8 barely moves WER: LibriSpeech is unchanged at 1.93%, EdAcc goes from 14.82% to 14.73%, and FLEURS German goes from 6.31% to 6.42%. It still changes the output on 12% to 25% of clips. The errors it adds and the errors it removes roughly cancel.
- Int8 runs at about half the real-time factor of ONNX fp32 (2.06x to 2.10x faster). Median latency on LibriSpeech drops from 788 ms (PyTorch) to 345 ms (ONNX fp32) to 163 ms (int8).
- ONNX latency grows linearly with clip length, at about 55 to 60 ms per second of audio for fp32 and 27 to 30 ms for int8, so p95 reflects the longest clips. PyTorch takes about 700 ms per call even on clips under 3 seconds, then only about 25 ms more per second of audio. On LibriSpeech and EdAcc that makes it 4.0x to 5.3x slower than ONNX fp32 on clips under 3 seconds and about as fast on clips over 20 seconds.
- PyTorch latency was not stable within a run. Partway through, the PyTorch process got about 300 ms slower per call and stayed that way. In the benchmark run this had happened by the time FLEURS started, so the FLEURS PyTorch row was measured entirely in the slower state; in a FLEURS-only rerun it happened after about 200 clips. Transcripts were identical in both runs, and the ONNX runs show no such step.
- Int8 weights are 2.7x smaller (892 MB vs 2432 MB). The ratio stays below 4x because 292 MB of encoder weights stay fp32, including the convolution modules. Peak memory falls from 3070 MB to 1886 MB, and load time from 5.1 s to 1.1 s.
- On EdAcc, no first-language group with at least 20 clips gets more than 0.2 points worse. Catalan is the only group whose WER rises (+0.18). The groups have only 28 to 85 clips each, so these differences are too small to rank accents by quantization sensitivity.

Typical int8 changes are word boundaries and spellings of rare words:

| Set | Reference | ONNX fp32 | ONNX int8 |
|---|---|---|---|
| LibriSpeech | anyone in the room | anyone in the room | any one in the room |
| LibriSpeech | her bains de mer | her band de mer | her bandamer |
| EdAcc | it was my naughty pleasure as well | it was my naughty | it was my nutty |
| FLEURS de | antike kulturen und volksstämme | antike kulturen und volksstämme | antikulturen und volksstämme |
| FLEURS de | im august 2005 | im august 2005 | im august zwei tausend fünf |

The last row counts as three errors even though the words are correct, because the basic normalizer used for German does not normalize numbers.

## What's in here

| Path | What it does |
|---|---|
| `src/parakeet_onnx/features.py` | Log-mel front end in NumPy. The filterbank and window are copied from the checkpoint, so the largest difference from NeMo's features over 24 real clips is 1.1e-4. |
| `src/parakeet_onnx/decode.py` | Greedy TDT decoding over the token and duration heads, mirroring NeMo's `GreedyTDTInfer`, including its two guards against getting stuck on a frame. |
| `src/parakeet_onnx/model.py` | The runtime: features, ONNX encoder, then the decoder/joint loop. No PyTorch or NeMo import. |
| `src/parakeet_onnx/export.py` | NeMo to ONNX: the encoder plus a fused decoder/joint graph. The fp32 encoder is over protobuf's 2 GB limit, so its weights go in one external `.data` file. |
| `scripts/quantize.py` | Dynamic int8 via ONNX Runtime: MatMul/Gemm weights only, one scale per output channel. Attention score MatMuls (two activations), the convolution modules and the front end stay fp32. |
| `scripts/prepare_data.py` | Downloads the test sets and writes 16 kHz manifests. |
| `scripts/bench.py` | Transcribes every clip with one variant at batch size 1 and logs per-clip timings. |
| `scripts/report.py` | WER, p50/p95 latency, real-time factor, token agreement between variants, EdAcc by first language, and the clips where int8 changed the output most. |
| `scripts/check.py` | Runs the ONNX path against NeMo on the real checkpoint before any benchmarking. |
| `scripts/transcribe.py` | Transcribes files with the ONNX model. |

## Test sets

| Name | Source | Speech | License |
|---|---|---|---|
| `librispeech` | [LibriSpeech test-clean](https://www.openslr.org/12) | Read English audiobooks | CC BY 4.0 |
| `edacc` | [EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc), first test shard | Conversational English from speakers with a wide range of accents | CC BY-SA 4.0 |
| `fleurs_de` | [FLEURS](https://huggingface.co/datasets/google/fleurs) `de_de` test | Read German | CC BY 4.0 |

Each set is filtered to clips of 1 to 30 seconds, then 300 clips are drawn at random with seed 0. WER is computed after Whisper's English text normalizer (English sets) or its basic normalizer (German).

## Reproduce

Needs Python 3.11.4 or newer (pass another interpreter with `PYTHON=...`) and about 12 GB of free disk: the checkpoint cache, the fp32 and int8 models, the virtualenv and the test audio.

```sh
make setup      # virtualenv with NeMo, ONNX Runtime and the eval tools
make export     # downloads the checkpoint (2.4 GB) and writes models/fp32 and models/assets
make quantize   # writes models/int8
make data       # about 1.5 GB of test audio
make check      # NumPy/ONNX path against NeMo on real clips; must end with OK
make bench      # PyTorch fp32, ONNX fp32 and ONNX int8, batch size 1, 4 threads
make report     # results/results.md and results/summary.json
```

`LIMIT=`, `THREADS=` and `DATASETS=` override the defaults. To transcribe your own files:

```sh
.venv/bin/python scripts/transcribe.py models/int8 recording.wav
```

## How correctness is checked

- `make test` builds small, randomly initialised NeMo modules (a FastConformer encoder, an LSTM prediction network and a TDT joint with five durations). It checks that the NumPy features match NeMo's preprocessor and that the ONNX pipeline emits exactly the tokens NeMo's `GreedyTDTInfer` emits, clip by clip. No checkpoint download needed.
- `make check` repeats the comparison on the real checkpoint and real clips.

## Scope

- CPU only, batch size 1, greedy decoding, full utterances. That is closest to running on a phone, but this is not a streaming setup.
- Latency comes from ONNX Runtime's CPU execution provider on the machine named in the results. A phone runtime such as [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) or Core ML will give different numbers.

## License

Code: MIT. The model weights are [CC BY 4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) from NVIDIA. No weights or audio are stored in this repo.
