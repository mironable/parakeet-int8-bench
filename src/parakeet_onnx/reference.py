"""The PyTorch/NeMo baseline: same checkpoint, same greedy TDT decoding, batch size 1, CPU."""

from __future__ import annotations

import time

import numpy as np

from .model import Result


class TorchParakeet:
    def __init__(self, name: str = "nvidia/parakeet-tdt-0.6b-v3", threads: int = 4):
        import nemo.collections.asr as nemo_asr
        import torch
        from omegaconf import open_dict

        torch.set_num_threads(threads)
        self._torch = torch
        self.model = nemo_asr.models.ASRModel.from_pretrained(name, map_location="cpu")
        self.model.eval()
        # Plain per-utterance greedy, the algorithm the ONNX path reimplements.
        cfg = self.model.cfg.decoding
        with open_dict(cfg):
            cfg.strategy = "greedy"
            cfg.greedy.max_symbols = cfg.greedy.get("max_symbols", 10) or 10
        self.model.change_decoding_strategy(cfg, verbose=False)

    def features(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        torch = self._torch
        with torch.inference_mode():
            feats, flen = self.model.preprocessor(
                input_signal=torch.from_numpy(np.asarray(audio, dtype=np.float32))[None],
                length=torch.tensor([audio.shape[0]]),
            )
        return feats[0].numpy(), int(flen[0])

    def transcribe(self, audio: np.ndarray) -> Result:
        torch = self._torch
        with torch.inference_mode():
            t0 = time.perf_counter()
            feats, flen = self.model.preprocessor(
                input_signal=torch.from_numpy(np.asarray(audio, dtype=np.float32))[None],
                length=torch.tensor([audio.shape[0]]),
            )
            t1 = time.perf_counter()
            enc, elen = self.model.encoder(audio_signal=feats, length=flen)
            t2 = time.perf_counter()
            hyps = self.model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=elen)
            t3 = time.perf_counter()
        hyp = hyps[0]
        seq = hyp.y_sequence
        tokens = [int(x) for x in (seq.tolist() if hasattr(seq, "tolist") else seq)]
        return Result(
            text=hyp.text if isinstance(hyp.text, str) else self.model.tokenizer.ids_to_text(tokens),
            tokens=tokens,
            feat_ms=(t1 - t0) * 1e3,
            enc_ms=(t2 - t1) * 1e3,
            dec_ms=(t3 - t2) * 1e3,
            decoder_steps=-1,
        )
