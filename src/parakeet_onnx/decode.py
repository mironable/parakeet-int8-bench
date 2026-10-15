"""Greedy TDT (token-and-duration transducer) decoding.

Mirrors NeMo's GreedyTDTInfer._greedy_decode for batch size 1, including its two guards:
a blank predicted with duration 0 is forced to advance one frame, and hitting max_symbols on a
frame forces one extra frame of progress.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# step(frame (D,), last_token, state) -> (logits (V + 1 + n_durations,), new_state)
StepFn = Callable[[np.ndarray, int, object], tuple[np.ndarray, object]]


@dataclass
class Hypothesis:
    tokens: list[int] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)  # encoder frame index where each token was emitted
    steps: int = 0  # decoder+joint calls, a proxy for decode cost


def greedy_tdt(
    encoder_frames: np.ndarray,
    step: StepFn,
    init_state: object,
    blank_id: int,
    durations: list[int],
    max_symbols: int = 10,
) -> Hypothesis:
    """encoder_frames: (T, D). Token logits are logits[: blank_id + 1], duration logits the rest."""
    n_frames = encoder_frames.shape[0]
    hyp = Hypothesis()
    state = init_state
    last_token = blank_id  # blank doubles as start-of-sequence; NeMo embeds it as zeros
    t = 0
    while t < n_frames:
        frame = encoder_frames[t]
        symbols = 0
        need_loop = True
        while need_loop and symbols < max_symbols:
            logits, new_state = step(frame, last_token, state)
            hyp.steps += 1
            token = int(np.argmax(logits[: blank_id + 1]))
            skip = durations[int(np.argmax(logits[blank_id + 1 :]))]
            if token == blank_id:
                if skip == 0:
                    skip = 1
            else:
                hyp.tokens.append(token)
                hyp.frames.append(t)
                state = new_state
                last_token = token
            symbols += 1
            t += skip
            need_loop = skip == 0
        if symbols == max_symbols:
            t += 1
    return hyp
