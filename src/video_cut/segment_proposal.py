"""Coarse candidate-window proposal.

Cheap first pass that narrows a long video to a handful of promising windows so
the omni thinker only has to reason over good candidates. Combines motion
(visual encoder), acoustic-event peaks (CLAP) and semantic salience (text) into
a per-window saliency score.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProposalWindow:
    start_s: float
    end_s: float
    score: float


@dataclass
class ProposalConfig:
    window_seconds: float = 8.0
    stride_seconds: float = 4.0
    top_k: int = 8
    # blend weights for the saliency score
    w_motion: float = 0.4
    w_audio: float = 0.4
    w_text: float = 0.2


def propose_windows(
    features: dict,
    config: ProposalConfig | None = None,
) -> list[ProposalWindow]:
    """Slide over the video and return the top-k highest-saliency windows.

    `features` holds per-window motion / audio-event / text scores produced at
    preprocessing time. Returns windows sorted by descending score.
    """
    # TODO: slide windows, blend the three signals, NMS-overlap, take top_k.
    raise NotImplementedError
