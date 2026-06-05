"""Multimodal fusion.

In the chosen UNIFIED-OMNI architecture there is no separately trained fusion
transformer: Qwen2.5-Omni already aligns video + audio + text in pretraining.
This module is therefore a thin adapter that packs raw modalities into the
processor inputs the omni backbone expects. (If you switch to the modular
architecture, replace `OmniFusion` with a real Q-Former / Perceiver here.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FusionInputs:
    """A time-aligned bundle of modalities for one video."""
    video_path: str
    frame_rate: float
    transcript: str | None = None
    captions: list[dict] | None = None   # [{start, end, text}, ...]


class OmniFusion:
    """Packs modalities into omni-processor inputs (no learned fusion params)."""

    def __init__(self, processor: Any | None = None, frame_rate: float = 2.0) -> None:
        self.processor = processor
        self.frame_rate = frame_rate

    def build_inputs(self, sample: FusionInputs, prompt: str) -> dict:
        """Return processor kwargs (pixel/audio/text tensors) for the thinker.

        The omni processor handles video frame sampling and audio resampling; we
        just assemble the chat turn and attach the media references.
        """
        # TODO: call self.processor(...) with video/audio + the text prompt.
        raise NotImplementedError(
            "Pack video/audio/text into Qwen2.5-Omni processor inputs here."
        )
