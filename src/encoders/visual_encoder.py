"""Motion-aware visual encoder used for fast candidate-segment proposal.

Backbone: InternVideo2 (configurable). Operates on short temporal windows so it
captures *motion* (a ball hitting the net, a tackle, a celebration) rather than
isolated frames. These embeddings feed `video_cut.segment_proposal`; the heavy
cross-modal reasoning is done later by the unified omni thinker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VisualEncoderConfig:
    backbone: str = "OpenGVLab/InternVideo2-Stage2_1B"
    clip_seconds: float = 4.0
    embed_dim: int = 768
    device: str = "cuda"


class VisualEncoder:
    """Encodes short video clips into motion-aware embeddings."""

    def __init__(self, config: VisualEncoderConfig | None = None) -> None:
        self.config = config or VisualEncoderConfig()
        self._model = None  # lazy-loaded backbone

    def load(self) -> None:
        """Load the InternVideo2 backbone onto the target device."""
        # TODO: instantiate InternVideo2 and move to self.config.device.
        raise NotImplementedError("Load the InternVideo2 backbone here.")

    def encode_clips(self, clips: list[np.ndarray]) -> np.ndarray:
        """Encode a list of clips (each [T, H, W, 3] uint8) into embeddings.

        Returns an array of shape ``[len(clips), embed_dim]``.
        """
        # TODO: preprocess frames, run the backbone, return pooled embeddings.
        raise NotImplementedError

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim
