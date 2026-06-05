"""Text encoder for proposal-time retrieval scoring.

Note: in the unified-omni design the captions/ASR text are consumed *natively*
by the omni thinker, so this encoder is optional (`enabled: false` in
configs/model.yaml). When enabled, it embeds caption/ASR windows with BGE-m3 so
`segment_proposal` can rank windows by semantic salience.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextEncoderConfig:
    backbone: str = "BAAI/bge-m3"
    max_tokens: int = 256
    embed_dim: int = 1024
    enabled: bool = True
    device: str = "cuda"


class TextEncoder:
    """Embeds caption/ASR text spans into a shared retrieval space."""

    def __init__(self, config: TextEncoderConfig | None = None) -> None:
        self.config = config or TextEncoderConfig()
        self._model = None

    def load(self) -> None:
        if not self.config.enabled:
            return
        # TODO: load BGE-m3.
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts -> ``[len(texts), embed_dim]`` (L2-normalized)."""
        # TODO: tokenize + embed + normalize.
        raise NotImplementedError
