"""Audio encoder for proposal: ASR (speech) + acoustic events (non-speech).

Two complementary signals matter for highlights:
  * **Speech / ASR** via Whisper-large-v3 -> transcript + commentary cues.
  * **Acoustic events** via CLAP -> crowd roar, music swell, whistle. CLAP is
    text-queryable, so we can score windows against prompts like
    "crowd cheering loudly" to find excitement that has no speech at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioEncoderConfig:
    asr_backbone: str = "openai/whisper-large-v3"
    event_backbone: str = "laion/clap-htsat-fused"
    sample_rate: int = 16000
    embed_dim: int = 512
    device: str = "cuda"
    event_queries: list[str] = field(
        default_factory=lambda: [
            "crowd cheering loudly",
            "exciting music swell",
            "referee whistle",
            "commentator shouting",
        ]
    )


@dataclass
class AudioFeatures:
    transcript: str
    asr_segments: list[dict]          # [{start, end, text}, ...]
    event_embedding: np.ndarray       # [embed_dim]
    event_scores: dict[str, float]    # query -> similarity in [0, 1]


class AudioEncoder:
    """Produces ASR transcript and acoustic-event features for an audio track."""

    def __init__(self, config: AudioEncoderConfig | None = None) -> None:
        self.config = config or AudioEncoderConfig()
        self._asr = None
        self._clap = None

    def load(self) -> None:
        # TODO: load Whisper + CLAP onto self.config.device.
        raise NotImplementedError

    def transcribe(self, waveform: np.ndarray) -> tuple[str, list[dict]]:
        """Return (full_transcript, timestamped_segments)."""
        # TODO: run Whisper.
        raise NotImplementedError

    def encode_events(self, waveform: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Return (clap_embedding, {query: similarity})."""
        # TODO: run CLAP audio encoder + text queries.
        raise NotImplementedError

    def encode(self, waveform: np.ndarray) -> AudioFeatures:
        transcript, segments = self.transcribe(waveform)
        emb, scores = self.encode_events(waveform)
        return AudioFeatures(transcript, segments, emb, scores)
