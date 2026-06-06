"""Multimodal fusion.

In the unified-omni architecture there is no separately trained fusion module:
Qwen2.5-Omni aligns video + audio + text in pretraining. This adapter packs a
video (with its audio) and a text prompt into the chat-format inputs the omni
processor expects, using ``qwen_omni_utils.process_mm_info``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert sports-trailer editor. You watch a video with its audio "
    "and pick the single most exciting, emotional, trailer-worthy moment."
)


@dataclass
class FusionInputs:
    """A video (audio travels inside it) plus optional precomputed text."""
    video_path: str
    transcript: str | None = None
    captions: list[dict] | None = None   # [{start, end, text}, ...]


class OmniFusion:
    """Builds Qwen2.5-Omni processor inputs (no learned fusion params)."""

    def __init__(self, processor: Any, frame_rate: float = 2.0, use_audio_in_video: bool = True,
                 video_max_pixels: int = 200704) -> None:
        self.processor = processor
        self.frame_rate = frame_rate
        self.use_audio_in_video = use_audio_in_video
        self.video_max_pixels = video_max_pixels
        # resolved per build_inputs (audio is dropped for clips with no audio track)
        self.last_use_audio = use_audio_in_video

    def _conversation(self, sample: FusionInputs, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    # max_pixels caps per-frame resolution -> bounds the number of
                    # vision tokens, whose self-attention is O(tokens^2) in memory.
                    {"type": "video", "video": sample.video_path, "fps": self.frame_rate,
                     "max_pixels": self.video_max_pixels},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    def build_inputs(self, sample: FusionInputs, prompt: str) -> dict:
        """Return a dict of model-ready tensors for the omni thinker.

        Audio is used only if requested AND the clip actually has an audio track
        (many stock clips don't); the resolved choice is stored in
        ``self.last_use_audio`` so generation/forward stay consistent.
        """
        from qwen_omni_utils import process_mm_info

        from ..utils.audio_utils import has_audio

        use_audio = self.use_audio_in_video and has_audio(sample.video_path)
        self.last_use_audio = use_audio

        conversation = self._conversation(sample, prompt)
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio,
        )
        return inputs
