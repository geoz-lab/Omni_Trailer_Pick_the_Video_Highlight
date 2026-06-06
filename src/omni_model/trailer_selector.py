"""Highlight Segment Selector — the policy head.

The selector asks the omni thinker to emit the highlight boundaries as *text
tokens*, e.g.:

    <start>02:14</start><end>02:59</end>

Emitting timestamps as tokens (rather than a regression head) keeps the policy a
plain language model, so GRPO can score whole completions with a scalar reward
and update via group-relative advantages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .fusion_model import FusionInputs
from .llm_thinker import OmniThinker

_SELECTOR_PROMPT = (
    "You are an expert trailer editor. From this video, pick the single best "
    "~15-second highlight for a trailer — the most exciting, emotional, "
    "trailer-worthy moment.\n\n"
    "Output ONLY the start and end timestamps, in EXACTLY this format, and "
    "nothing else (no words, no explanation):\n"
    "<start>MM:SS</start><end>MM:SS</end>\n\n"
    "Example: <start>00:42</start><end>00:57</end>"
)

_BOUNDARY_RE = re.compile(r"<start>(\d+:\d+)</start>\s*<end>(\d+:\d+)</end>")


@dataclass
class HighlightSpan:
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


def _mmss_to_seconds(value: str) -> float:
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


class TrailerSelector:
    """Turns omni-model completions into highlight spans (and back, for RL)."""

    def __init__(
        self,
        thinker: OmniThinker,
        min_clip_seconds: float = 5.0,
        max_clip_seconds: float = 60.0,
    ) -> None:
        self.thinker = thinker
        self.min_clip_seconds = min_clip_seconds
        self.max_clip_seconds = max_clip_seconds

    def parse(self, completion: str) -> HighlightSpan | None:
        """Parse `<start>/<end>` tokens; return None if malformed."""
        match = _BOUNDARY_RE.search(completion)
        if not match:
            return None
        start_s, end_s = (_mmss_to_seconds(g) for g in match.groups())
        if not self.min_clip_seconds <= (end_s - start_s) <= self.max_clip_seconds:
            return None
        return HighlightSpan(start_s, end_s)

    def select(self, sample: FusionInputs, total_duration: float | None = None) -> HighlightSpan | None:
        """Greedy single-best selection (inference path).

        If ``total_duration`` is given, the span is clamped into ``[0, duration]``.
        """
        completion = self.thinker.reason(sample, _SELECTOR_PROMPT)
        span = self.parse(completion)
        if span is not None and total_duration is not None:
            start = max(0.0, min(span.start_s, total_duration))
            end = max(start, min(span.end_s, total_duration))
            span = HighlightSpan(start, end)
        return span

    def propose_candidates(self, inputs: dict, group_size: int, **gen_kwargs) -> list[dict]:
        """Sample `group_size` candidate completions for a GRPO rollout group.

        Returns raw rollout dicts {text, token_ids, logprobs}; parsing into spans
        and rewarding happens in `rl.rollout`.
        """
        return [self.thinker.generate(inputs, **gen_kwargs) for _ in range(group_size)]
