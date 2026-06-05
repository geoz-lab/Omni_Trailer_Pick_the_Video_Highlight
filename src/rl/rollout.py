"""Rollout: turn a video into a GRPO group of scored candidate clips.

For each training video we:
  1. propose coarse candidate windows (cheap proposal encoders),
  2. sample `group_size` highlight completions from the policy,
  3. cut each candidate clip,
  4. score it with the reward model (VLM judge).

The resulting group of (completion, logprobs, reward) tuples is what GRPO
consumes — advantages are computed *relative to the group mean*.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..omni_model.fusion_model import FusionInputs
from ..omni_model.trailer_selector import HighlightSpan, TrailerSelector
from ..reward.reward_model import RewardModel
from ..video_cut.video_exporter import export_clip


@dataclass
class Candidate:
    span: HighlightSpan | None
    token_ids: list[int]
    logprobs: list[float]
    reward: float
    axes: dict[str, float]


def rollout_group(
    sample: FusionInputs,
    inputs: dict,
    selector: TrailerSelector,
    reward_model: RewardModel,
    video_summary: str,
    group_size: int,
    workdir: str,
    malformed_penalty: float = -1.0,
) -> list[Candidate]:
    """Produce one GRPO group of scored candidates for a single video."""
    raw = selector.propose_candidates(inputs, group_size)
    candidates: list[Candidate] = []
    for i, roll in enumerate(raw):
        span = selector.parse(roll["text"])
        if span is None:
            # Malformed boundaries get a fixed penalty so the policy learns format.
            candidates.append(
                Candidate(None, roll["token_ids"], roll["logprobs"], malformed_penalty, {})
            )
            continue
        clip_path = export_clip(sample.video_path, span.start_s, span.end_s, f"{workdir}/cand_{i}.mp4")
        scored = reward_model.score(clip_path, video_summary, span.duration)
        candidates.append(
            Candidate(span, roll["token_ids"], roll["logprobs"], scored["reward"], scored["axes"])
        )
    return candidates
