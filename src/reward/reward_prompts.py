"""Prompts and parsing for the VLM judge.

The judge receives a candidate clip (sampled frames + audio/transcript) plus a
short summary of the *full* video for the relevance axis, and returns a JSON
object with one score in [0, 1] per axis.
"""
from __future__ import annotations

import json

AXES = [
    "excitement",
    "emotional_impact",
    "story_completeness",
    "relevance_to_video",
    "audiovisual_alignment",
    "trailer_quality",
]

SYSTEM_PROMPT = (
    "You are an expert trailer editor. You judge a SHORT highlight clip cut from a "
    "longer video — the kind of punchy moment that would headline a trailer. The "
    "clip is short BY DESIGN: never penalize it for being brief or for not covering "
    "the whole video. Reward it for capturing the single most compelling moment. Be "
    "strict, calibrated, and consistent."
)

_RUBRIC = """This clip is a short highlight extracted from a longer video. Judge it
AS A TRAILER MOMENT, not as a summary. Do NOT reward length or completeness of
coverage — a tight, punchy clip should beat a long, sprawling one.

Score each axis from 0.0 (poor) to 1.0 (excellent):
- excitement: peak energy / thrill of THIS moment (intensity per second, not total).
- emotional_impact: how moving or affecting the moment is.
- story_completeness: does the moment itself land (a clear beat), even if brief?
  Do not penalize it for omitting the rest of the video.
- relevance_to_video: does it capture THE most important / defining moment of the
  video (the thing a viewer must see)? A representative-but-dull clip scores low.
- audiovisual_alignment: do visuals, audio and crowd/commentary reinforce the peak?
- trailer_quality: would this hook a viewer in seconds? Concise and high-impact
  scores high; slow or padded scores low.

Full-video context (for the relevance axis):
{video_summary}

Respond with ONLY a JSON object, e.g.:
{{"excitement": 0.9, "emotional_impact": 0.8, "story_completeness": 0.7,
  "relevance_to_video": 0.85, "audiovisual_alignment": 0.8, "trailer_quality": 0.88}}"""


def build_user_prompt(video_summary: str) -> str:
    return _RUBRIC.format(video_summary=video_summary)


def parse_scores(raw: str) -> dict[str, float]:
    """Extract the JSON scores from the judge's response; clamp to [0, 1]."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in judge response: {raw!r}")
    data = json.loads(raw[start : end + 1])
    scores = {}
    for axis in AXES:
        if axis not in data:
            raise ValueError(f"Judge response missing axis {axis!r}: {data}")
        scores[axis] = max(0.0, min(1.0, float(data[axis])))
    return scores
