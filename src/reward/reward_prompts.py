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
    "You are an expert trailer editor and film critic. You evaluate whether a "
    "short candidate clip works as the single highlight of a longer video. Be "
    "strict, calibrated, and consistent."
)

_RUBRIC = """Score the candidate clip on each axis from 0.0 (poor) to 1.0 (excellent):
- excitement: peak energy / thrill of the moment.
- emotional_impact: how moving or affecting it is.
- story_completeness: does it stand alone with a beginning-middle-end?
- relevance_to_video: how representative it is of the full video below.
- audiovisual_alignment: do visuals, audio and speech reinforce each other?
- trailer_quality: would this hook a viewer as a trailer?

Full-video summary:
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
