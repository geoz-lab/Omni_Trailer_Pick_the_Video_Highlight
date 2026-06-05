"""Snap highlight boundaries to natural cut points.

The policy predicts approximate start/end timestamps; this module nudges them to
the nearest shot boundary / silence so the exported clip doesn't start or end
mid-word or mid-shot.
"""
from __future__ import annotations


def detect_shot_boundaries(video_path: str, threshold: float = 0.3) -> list[float]:
    """Return shot-change timestamps (seconds) via frame-difference detection."""
    # TODO: PySceneDetect / frame histogram diff.
    raise NotImplementedError


def detect_silences(video_path: str, min_silence_s: float = 0.3) -> list[tuple[float, float]]:
    """Return (start, end) of silent intervals for clean audio cuts."""
    # TODO: librosa / ffmpeg silencedetect.
    raise NotImplementedError


def snap_boundaries(
    start_s: float,
    end_s: float,
    shot_boundaries: list[float],
    max_shift_s: float = 1.0,
) -> tuple[float, float]:
    """Move start/end to the nearest shot boundary within `max_shift_s`."""

    def nearest(t: float) -> float:
        if not shot_boundaries:
            return t
        cand = min(shot_boundaries, key=lambda b: abs(b - t))
        return cand if abs(cand - t) <= max_shift_s else t

    return nearest(start_s), nearest(end_s)
