"""Snap highlight boundaries to natural cut points.

The policy predicts approximate start/end timestamps; this module nudges them to
the nearest shot boundary / silence so the exported clip doesn't start or end
mid-shot. Shot detection is a lightweight frame-difference method (ffmpeg-backed,
runs anywhere) so it needs no extra heavy dependency.
"""
from __future__ import annotations

import numpy as np


def _to_gray(frames: np.ndarray) -> np.ndarray:
    """[N, H, W, 3] uint8 -> [N, H, W] float luminance."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return frames.astype(np.float32) @ weights


def detect_shot_boundaries(video_path: str, sample_fps: float = 4.0, z: float = 3.0) -> list[float]:
    """Return shot-change timestamps (seconds) via frame-difference detection.

    A boundary is flagged when the mean absolute inter-frame difference exceeds
    ``mean + z * std`` of the difference signal.
    """
    from ..utils.video_utils import sample_frames

    frames = sample_frames(video_path, fps=sample_fps)
    if len(frames) < 3:
        return []
    gray = _to_gray(frames)
    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    thresh = diffs.mean() + z * diffs.std()
    # diff[i] is the change between frame i and i+1 -> boundary at time (i+1)/fps
    return [float((i + 1) / sample_fps) for i in np.where(diffs > thresh)[0]]


def detect_silences(video_path: str, sample_rate: int = 16000,
                    min_silence_s: float = 0.3, rms_db: float = -35.0) -> list[tuple[float, float]]:
    """Return (start, end) of silent intervals for clean audio cuts."""
    from ..utils.audio_utils import extract_waveform

    wf = extract_waveform(video_path, sample_rate)
    win = max(1, int(0.02 * sample_rate))                 # 20 ms frames
    n = len(wf) // win
    if n == 0:
        return []
    rms = np.sqrt((wf[: n * win].reshape(n, win) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    silent = db < rms_db
    out: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if (j - i) * 0.02 >= min_silence_s:
                out.append((i * 0.02, j * 0.02))
            i = j
        else:
            i += 1
    return out


def snap_boundaries(
    start_s: float,
    end_s: float,
    shot_boundaries: list[float],
    max_shift_s: float = 1.0,
) -> tuple[float, float]:
    """Move start/end to the nearest shot boundary within ``max_shift_s``."""

    def nearest(t: float) -> float:
        if not shot_boundaries:
            return t
        cand = min(shot_boundaries, key=lambda b: abs(b - t))
        return cand if abs(cand - t) <= max_shift_s else t

    return nearest(start_s), nearest(end_s)
