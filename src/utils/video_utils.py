"""Video I/O helpers: frame sampling and metadata probing."""
from __future__ import annotations

import json
import subprocess

import numpy as np


def probe_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def sample_frames(video_path: str, fps: float, start_s: float = 0.0, end_s: float | None = None) -> np.ndarray:
    """Sample frames at `fps` from [start_s, end_s] -> array [N, H, W, 3] uint8."""
    # TODO: decode with opencv / decord at the requested rate.
    raise NotImplementedError
