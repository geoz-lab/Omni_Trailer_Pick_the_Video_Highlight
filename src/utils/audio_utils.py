"""Audio I/O helpers: extract and resample the audio track."""
from __future__ import annotations

import subprocess

import numpy as np


def extract_waveform(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract a mono waveform (float32, [-1, 1]) from a video's audio track."""
    raw = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
            "-",
        ],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)
