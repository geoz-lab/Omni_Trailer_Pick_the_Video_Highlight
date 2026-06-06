"""Audio I/O helpers: extract and resample the audio track via ffmpeg."""
from __future__ import annotations

import subprocess

import numpy as np

from ..video_cut.video_exporter import ffmpeg_exe


def has_audio(video_path: str) -> bool:
    """True if the video has at least one audio stream (ffmpeg prints 'Audio:')."""
    # `ffmpeg -i` returns non-zero (no output specified) but lists streams on stderr.
    proc = subprocess.run([ffmpeg_exe(), "-i", video_path], capture_output=True, text=True)
    return "Audio:" in proc.stderr


def extract_waveform(video_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Extract a mono waveform (float32 in [-1, 1]) from a video's audio track."""
    raw = subprocess.run(
        [
            ffmpeg_exe(), "-i", video_path,
            "-vn",
            "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
            "-",
        ],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
