"""Export a highlight span to an .mp4 clip (ffmpeg)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def export_clip(video_path: str, start_s: float, end_s: float, out_path: str) -> str:
    """Cut [start_s, end_s] from `video_path` into `out_path`; return out_path.

    Uses ffmpeg with re-encoding for frame-accurate cuts (stream copy can be
    off by a GOP). Swap to `-c copy` if speed matters more than precision.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_s - start_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
