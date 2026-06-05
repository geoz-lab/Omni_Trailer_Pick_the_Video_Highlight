"""Video I/O helpers: frame sampling and metadata probing (ffmpeg-backed)."""
from __future__ import annotations

import numpy as np


def probe_duration(video_path: str) -> float:
    """Return video duration in seconds."""
    import imageio_ffmpeg

    meta = next(imageio_ffmpeg.read_frames(video_path))  # first yield = metadata
    duration = meta.get("duration")
    if duration:
        return float(duration)
    fps, nframes = meta.get("fps"), meta.get("nframes")
    if fps and nframes and np.isfinite(nframes):
        return float(nframes) / float(fps)
    raise RuntimeError(f"Could not determine duration for {video_path!r}")


def sample_frames(
    video_path: str,
    fps: float,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> np.ndarray:
    """Sample frames at ``fps`` from ``[start_s, end_s]`` -> ``[N, H, W, 3]`` uint8."""
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(
        video_path, output_params=["-r", str(fps)]
    )
    meta = next(reader)
    w, h = meta["size"]
    frames: list[np.ndarray] = []
    for i, raw in enumerate(reader):
        t = i / fps
        if t < start_s:
            continue
        if end_s is not None and t > end_s:
            break
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
    return np.stack(frames) if frames else np.empty((0, h, w, 3), dtype=np.uint8)
