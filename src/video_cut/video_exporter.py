"""Export a highlight span to an .mp4 clip and render GIF previews (ffmpeg).

The ffmpeg binary is resolved from the ``imageio-ffmpeg`` wheel when present
(so no system ffmpeg / brew is required), falling back to ``ffmpeg`` on PATH.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Return a usable ffmpeg executable path."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def export_clip(video_path: str, start_s: float, end_s: float, out_path: str) -> str:
    """Cut ``[start_s, end_s]`` from ``video_path`` into ``out_path``; return it.

    Re-encodes for frame-accurate cuts (stream copy can be off by a GOP). Swap to
    ``-c copy`` if speed matters more than precision.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_s - start_s)
    cmd = [
        ffmpeg_exe(), "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def video_to_gif(
    src: str,
    out_gif: str,
    fps: int = 12,
    width: int = 480,
    start_s: float | None = None,
    end_s: float | None = None,
) -> str:
    """Render ``src`` (optionally a sub-span) to a small, clean looping GIF.

    Uses a two-pass palettegen/paletteuse filter so colours stay crisp and the
    file stays small enough to embed in the README.
    """
    out = Path(out_gif)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette = out.with_suffix(".palette.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"

    trim: list[str] = []
    if start_s is not None:
        trim += ["-ss", f"{start_s:.3f}"]
    if start_s is not None and end_s is not None:
        trim += ["-t", f"{max(0.0, end_s - start_s):.3f}"]

    ff = ffmpeg_exe()
    # Pass 1: build an optimal 256-colour palette.
    subprocess.run(
        [ff, "-y", *trim, "-i", src, "-vf", f"{vf},palettegen", str(palette)],
        check=True, capture_output=True,
    )
    # Pass 2: apply the palette.
    subprocess.run(
        [ff, "-y", *trim, "-i", src, "-i", str(palette),
         "-lavfi", f"{vf}[x];[x][1:v]paletteuse", str(out)],
        check=True, capture_output=True,
    )
    palette.unlink(missing_ok=True)
    return str(out)
