"""Download videos with yt-dlp (research use only — respect each platform's terms
and copyright). Use it to fetch soccer/football clips while SoccerNet access is
pending, or to pull Mr.HiSum's YouTube IDs.

  # by search query
  python scripts/download_youtube.py --search "soccer goals highlights" --limit 40 \
      --out $SCRATCH/omni_data/yt_soccer

  # by an explicit list of YouTube IDs/URLs (e.g. Mr.HiSum soccer subset, one per line)
  python scripts/download_youtube.py --ids soccer_ids.txt --out $SCRATCH/omni_data/yt_soccer

Then segment + manifest:
  python scripts/build_manifest.py --videos-dir $SCRATCH/omni_data/yt_soccer \
      --out data/metadata/all.jsonl --segment 90 --require-audio --summary "Soccer highlights (YouTube)"
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--search", help="search query (downloads the top --limit results)")
    src.add_argument("--ids", help="file with one YouTube ID or URL per line")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--limit", type=int, default=30, help="max videos for --search")
    p.add_argument("--max-height", type=int, default=480, help="cap resolution to keep size down")
    p.add_argument("--max-duration", type=int, default=1200, help="skip videos longer than this (s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not shutil.which("yt-dlp"):
        sys.exit("yt-dlp not found — install it first:  pip install yt-dlp")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # point yt-dlp at the bundled ffmpeg (for muxing video+audio into mp4)
    try:
        import imageio_ffmpeg
        ffmpeg_loc = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    except Exception:
        ffmpeg_loc = None

    cmd = [
        "yt-dlp",
        "-f", f"bv*[height<={args.max_height}]+ba/b[height<={args.max_height}]",
        "--merge-output-format", "mp4",
        "--match-filter", f"duration < {args.max_duration}",
        "-o", str(out / "%(id)s.%(ext)s"),
        "--download-archive", str(out / "downloaded.txt"),   # skip already-fetched on reruns
        "--no-overwrites", "--ignore-errors", "--no-playlist",
    ]
    if ffmpeg_loc:
        cmd += ["--ffmpeg-location", ffmpeg_loc]

    if args.search:
        cmd.append(f"ytsearch{args.limit}:{args.search}")
    else:
        ids = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()]
        cmd += ids

    print("running:", " ".join(cmd[:8]), "...")
    rc = subprocess.run(cmd).returncode
    n = len(list(out.glob("*.mp4")))
    print(f"\nyt-dlp exit={rc}; {n} mp4 files now in {out}")
    print("Next: build_manifest.py --videos-dir", out, "--segment 90 --require-audio")


if __name__ == "__main__":
    main()
