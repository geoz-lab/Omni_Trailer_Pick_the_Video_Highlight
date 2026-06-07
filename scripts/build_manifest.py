"""Build a GRPO manifest from a folder of videos (any source: SoccerNet, TVSum,
YouTube Highlights, Mr.HiSum, open clips...).

- Whole-video mode (default): one manifest line per video file.
- Segment mode (--segment N): cut long videos into N-second clips (e.g. SoccerNet
  full halves -> 90s clips) and manifest each clip.
- --require-audio drops clips with no audio track.

    # individual videos (TVSum/YouTube Highlights/demo), trim to <=90s, audio only
    python scripts/build_manifest.py --videos-dir $SCRATCH/omni_data/tvsum \
        --out data/metadata/tvsum.jsonl --max-seconds 90 --require-audio --summary "TVSum video"

    # SoccerNet full halves -> 90s clips written under <dir>/clips
    python scripts/build_manifest.py --videos-dir $SCRATCH/omni_data/soccernet \
        --out data/metadata/soccernet.jsonl --segment 90 --require-audio \
        --summary "Soccer match broadcast (SoccerNet)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.audio_utils import has_audio          # noqa: E402
from src.utils.video_utils import probe_duration      # noqa: E402
from src.video_cut.video_exporter import export_clip  # noqa: E402

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", required=True, help="folder of source videos (recursed)")
    p.add_argument("--out", required=True, help="manifest path to write (JSONL)")
    p.add_argument("--summary", default="", help="summary text stored for every clip (judge relevance context)")
    p.add_argument("--segment", type=int, default=0, help="cut into N-second clips (0 = keep whole video)")
    p.add_argument("--max-seconds", type=float, default=None, help="trim whole videos to this length (whole-video mode)")
    p.add_argument("--min-seconds", type=float, default=3.0, help="skip clips shorter than this")
    p.add_argument("--require-audio", action="store_true", help="skip videos/clips with no audio track")
    p.add_argument("--clips-dir", default=None, help="where to write segmented clips (default: <videos-dir>/clips)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.videos_dir)
    vids = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and ".clips" not in p.parts)
    if not vids:
        print(f"No videos found under {root}")
        return
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(args.clips_dir) if args.clips_dir else root / "clips"

    n_written = n_skip_audio = n_skip_short = 0
    with open(out, "w") as man:
        for v in vids:
            try:
                dur = probe_duration(str(v))
            except Exception as exc:  # noqa: BLE001
                print(f"  skip (probe failed): {v} ({exc})")
                continue
            if args.require_audio and not has_audio(str(v)):
                n_skip_audio += 1
                continue

            if args.segment and dur > args.segment:
                clips_dir.mkdir(parents=True, exist_ok=True)
                n = int(dur // args.segment)
                for i in range(n):
                    start = i * args.segment
                    dst = clips_dir / f"{v.stem}_{i:03d}.mp4"
                    export_clip(str(v), start, start + args.segment, str(dst))
                    man.write(json.dumps({"video": str(dst), "summary": args.summary}) + "\n")
                    n_written += 1
            else:
                if dur < args.min_seconds:
                    n_skip_short += 1
                    continue
                if args.max_seconds and dur > args.max_seconds:
                    dst = clips_dir / f"{v.stem}_trim.mp4"
                    clips_dir.mkdir(parents=True, exist_ok=True)
                    export_clip(str(v), 0.0, args.max_seconds, str(dst))
                    man.write(json.dumps({"video": str(dst), "summary": args.summary}) + "\n")
                else:
                    man.write(json.dumps({"video": str(v), "summary": args.summary}) + "\n")
                n_written += 1

    print(f"wrote {n_written} clips -> {out}")
    if n_skip_audio:
        print(f"  skipped {n_skip_audio} videos with no audio track")
    if n_skip_short:
        print(f"  skipped {n_skip_short} videos shorter than {args.min_seconds}s")
    print("Next: scripts/split_manifest.py to make train/test, then point configs/train_rl.yaml at it.")


if __name__ == "__main__":
    main()
