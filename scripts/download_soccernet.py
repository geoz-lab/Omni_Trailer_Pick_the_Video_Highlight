"""Download SoccerNet broadcast videos (NDA password required; research use only).

The password is read from the SOCCERNET_PASSWORD env var (or .env) — do NOT hardcode
or commit it.

  export SOCCERNET_PASSWORD=...            # your NDA password
  python scripts/download_soccernet.py --dir $SCRATCH/omni_data/soccernet --split valid --res 224p

Notes:
- This downloads EVERY game in the chosen split(s); SoccerNet is large. Start with
  one split (e.g. valid) and/or Ctrl-C once enough games have landed to prototype.
- Each game = two ~45-min halves with commentary + crowd audio.
- 224p keeps size/compute down (fine for our low-fps pipeline); 720p also exists.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.env import load_env_file   # noqa: E402

load_env_file()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="local directory (use $SCRATCH, not the repo)")
    p.add_argument("--split", nargs="+", default=["valid"], help="train / valid / test")
    p.add_argument("--res", default="224p", choices=["224p", "720p"])
    p.add_argument("--password", default=None, help="overrides $SOCCERNET_PASSWORD")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pw = args.password or os.environ.get("SOCCERNET_PASSWORD")
    if not pw:
        sys.exit("Set the NDA password:  export SOCCERNET_PASSWORD=...  (or put it in .env)")

    from SoccerNet.Downloader import SoccerNetDownloader

    d = SoccerNetDownloader(LocalDirectory=args.dir)
    d.password = pw
    files = [f"1_{args.res}.mkv", f"2_{args.res}.mkv"]
    print(f"Downloading {files} for split(s) {args.split} -> {args.dir}")
    d.downloadGames(files=files, split=args.split)
    print("Done. Next: build_manifest.py --videos-dir", args.dir, "--segment 90 --require-audio")


if __name__ == "__main__":
    main()
