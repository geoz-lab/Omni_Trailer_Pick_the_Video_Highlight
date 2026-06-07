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
    p.add_argument("--filter", default=None,
                   help="only games whose path contains this (case-insensitive), "
                        "e.g. 'champions-league', 'real madrid', 'liverpool'")
    p.add_argument("--limit", type=int, default=None, help="download at most N matching games")
    p.add_argument("--list", action="store_true", help="just list matching games, don't download")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from SoccerNet.Downloader import SoccerNetDownloader

    d = SoccerNetDownloader(LocalDirectory=args.dir)
    files = [f"1_{args.res}.mkv", f"2_{args.res}.mkv"]

    # --list / --filter: pick specific games (e.g. Champions League, a team)
    if args.list or args.filter:
        games = []
        for sp in args.split:
            try:
                games += d.getListGames(sp)
            except Exception as exc:  # noqa: BLE001
                print(f"getListGames({sp}) failed: {exc}")
        if args.filter:
            f = args.filter.lower()
            games = [g for g in games if f in g.lower()]
        if args.limit:
            games = games[: args.limit]
        print(f"{len(games)} matching game(s):")
        for g in games:
            print("  ", g)
        if args.list:
            return
        pw = args.password or os.environ.get("SOCCERNET_PASSWORD")
        if not pw:
            sys.exit("Set SOCCERNET_PASSWORD to download (or use --list to just browse).")
        d.password = pw
        for g in games:
            try:
                d.downloadGame(files=files, game=g)
            except Exception as exc:  # noqa: BLE001
                sys.exit(f"per-game download not supported by this SoccerNet version ({exc}).\n"
                         f"Fall back to a whole split:  python scripts/download_soccernet.py "
                         f"--dir {args.dir} --split {' '.join(args.split)}  (then delete games you don't want).")
        print("Done.")
        return

    # default: download whole split(s)
    pw = args.password or os.environ.get("SOCCERNET_PASSWORD")
    if not pw:
        sys.exit("Set the NDA password:  export SOCCERNET_PASSWORD=...  (or put it in .env)")
    d.password = pw
    print(f"Downloading {files} for split(s) {args.split} -> {args.dir}")
    d.downloadGames(files=files, split=args.split)
    print("Done. Next: build_manifest.py --videos-dir", args.dir, "--segment 90 --require-audio --copy")


if __name__ == "__main__":
    main()
