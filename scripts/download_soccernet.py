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


def _catalog_games(splits) -> list | None:
    """Full list of SoccerNet game paths for the splits (getListGames is a module
    function in current builds). Returns None if the API isn't available."""
    fn = None
    for modpath in ("SoccerNet.utils", "SoccerNet.DataLoader", "SoccerNet"):
        try:
            mod = __import__(modpath, fromlist=["getListGames"])
            fn = getattr(mod, "getListGames", None)
            if fn:
                break
        except Exception:
            continue
    if not fn:
        return None
    games: list = []
    for sp in splits:
        try:
            games += list(fn(sp))
        except Exception:
            pass
    return games or None


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

    # --list / --filter: browse the SoccerNet catalog (getListGames is a module
    # function in current builds, not a downloader method) and optionally download
    # the matching games selectively.
    if args.list or args.filter:
        catalog = _catalog_games(args.split)
        if catalog is None:                       # API unavailable -> browse disk
            mkvs = sorted(Path(args.dir).rglob("*.mkv"))
            games = sorted({str(p.parent) for p in mkvs})
            source = "on disk"
        else:
            games = catalog
            source = "in catalog"
        if args.filter:
            f = args.filter.lower()
            games = [g for g in games if f in g.lower()]
        if args.limit:
            games = games[: args.limit]
        print(f"{len(games)} game(s) {source}" + (f" matching '{args.filter}'" if args.filter else "") + ":")
        for g in games:
            print("  ", g)
        if args.list or catalog is None:
            if catalog is None:
                print("\n(catalog API unavailable — download a whole split, then "
                      "build_manifest.py --filter '<substr>' to train on a subset)")
            return
        # selective download of the matched catalog games
        pw = args.password or os.environ.get("SOCCERNET_PASSWORD")
        if not pw:
            sys.exit("Set SOCCERNET_PASSWORD to download (or use --list to browse).")
        d.password = pw
        for g in games:
            try:
                d.downloadGame(files=files, game=g)
            except Exception as exc:  # noqa: BLE001
                sys.exit(f"per-game download failed ({type(exc).__name__}: {exc}).\n"
                         f"Paste the introspection output and I'll wire it; meanwhile download a "
                         f"whole split and use build_manifest.py --filter '{args.filter}'.")
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
