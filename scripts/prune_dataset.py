"""Prune the training manifest: drop clips that lack an audio track and/or whose
topic (from the manifest summary) isn't wanted.

Audit first (no changes):
    python scripts/prune_dataset.py

Apply (rewrite manifest; optionally delete dropped files):
    python scripts/prune_dataset.py --apply --require-audio \
        --drop-topics "wildlife,dancing,concert" --delete

Or keep only certain topics:
    python scripts/prune_dataset.py --apply --keep-topics "soccer,football,fifa,world cup" --require-audio --delete
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.audio_utils import has_audio   # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/metadata/train.jsonl")
    p.add_argument("--require-audio", action="store_true", help="drop clips with no audio track")
    p.add_argument("--drop-topics", default="", help="comma list; drop if the summary contains any")
    p.add_argument("--keep-topics", default="", help="comma list; keep ONLY if the summary contains any")
    p.add_argument("--apply", action="store_true", help="rewrite the manifest (otherwise just audit)")
    p.add_argument("--delete", action="store_true", help="also delete dropped video files (with --apply)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    man = Path(args.manifest)
    rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
    drop_topics = [t.strip().lower() for t in args.drop_topics.split(",") if t.strip()]
    keep_topics = [t.strip().lower() for t in args.keep_topics.split(",") if t.strip()]

    kept, dropped = [], []
    n_no_audio = n_no_file = 0
    for r in rows:
        summ = r.get("summary", "").lower()
        reason = None
        if keep_topics and not any(t in summ for t in keep_topics):
            reason = "topic"
        elif drop_topics and any(t in summ for t in drop_topics):
            reason = "topic"
        elif args.require_audio:
            if not Path(r["video"]).exists():
                reason = "missing-file"; n_no_file += 1
            elif not has_audio(r["video"]):
                reason = "no-audio"; n_no_audio += 1
        (dropped if reason else kept).append((r, reason))

    print(f"manifest: {man}  ({len(rows)} entries)")
    print(f"  keep : {len(kept)}")
    print(f"  drop : {len(dropped)}   (no-audio={n_no_audio}, missing-file={n_no_file}, "
          f"topic={sum(1 for _, why in dropped if why == 'topic')})")

    if not args.apply:
        print("\n(audit only — re-run with --apply to rewrite the manifest; add --delete to remove files)")
        return

    # back up, rewrite manifest with kept rows
    man.rename(man.with_suffix(".jsonl.bak"))
    man.write_text("".join(json.dumps(r) + "\n" for r, _ in kept))
    print(f"\nwrote {len(kept)} entries to {man} (backup: {man.with_suffix('.jsonl.bak')})")

    if args.delete:
        removed = 0
        for r, _ in dropped:
            p = Path(r["video"])
            if p.exists():
                p.unlink(); removed += 1
        print(f"deleted {removed} dropped video files")


if __name__ == "__main__":
    main()
