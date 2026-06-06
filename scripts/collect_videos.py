"""Collect short videos into data/raw_videos/ and append the GRPO manifest.

Two sources:

  # Real / topical clips from Pexels (free key: https://www.pexels.com/api/).
  # Put PEXELS_API_KEY in .env, then:
  python scripts/collect_videos.py --source pexels --query "soccer goal" --limit 10

  # Zero-config sample clips (stable public sample videos) for a quick smoke test:
  python scripts/collect_videos.py --source samples --limit 5

Each video is trimmed to --max-seconds (vision attention is O(tokens^2), so keep
clips short) and a line is appended to data/metadata/train.jsonl:
    {"video": "data/raw_videos/xxx.mp4", "summary": "..."}

Edit the summaries afterwards for better reward-model context if you like.
Respect each source's license; for shareable work prefer Pexels/owned footage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.env import load_env_file          # noqa: E402
from src.utils.video_utils import probe_duration  # noqa: E402
from src.video_cut.video_exporter import export_clip  # noqa: E402

load_env_file()

_UA = {"User-Agent": "Mozilla/5.0 (omni-trailer collect_videos)"}

# Stable, freely-licensed sample clips (Blender open movies + test footage, ~10s
# 720p). Good for a pipeline smoke test; for real RL training use Pexels or your
# own footage. To add more, append (url, summary) tuples — the run trims them.
SAMPLE_CLIPS = [
    ("https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",
     "A big rabbit and forest creatures in a comedic animated short"),
    ("https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_1MB.mp4",
     "Jellyfish drifting in deep blue water, calm underwater nature footage"),
    ("https://test-videos.co.uk/vids/sintel/mp4/h264/720/Sintel_720_10s_1MB.mp4",
     "A fantasy animated short with a girl and a dragon"),
    ("https://www.w3schools.com/html/mov_bbb.mp4",
     "Big Buck Bunny clip, animated forest scene"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["pexels", "samples"], default="samples")
    p.add_argument("--query", default="sports highlights",
                   help="Pexels query; comma-separate several to mix topics, e.g. 'soccer,basketball,surfing'")
    p.add_argument("--limit", type=int, default=5, help="max number of videos to fetch")
    p.add_argument("--max-seconds", type=float, default=60.0, help="trim each clip to this length")
    p.add_argument("--out", default="data/raw_videos")
    p.add_argument("--manifest", default="data/metadata/train.jsonl")
    p.add_argument("--max-width", type=int, default=1280, help="prefer Pexels files up to this width")
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _pexels_page(key: str, query: str, page: int, per_page: int = 80) -> dict:
    url = (f"https://api.pexels.com/videos/search?query={urllib.parse.quote(query)}"
           f"&per_page={per_page}&page={page}&size=medium")
    req = urllib.request.Request(url, headers={"Authorization": key, **_UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _pick_mp4(video: dict, max_width: int) -> str | None:
    mp4s = [f for f in video.get("video_files", []) if f.get("file_type") == "video/mp4"]
    if not mp4s:
        return None
    ok = [f for f in mp4s if (f.get("width") or 0) <= max_width]
    return max(ok or mp4s, key=lambda f: f.get("width") or 0)["link"]


def pexels_items(queries: list[str], limit: int, max_width: int) -> list[tuple[str, str, str]]:
    """Return up to `limit` (url, stem, summary) across one or more queries.

    Paginates the Pexels API (80/page) and spreads the budget over the queries so
    you can reach hundreds/thousands of clips. Dedupes by video id.
    """
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise RuntimeError("Set PEXELS_API_KEY (in .env) for --source pexels. "
                           "Get a free key at https://www.pexels.com/api/")
    per_query = max(1, -(-limit // len(queries)))   # ceil
    items: list[tuple[str, str, str]] = []
    seen: set[int] = set()
    for query in queries:
        got = 0
        page = 1
        while got < per_query and len(items) < limit:
            data = _pexels_page(key, query, page)
            videos = data.get("videos", [])
            if not videos:
                break
            for v in videos:
                vid = v["id"]
                if vid in seen:
                    continue
                link = _pick_mp4(v, max_width)
                if not link:
                    continue
                summary = f"{query} (Pexels stock video by {v.get('user', {}).get('name', 'unknown')})"
                items.append((link, f"pexels_{vid}", summary))
                seen.add(vid)
                got += 1
                if got >= per_query or len(items) >= limit:
                    break
            page += 1
            if page > (data.get("total_results", 0) // 80) + 1:
                break   # no more results for this query
        print(f"  query '{query}': collected {got}")
    return items[:limit]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    man_path = Path(args.manifest); man_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "pexels":
        queries = [q.strip() for q in args.query.split(",") if q.strip()]
        items = pexels_items(queries, args.limit, args.max_width)
    else:
        items = [(u, Path(u).stem, s) for u, s in SAMPLE_CLIPS[:args.limit]]

    if not items:
        print("No videos found.")
        return

    # avoid duplicate manifest lines
    existing = set()
    if man_path.exists():
        for line in man_path.read_text().splitlines():
            line = line.strip()
            if line:
                existing.add(json.loads(line)["video"])

    added = 0
    with open(man_path, "a") as man:
        for url, stem, summary in items:
            final = out_dir / f"{stem}.mp4"
            rel = str(final)
            if rel in existing:
                print(f"skip (already in manifest): {rel}")
                continue
            try:
                tmp = out_dir / f"{stem}.download.mp4"
                print(f"downloading {url}")
                _download(url, tmp)
                dur = probe_duration(str(tmp))
                if dur <= args.max_seconds:
                    tmp.replace(final)          # already short -> no re-encode
                else:
                    export_clip(str(tmp), 0.0, args.max_seconds, str(final))
                    tmp.unlink(missing_ok=True)
                man.write(json.dumps({"video": rel, "summary": summary}) + "\n")
                added += 1
                print(f"  saved {final}  ({min(dur, args.max_seconds):.0f}s)")
            except Exception as exc:  # noqa: BLE001 - keep going on a single failure
                print(f"  FAILED {url}: {exc}")

    print(f"\nAdded {added} clip(s) to {out_dir} and appended to {man_path}.")
    print("Tip: edit the summaries in the manifest for better reward-model context.")


if __name__ == "__main__":
    main()
