"""Score a folder of candidate clips with the reward model (judge).

Useful for sanity-checking the judge and inspecting per-axis breakdowns.

Usage:
    python scripts/evaluate_reward.py --clips output --summary "Champions League final ..."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.reward.reward_model import RewardConfig, RewardModel  # noqa: E402
from src.reward.reward_prompts import AXES                     # noqa: E402
from src.utils.env import load_env_file                        # noqa: E402
from src.utils.video_utils import probe_duration              # noqa: E402

load_env_file()   # pick up GEMINI_API_KEY / OPENAI_API_KEY from .env if present


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", required=True, help="dir of candidate .mp4 clips")
    p.add_argument("--config", default="configs/reward.yaml")
    p.add_argument("--summary", default="", help="full-video summary for the relevance axis")
    p.add_argument("--eval-model", action="store_true",
                   help="use judge.eval_model (e.g. gemini-2.5-pro) instead of the RL-loop model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clips = sorted(Path(args.clips).glob("*.mp4"))
    if not clips:
        print(f"No .mp4 clips found in {args.clips}")
        return

    cfg = yaml.safe_load(open(args.config))
    j = cfg["judge"]
    model = j.get("eval_model", j["model"]) if args.eval_model else j["model"]
    reward = RewardModel(RewardConfig(
        provider=j["provider"], model=model, api_key_env=j["api_key_env"],
        temperature=j.get("temperature", 0.0), max_retries=j.get("max_retries", 4),
        weights=cfg.get("axes", {}),
        cache_dir=(cfg.get("cache", {}) or {}).get("dir"),
        length_penalty=(cfg.get("shaping", {}) or {}).get("length_penalty", 0.05),
    ))

    rows = []
    for clip in clips:
        res = reward.score(str(clip), args.summary, probe_duration(str(clip)))
        rows.append((clip.name, res["reward"], res["axes"]))

    rows.sort(key=lambda r: r[1], reverse=True)
    header = f"{'clip':<32}{'reward':>8}  " + "  ".join(f"{a[:8]:>8}" for a in AXES)
    print(header)
    print("-" * len(header))
    for name, rew, axes in rows:
        cells = "  ".join(f"{axes.get(a, 0.0):>8.2f}" for a in AXES)
        print(f"{name:<32}{rew:>8.3f}  {cells}")


if __name__ == "__main__":
    main()
