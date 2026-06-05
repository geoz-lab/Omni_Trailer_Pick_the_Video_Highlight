"""Score a folder of candidate clips with the reward model (judge).

Useful for sanity-checking the judge and inspecting per-axis breakdowns.

Usage:
    python scripts/evaluate_reward.py --clips examples/demo_output
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", required=True, help="dir of candidate .mp4 clips")
    p.add_argument("--config", default="configs/reward.yaml")
    p.add_argument("--summary", default="", help="full-video summary for relevance axis")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clips = sorted(Path(args.clips).glob("*.mp4"))
    if not clips:
        print(f"No .mp4 clips found in {args.clips}")
        return
    # TODO: build RewardModel from configs/reward.yaml; score each clip; print a
    #       table of per-axis scores + total reward, ranked best-first.
    raise NotImplementedError("Wire RewardModel scoring + reporting here.")


if __name__ == "__main__":
    main()
