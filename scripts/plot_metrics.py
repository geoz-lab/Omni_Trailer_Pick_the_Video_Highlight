"""Summarize / plot GRPO training metrics from checkpoints/metrics.jsonl.

    python scripts/plot_metrics.py                          # print a table
    python scripts/plot_metrics.py --png checkpoints/reward.png   # also save a plot

Reads the JSONL written by MetricLogger (one {step, reward_mean, ...} per line).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="checkpoints/metrics.jsonl")
    p.add_argument("--png", default=None, help="optional path to save a reward plot (needs matplotlib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"No metrics file at {path} (set logging.save_dir / run training first).")
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print("metrics file is empty")
        return

    cols = ["step", "loss", "reward_mean", "reward_max", "kl", "malformed_frac"]
    print("  ".join(f"{c:>13}" for c in cols))
    print("-" * (15 * len(cols)))
    for r in rows:
        print("  ".join(f"{r.get(c, ''):>13.4f}" if isinstance(r.get(c), (int, float))
                         else f"{str(r.get(c, '')):>13}" for c in cols))

    rm = [r["reward_mean"] for r in rows if "reward_mean" in r]
    if rm:
        print(f"\nreward_mean: first={rm[0]:.4f}  last={rm[-1]:.4f}  "
              f"min={min(rm):.4f}  max={max(rm):.4f}")

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not installed; skipping --png")
            return
        steps = [r.get("step") for r in rows]
        plt.figure(figsize=(7, 4))
        for key in ("reward_mean", "reward_max"):
            ys = [r.get(key) for r in rows]
            plt.plot(steps, ys, marker="o", label=key)
        plt.xlabel("step"); plt.ylabel("reward"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(args.png, dpi=120)
        print(f"saved plot -> {args.png}")


if __name__ == "__main__":
    main()
