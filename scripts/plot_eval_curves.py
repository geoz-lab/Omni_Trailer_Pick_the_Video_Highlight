"""Plot test-set evaluation curves vs GRPO training steps.

Reads the JSONL written by ``evaluate_testset.py`` (one summary per tag) and draws
two panels: (left) mean reward + the six quality axes, (right) what GRPO actually
learned — mean clip length toward the 15 s target and the malformed rate.

Usage:
    python scripts/plot_eval_curves.py --input eval_results.jsonl --output Omni_Trailer_Results.png

Each summary's training-step count is parsed from its ``tag`` (base -> 0,
grpo50 -> 50, grpo300 -> 300, ...).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXES = ["excitement", "emotional_impact", "story_completeness",
        "relevance_to_video", "audiovisual_alignment", "trailer_quality"]
AXIS_LABEL = {a: a.replace("_", " ") for a in AXES}
TARGET_CLIP_SECONDS = 15.0


def steps_of(tag: str) -> int:
    if tag.lower() in ("base", "baseline"):
        return 0
    m = re.search(r"(\d+)", tag)
    return int(m.group(1)) if m else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="eval_results.jsonl")
    ap.add_argument("--output", default="Omni_Trailer_Results.png")
    args = ap.parse_args()

    rows = []
    seen = {}  # keep the last summary per tag
    for line in Path(args.input).read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            seen[r["tag"]] = r
    rows = sorted(seen.values(), key=lambda r: steps_of(r["tag"]))

    xs = [steps_of(r["tag"]) for r in rows]
    reward = [r["mean_reward"] for r in rows]
    malformed = [r["malformed_rate"] for r in rows]
    duration = [r["mean_duration"] for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # --- left: reward (headline) + the six quality axes ---
    for a in AXES:
        axL.plot(xs, [r["axes_mean"].get(a, float("nan")) for r in rows],
                 marker="o", ms=4, lw=1.3, alpha=0.75, label=AXIS_LABEL[a])
    axL.plot(xs, reward, marker="o", ms=7, lw=3, color="black", label="mean reward", zorder=5)
    axL.set_title("Reward & quality axes vs GRPO steps", fontweight="bold")
    axL.set_xlabel("GRPO training steps")
    axL.set_ylabel("Gemini 2.5 Pro score / reward")
    axL.set_xticks(xs)
    axL.grid(alpha=0.3)
    axL.legend(fontsize=8, ncol=2, loc="upper right")

    # --- right: what GRPO actually learned (length + format) ---
    axR.plot(xs, duration, marker="s", ms=6, lw=2.2, color="#1f77b4", label="mean clip length (s)")
    axR.axhline(TARGET_CLIP_SECONDS, ls="--", lw=1.4, color="#1f77b4", alpha=0.6,
                label=f"target {TARGET_CLIP_SECONDS:.0f} s")
    axR.set_xlabel("GRPO training steps")
    axR.set_ylabel("mean clip length (s)", color="#1f77b4")
    axR.tick_params(axis="y", labelcolor="#1f77b4")
    axR.set_xticks(xs)
    axR.set_ylim(0, max(TARGET_CLIP_SECONDS, max(duration)) * 1.15)
    axR.grid(alpha=0.3)

    axR2 = axR.twinx()
    axR2.plot(xs, malformed, marker="^", ms=6, lw=2.2, color="#d62728", label="malformed rate")
    axR2.set_ylabel("malformed rate", color="#d62728")
    axR2.tick_params(axis="y", labelcolor="#d62728")
    axR2.set_ylim(0, max(0.1, max(malformed) * 1.4))
    axR.set_title("What GRPO learned: length + valid format", fontweight="bold")

    lines = axR.get_lines()[:2] + axR2.get_lines()
    axR.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
