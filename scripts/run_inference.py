"""Pick the highlight from a single video and export the trailer clip.

Usage:
    python scripts/run_inference.py --video examples/demo_video.mp4 --output examples/demo_output
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="examples/demo_output")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--checkpoint", default=None, help="trained LoRA adapter (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO:
    #   1. load OmniThinker (+ optional trained LoRA from args.checkpoint)
    #   2. (optional) segment_proposal to narrow candidates
    #   3. TrailerSelector.select(...) -> HighlightSpan
    #   4. snap_boundaries -> export_clip into args.output
    raise NotImplementedError("Wire the inference pipeline here.")


if __name__ == "__main__":
    main()
