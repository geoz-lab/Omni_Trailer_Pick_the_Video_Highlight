"""Pick the highlight from a single video with the omni model and export the clip.

Omni-only: requires the Qwen2.5-Omni weights + a GPU (A100/H100). Defaults run on
the bundled Ronaldo demo.

Usage:
    python scripts/run_inference.py
    python scripts/run_inference.py --video path/to.mp4 --output output --checkpoint ckpts/lora
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.omni_model.fusion_model import FusionInputs            # noqa: E402
from src.omni_model.llm_thinker import OmniThinker, ThinkerConfig  # noqa: E402
from src.omni_model.trailer_selector import TrailerSelector     # noqa: E402
from src.utils.logging_utils import get_logger                  # noqa: E402
from src.utils.video_utils import probe_duration                # noqa: E402
from src.video_cut.boundary_detection import detect_shot_boundaries, snap_boundaries  # noqa: E402
from src.video_cut.video_exporter import export_clip, video_to_gif  # noqa: E402

log = get_logger("run_inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="demo_video/Ronaldo_goal_demo.mp4")
    p.add_argument("--output", default="output")
    p.add_argument("--config", default="configs/model.yaml")
    p.add_argument("--checkpoint", default=None, help="optional trained LoRA adapter")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--frame-rate", type=float, default=None,
                   help="override fps fed to the model (lower for long videos)")
    p.add_argument("--max-pixels", type=int, default=None,
                   help="override per-frame pixel cap (lower for long videos)")
    p.add_argument("--gif", dest="gif", action="store_true", default=True)
    p.add_argument("--no-gif", dest="gif", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    tcfg, scfg = cfg["thinker"], cfg["selector"]
    stem = Path(args.video).stem

    log.info("Loading omni thinker %s ...", tcfg["backbone"])
    thinker = OmniThinker(ThinkerConfig(
        backbone=tcfg["backbone"],
        frame_rate=args.frame_rate or tcfg.get("frame_rate", 2.0),
        video_max_pixels=args.max_pixels or tcfg.get("video_max_pixels", 200704),
        dtype=tcfg.get("dtype", "bfloat16"),
        attn_implementation=tcfg.get("attn_implementation", "flash_attention_2"),
    )).load(lora_path=args.checkpoint)

    selector = TrailerSelector(
        thinker,
        min_clip_seconds=scfg.get("min_clip_seconds", 5),
        max_clip_seconds=scfg.get("max_clip_seconds", 60),
    )

    duration = probe_duration(args.video)
    log.info("Reasoning over %s (%.1fs) ...", args.video, duration)
    span = selector.select(FusionInputs(video_path=args.video), total_duration=duration)
    if span is None:
        log.error("Model did not return a valid <start>/<end> span. Try raising --max-new-tokens.")
        sys.exit(1)

    # Snap to the nearest shot boundaries for a clean cut.
    shots = detect_shot_boundaries(args.video)
    start_s, end_s = snap_boundaries(span.start_s, span.end_s, shots)
    log.info("Highlight: %.2fs -> %.2fs (%.1fs)", start_s, end_s, end_s - start_s)

    out_mp4 = str(Path(args.output) / f"{stem}_highlight.mp4")
    export_clip(args.video, start_s, end_s, out_mp4)
    log.info("Wrote %s", out_mp4)

    if args.gif:
        out_gif = str(Path(args.output) / f"{stem}_highlight.gif")
        video_to_gif(out_mp4, out_gif, fps=12, width=480)
        log.info("Wrote %s", out_gif)


if __name__ == "__main__":
    main()
