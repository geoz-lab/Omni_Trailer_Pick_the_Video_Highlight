"""Evaluate a policy on the held-out test set: pick a highlight per video, score
it with the Gemini judge, and report the average reward.

Run it for each condition to compare (Base vs GRPO-50 vs GRPO-100):

    # original model (no adapter)
    python scripts/evaluate_testset.py --eval-model --tag base
    # after 50 GRPO steps
    python scripts/evaluate_testset.py --eval-model --checkpoint checkpoints/adapter_50  --tag grpo50
    # after 100 GRPO steps
    python scripts/evaluate_testset.py --eval-model --checkpoint checkpoints/adapter_final --tag grpo100

Use --limit to evaluate a subset (faster / fewer API calls). Results are appended
to eval_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.omni_model.fusion_model import FusionInputs                       # noqa: E402
from src.omni_model.llm_thinker import OmniThinker, ThinkerConfig          # noqa: E402
from src.omni_model.trailer_selector import TrailerSelector                # noqa: E402
from src.reward.reward_model import RewardConfig, RewardModel              # noqa: E402
from src.reward.reward_prompts import AXES                                 # noqa: E402
from src.utils.env import load_env_file                                    # noqa: E402
from src.utils.logging_utils import get_logger                            # noqa: E402
from src.utils.video_utils import probe_duration                          # noqa: E402
from src.video_cut.video_exporter import export_clip                       # noqa: E402

load_env_file()
log = get_logger("evaluate_testset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/metadata/test.jsonl")
    p.add_argument("--checkpoint", default=None, help="LoRA adapter dir (omit for the base/original model)")
    p.add_argument("--model-config", default="configs/model.yaml")
    p.add_argument("--reward-config", default="configs/reward.yaml")
    p.add_argument("--eval-model", action="store_true", help="use judge.eval_model (gemini-2.5-pro)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N clips")
    p.add_argument("--tag", default="run", help="label for this condition (base / grpo50 / grpo100)")
    p.add_argument("--out", default="eval_results.jsonl")
    p.add_argument("--clips-dir", default="output/eval")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mcfg = yaml.safe_load(open(args.model_config))
    rcfg = yaml.safe_load(open(args.reward_config))
    tcfg, scfg = mcfg["thinker"], mcfg["selector"]
    j = rcfg["judge"]

    items = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        items = items[: args.limit]
    log.info("Evaluating %d clips | tag=%s | checkpoint=%s", len(items), args.tag, args.checkpoint or "BASE")

    thinker = OmniThinker(ThinkerConfig(
        backbone=tcfg["backbone"], frame_rate=tcfg.get("frame_rate", 2.0),
        video_max_pixels=tcfg.get("video_max_pixels", 200704),
        dtype=tcfg.get("dtype", "bfloat16"),
        attn_implementation=tcfg.get("attn_implementation", "flash_attention_2"),
    )).load(lora_path=args.checkpoint)
    selector = TrailerSelector(thinker, scfg.get("min_clip_seconds", 5), scfg.get("max_clip_seconds", 60))

    reward = RewardModel(RewardConfig(
        provider=j["provider"],
        model=(j.get("eval_model", j["model"]) if args.eval_model else j["model"]),
        api_key_env=j["api_key_env"], temperature=j.get("temperature", 0.0),
        max_retries=j.get("max_retries", 6), weights=rcfg.get("axes", {}),
        cache_dir=(rcfg.get("cache", {}) or {}).get("dir"),
        length_penalty=(rcfg.get("shaping", {}) or {}).get("length_penalty", 0.05),
        target_clip_seconds=(rcfg.get("shaping", {}) or {}).get("target_clip_seconds", 45),
    ))
    malformed_penalty = -1.0
    clips_dir = Path(args.clips_dir); clips_dir.mkdir(parents=True, exist_ok=True)

    rewards, axis_acc, durations, n_malformed = [], {a: [] for a in AXES}, [], 0
    for k, it in enumerate(items):
        video = it["video"]
        try:
            dur = probe_duration(video)
            span = selector.select(FusionInputs(video_path=video), total_duration=dur)
        except Exception as exc:  # noqa: BLE001
            log.warning("  [%d] inference failed for %s: %s", k, video, exc)
            continue
        if span is None:
            n_malformed += 1
            rewards.append(malformed_penalty)
            log.info("  [%d] malformed -> %.3f  %s", k, malformed_penalty, video)
            continue
        clip = export_clip(video, span.start_s, span.end_s, str(clips_dir / f"{Path(video).stem}_hl.mp4"))
        try:
            res = reward.score(clip, it.get("summary", ""), span.duration)
        except Exception as exc:  # noqa: BLE001
            log.warning("  [%d] judge failed for %s: %s", k, video, exc)
            continue
        rewards.append(res["reward"]); durations.append(span.duration)
        for a in AXES:
            axis_acc[a].append(res["axes"].get(a, 0.0))
        log.info("  [%d] reward %.3f (%.0fs)  %s", k, res["reward"], span.duration, Path(video).name)

    n = len(rewards)
    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint or "base",
        "judge": reward.config.model,
        "n": n,
        "mean_reward": round(statistics.mean(rewards), 4) if rewards else None,
        "malformed_rate": round(n_malformed / n, 3) if n else None,
        "mean_duration": round(statistics.mean(durations), 1) if durations else None,
        "axes_mean": {a: round(statistics.mean(v), 3) for a, v in axis_acc.items() if v},
    }
    print("\n=== test-set summary ===")
    print(json.dumps(summary, indent=2))
    with open(args.out, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
