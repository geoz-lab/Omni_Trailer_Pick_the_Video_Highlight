"""Train the trailer selector with GRPO (runs on the GPU cluster).

Usage:
    python scripts/train_rl.py --config configs/train_rl.yaml

Manifest format (one JSON object per line), e.g. data/metadata/train.jsonl:
    {"video": "data/raw_videos/match1.mp4", "summary": "Champions League final ..."}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.omni_model.fusion_model import FusionInputs                       # noqa: E402
from src.omni_model.llm_thinker import OmniThinker, ThinkerConfig          # noqa: E402
from src.omni_model.trailer_selector import TrailerSelector, _SELECTOR_PROMPT  # noqa: E402
from src.reward.reward_model import RewardConfig, RewardModel              # noqa: E402
from src.rl.grpo_trainer import GRPOConfig, GRPOTrainer                    # noqa: E402
from src.rl.rollout import rollout_group                                   # noqa: E402
from src.utils.env import load_env_file                                    # noqa: E402
from src.utils.logging_utils import MetricLogger                          # noqa: E402

load_env_file()   # pick up GEMINI_API_KEY / WANDB_API_KEY from .env if present


def read_manifest(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_thinker(model_cfg: dict, trainable_lora: dict | None) -> OmniThinker:
    tc = model_cfg["thinker"]
    thinker = OmniThinker(ThinkerConfig(
        backbone=tc["backbone"], frame_rate=tc.get("frame_rate", 2.0),
        dtype=tc.get("dtype", "bfloat16"),
        attn_implementation=tc.get("attn_implementation", "flash_attention_2"),
    )).load()
    if trainable_lora and trainable_lora.get("use_lora", True):
        from peft import LoraConfig, get_peft_model

        thinker.model = get_peft_model(thinker.model, LoraConfig(
            r=trainable_lora.get("lora_r", 16),
            lora_alpha=trainable_lora.get("lora_alpha", 32),
            lora_dropout=trainable_lora.get("lora_dropout", 0.05),
            target_modules=trainable_lora.get("target_modules"),
            task_type="CAUSAL_LM",
        ))
        thinker.model.print_trainable_parameters()
    return thinker


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_rl.yaml")
    args = ap.parse_args()

    import torch

    cfg = yaml.safe_load(open(args.config))
    assert cfg["algorithm"] == "grpo", "default recipe is GRPO; see ppo_trainer for PPO"
    model_cfg = yaml.safe_load(open(cfg["model_config"]))
    reward_cfg = yaml.safe_load(open(cfg["reward_config"]))
    g = cfg["grpo"]

    # policy (trainable LoRA). The KL reference reuses the policy base with the
    # LoRA adapter disabled (reference=None) -> one 7B model, not two. Set a
    # separate frozen thinker here only if you have spare GPU memory.
    policy = build_thinker(model_cfg, cfg.get("peft"))
    reference = None

    selector = TrailerSelector(
        policy,
        min_clip_seconds=model_cfg["selector"].get("min_clip_seconds", 5),
        max_clip_seconds=model_cfg["selector"].get("max_clip_seconds", 60),
    )
    reward_model = RewardModel(RewardConfig(
        provider=reward_cfg["judge"]["provider"],
        model=reward_cfg["judge"]["model"],
        api_key_env=reward_cfg["judge"]["api_key_env"],
        temperature=reward_cfg["judge"].get("temperature", 0.0),
        max_retries=reward_cfg["judge"].get("max_retries", 4),
        weights=reward_cfg.get("axes", {}),
        cache_dir=(reward_cfg.get("cache", {}) or {}).get("dir"),
        length_penalty=(reward_cfg.get("shaping", {}) or {}).get("length_penalty", 0.05),
        target_clip_seconds=model_cfg["selector"].get("target_clip_seconds", 45),
    ))

    workdir = Path("data/processed/rollouts"); workdir.mkdir(parents=True, exist_ok=True)
    sampling = g.get("sampling", {})

    gen_kwargs = {
        "temperature": sampling.get("temperature", 1.0),
        "top_p": sampling.get("top_p", 0.95),
        "max_new_tokens": sampling.get("max_new_tokens", 64),
    }

    def rollout_fn(sample):
        fin = FusionInputs(video_path=sample["video"])
        inputs = policy.fusion.build_inputs(fin, _SELECTOR_PROMPT)
        group = rollout_group(
            fin, inputs, selector, reward_model,
            video_summary=sample.get("summary", ""),
            group_size=g["group_size"], workdir=str(workdir),
            gen_kwargs=gen_kwargs,
        )
        return inputs, group

    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=optim_cfg["lr"], weight_decay=optim_cfg.get("weight_decay", 0.0),
    )
    logger = MetricLogger(
        backend=cfg.get("logging", {}).get("backend", "none"),
        project=cfg.get("logging", {}).get("project", "omni-trailer"),
    )

    trainer = GRPOTrainer(policy, reference, optimizer, GRPOConfig(
        group_size=g["group_size"], kl_coeff=g.get("kl_coeff", 0.04),
        clip_ratio=g.get("clip_ratio", 0.2),
        normalize_advantages=g.get("normalize_advantages", True),
        lr=optim_cfg["lr"], max_steps=optim_cfg.get("max_steps", 5000),
        grad_accum_steps=optim_cfg.get("grad_accum_steps", 8),
        max_grad_norm=optim_cfg.get("max_grad_norm", 1.0),
    ), logger=logger)

    log_cfg = cfg.get("logging", {})
    samples = read_manifest(cfg["data"]["train_manifest"])
    trainer.train(
        samples, rollout_fn,
        save_dir=log_cfg.get("save_dir", "checkpoints/"),
        save_every=log_cfg.get("save_every_steps", 0),
    )


if __name__ == "__main__":
    main()
