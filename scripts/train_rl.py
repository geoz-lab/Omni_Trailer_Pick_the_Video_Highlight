"""Train the trailer selector with GRPO (runs on the GPU cluster).

Usage:
    python scripts/train_rl.py --config configs/train_rl.yaml

Manifest format (one JSON object per line), e.g. data/metadata/train.jsonl:
    {"video": "data/raw_videos/match1.mp4", "summary": "Champions League final ..."}
"""
from __future__ import annotations

import argparse
import json
import os
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


def build_thinker(model_cfg: dict, trainable_lora: dict | None, resume: str | None = None,
                  device_map="auto") -> OmniThinker:
    tc = model_cfg["thinker"]
    thinker = OmniThinker(ThinkerConfig(
        backbone=tc["backbone"], frame_rate=tc.get("frame_rate", 2.0),
        video_max_pixels=tc.get("video_max_pixels", 200704),
        dtype=tc.get("dtype", "bfloat16"),
        attn_implementation=tc.get("attn_implementation", "flash_attention_2"),
        use_audio_in_video=tc.get("use_audio_in_video", True),
        device_map=device_map,
    )).load()
    if resume:
        # continue training from a saved adapter (chunked training)
        from peft import PeftModel

        thinker.model = PeftModel.from_pretrained(thinker.model, resume, is_trainable=True)
        print(f"[resume] loaded trainable adapter from {resume}")
        thinker.model.print_trainable_parameters()
    elif trainable_lora and trainable_lora.get("use_lora", True):
        from peft import LoraConfig, get_peft_model

        thinker.model = get_peft_model(thinker.model, LoraConfig(
            r=trainable_lora.get("lora_r", 16),
            lora_alpha=trainable_lora.get("lora_alpha", 32),
            lora_dropout=trainable_lora.get("lora_dropout", 0.05),
            target_modules=trainable_lora.get("target_modules"),
            task_type="CAUSAL_LM",
        ))
        thinker.model.print_trainable_parameters()

    # gradient checkpointing: recompute activations during backward -> big training
    # memory cut (needed for 90s video + audio on one GPU). enable_input_require_grads
    # lets grads flow through the frozen base to the LoRA params.
    m = thinker.model
    try:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
        print("[train] gradient checkpointing enabled")
    except Exception as exc:  # noqa: BLE001
        print(f"[train] gradient checkpointing not enabled: {exc}")
    return thinker


def setup_distributed():
    """Return (rank, world_size, local_rank). >1 world_size when launched via torchrun."""
    import torch

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        import torch.distributed as dist

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), world_size, local_rank
    return 0, 1, 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_rl.yaml")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override optim.max_steps (per rank; e.g. --max-steps 2 for a sanity run)")
    ap.add_argument("--resume", default=None,
                    help="continue from a saved adapter dir (e.g. checkpoints/adapter_final) for chunked training")
    args = ap.parse_args()

    import torch

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps is not None:
        cfg.setdefault("optim", {})["max_steps"] = args.max_steps
    assert cfg["algorithm"] == "grpo", "default recipe is GRPO; see ppo_trainer for PPO"
    model_cfg = yaml.safe_load(open(cfg["model_config"]))
    reward_cfg = yaml.safe_load(open(cfg["reward_config"]))
    g = cfg["grpo"]

    # policy (trainable LoRA). The KL reference reuses the policy base with the
    # LoRA adapter disabled (reference=None) -> one 7B model, not two.
    # Under DDP each rank holds one full replica on its own GPU ({"": local_rank});
    # single-GPU uses device_map="auto".
    device_map = {"": local_rank} if world_size > 1 else "auto"
    policy = build_thinker(model_cfg, cfg.get("peft"), resume=args.resume, device_map=device_map)
    reference = None

    # make every rank start from identical LoRA weights (random A init differs per rank)
    if world_size > 1:
        import torch.distributed as dist
        for p in policy.model.parameters():
            if p.requires_grad:
                dist.broadcast(p.data, src=0)

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
        target_clip_seconds=(reward_cfg.get("shaping", {}) or {}).get(
            "target_clip_seconds", model_cfg["selector"].get("target_clip_seconds", 45)),
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
    log_cfg = cfg.get("logging", {})
    logger = MetricLogger(                       # only rank 0 logs / writes metrics
        backend=log_cfg.get("backend", "none"),
        project=log_cfg.get("project", "omni-trailer"),
        metrics_file=str(Path(log_cfg.get("save_dir", "checkpoints/")) / "metrics.jsonl"),
    ) if is_main else None

    trainer = GRPOTrainer(policy, reference, optimizer, GRPOConfig(
        group_size=g["group_size"], kl_coeff=g.get("kl_coeff", 0.04),
        clip_ratio=g.get("clip_ratio", 0.2),
        normalize_advantages=g.get("normalize_advantages", True),
        lr=optim_cfg["lr"], max_steps=optim_cfg.get("max_steps", 5000),
        grad_accum_steps=optim_cfg.get("grad_accum_steps", 8),
        max_grad_norm=optim_cfg.get("max_grad_norm", 1.0),
    ), logger=logger, is_main=is_main, world_size=world_size)

    # shard the manifest across ranks (each rank trains on different videos)
    samples = list(read_manifest(cfg["data"]["train_manifest"]))
    if world_size > 1:
        samples = samples[rank::world_size]
    if is_main:
        print(f"world_size={world_size}  shard={len(samples)} videos/rank  "
              f"max_steps={optim_cfg.get('max_steps')} (per rank)")
    trainer.train(
        samples, rollout_fn,
        save_dir=log_cfg.get("save_dir", "checkpoints/"),
        save_every=log_cfg.get("save_every_steps", 0),
    )

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
