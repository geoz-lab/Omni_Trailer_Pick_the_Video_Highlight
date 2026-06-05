"""Train the trailer selector with GRPO.

Usage:
    python scripts/train_rl.py --config configs/train_rl.yaml
"""
from __future__ import annotations

import argparse

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_rl.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    assert cfg["algorithm"] == "grpo", "default recipe is GRPO; see ppo_trainer for PPO"
    # TODO:
    #   1. load OmniThinker (policy w/ LoRA) + frozen reference
    #   2. build TrailerSelector + RewardModel (API judge)
    #   3. build dataloader from cfg['data']['train_manifest']
    #   4. GRPOTrainer(...).train(dataloader, rollout_fn=rollout_group)
    raise NotImplementedError("Wire policy + reward + GRPOTrainer here.")


if __name__ == "__main__":
    main()
