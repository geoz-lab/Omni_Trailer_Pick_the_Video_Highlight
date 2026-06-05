"""GRPO trainer (primary RL algorithm).

GRPO needs no value network: for each video it samples a *group* of candidate
highlights, then sets each candidate's advantage to its reward minus the group
mean (optionally divided by the group std). The policy is nudged toward
above-average candidates, with a KL leash to a frozen reference policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from .rollout import Candidate


@dataclass
class GRPOConfig:
    group_size: int = 4
    kl_coeff: float = 0.04
    clip_ratio: float = 0.2
    normalize_advantages: bool = True
    lr: float = 1e-6
    max_steps: int = 5000
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0


def group_advantages(rewards: list[float], normalize: bool) -> list[float]:
    """Group-relative advantages: r - mean(r) [/ std(r)]."""
    n = len(rewards)
    mean = sum(rewards) / n
    advs = [r - mean for r in rewards]
    if normalize and n > 1:
        var = sum(a * a for a in advs) / n
        std = var ** 0.5
        if std > 1e-8:
            advs = [a / std for a in advs]
    return advs


class GRPOTrainer:
    """Optimizes the omni policy (LoRA) from groups of scored candidates."""

    def __init__(self, policy, reference, optimizer, config: GRPOConfig | None = None) -> None:
        self.policy = policy            # OmniThinker (trainable LoRA)
        self.reference = reference      # frozen copy for the KL term
        self.optimizer = optimizer
        self.config = config or GRPOConfig()

    def step(self, inputs: dict, group: list[Candidate]) -> dict:
        """Single optimization step over one GRPO group. Returns metrics."""
        advs = group_advantages([c.reward for c in group], self.config.normalize_advantages)

        # TODO: for each candidate, recompute current-policy logprobs of its
        #       token_ids, form ratio = exp(logp_new - logp_old), apply the
        #       clipped surrogate with `adv`, add kl_coeff * KL(policy||reference),
        #       backprop with grad accumulation + clipping.
        _ = advs  # placeholder until the loss is wired
        raise NotImplementedError("Wire the clipped GRPO surrogate + KL here.")

    def train(self, dataloader, rollout_fn) -> None:
        """Main loop: rollout a group per video, then `step` on it."""
        # for step, sample in enumerate(dataloader): group = rollout_fn(sample); self.step(...)
        raise NotImplementedError
