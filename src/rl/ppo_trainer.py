"""PPO trainer (optional alternative to GRPO).

Kept as a fallback for when you want an explicit value head and dense/step-wise
rewards. Heavier than GRPO (extra critic network + GAE) but more standard.
GRPO is the project default — see `grpo_trainer.py`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    kl_coeff: float = 0.04
    lr: float = 1e-6
    ppo_epochs: int = 4


class PPOTrainer:
    """Actor-critic PPO over highlight-selection completions."""

    def __init__(self, policy, value_head, reference, optimizer, config: PPOConfig | None = None) -> None:
        self.policy = policy
        self.value_head = value_head
        self.reference = reference
        self.optimizer = optimizer
        self.config = config or PPOConfig()

    def compute_gae(self, rewards, values):
        """Generalized Advantage Estimation."""
        raise NotImplementedError

    def step(self, batch) -> dict:
        """Clipped policy loss + value loss + KL penalty over `ppo_epochs`."""
        raise NotImplementedError
