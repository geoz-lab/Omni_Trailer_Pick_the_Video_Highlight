"""GRPO trainer (primary RL algorithm).

GRPO needs no value network: for each video it samples a *group* of candidate
highlights, then sets each candidate's advantage to its reward minus the group
mean (optionally / std). The policy is nudged toward above-average candidates via
a PPO-style clipped surrogate, with a KL leash to a frozen reference policy.
Only a LoRA adapter on the omni thinker is trained.
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
        std = (sum(a * a for a in advs) / n) ** 0.5
        if std > 1e-8:
            advs = [a / std for a in advs]
    return advs


class GRPOTrainer:
    """Optimizes the omni policy (LoRA) from groups of scored candidates."""

    def __init__(self, policy, reference, optimizer, config: GRPOConfig | None = None,
                 logger=None) -> None:
        self.policy = policy            # OmniThinker (trainable LoRA)
        # reference for the KL term. If None, we reuse the policy's *base* model
        # with the LoRA adapter disabled — one model instead of two (~half the
        # GPU memory). Pass a separate frozen OmniThinker only if you want it.
        self.reference = reference
        self.optimizer = optimizer
        self.config = config or GRPOConfig()
        self.logger = logger

    def _reference_logprobs(self, inputs: dict, token_ids: list[int]):
        """Frozen-policy log-probs for the KL term (no grad)."""
        import torch

        with torch.no_grad():
            if self.reference is not None:
                return self.reference.logprobs_of(inputs, token_ids)
            # reuse the policy base with the adapter turned off
            if hasattr(self.policy.model, "disable_adapter"):
                with self.policy.model.disable_adapter():
                    return self.policy.logprobs_of(inputs, token_ids)
            # no adapter at all -> reference == policy, KL contributes ~0
            return self.policy.logprobs_of(inputs, token_ids)

    def step(self, inputs: dict, group: list[Candidate]) -> dict:
        """Accumulate the GRPO loss for one group (calls backward). Returns metrics."""
        import torch

        cfg = self.config
        advs = group_advantages([c.reward for c in group], cfg.normalize_advantages)

        total_loss = 0.0
        total_kl = 0.0
        for cand, adv in zip(group, advs):
            if not cand.token_ids:
                continue
            new_logp = self.policy.logprobs_of(inputs, cand.token_ids)          # [T], grad
            ref_logp = self._reference_logprobs(inputs, cand.token_ids).to(new_logp.device)
            old_logp = torch.as_tensor(cand.logprobs, dtype=new_logp.dtype, device=new_logp.device)

            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
            pg_loss = -torch.min(unclipped, clipped)
            # k3 unbiased KL estimator (Schulman): exp(d) - d - 1, d = ref - new
            d = ref_logp - new_logp
            kl = torch.exp(d) - d - 1
            loss = (pg_loss + cfg.kl_coeff * kl).mean()
            (loss / cfg.group_size).backward()
            total_loss += loss.item()
            total_kl += kl.mean().item()

        g = len(group)
        return {
            "loss": total_loss / g,
            "kl": total_kl / g,
            "reward_mean": sum(c.reward for c in group) / g,
            "reward_max": max(c.reward for c in group),
            "malformed_frac": sum(c.span is None for c in group) / g,
        }

    def _save(self, save_dir: str, tag) -> None:
        """Save the trained LoRA adapter (reload via OmniThinker.load(lora_path=...))."""
        from pathlib import Path

        out = Path(save_dir) / f"adapter_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        self.policy.model.save_pretrained(str(out))
        if self.logger is not None:
            self.logger.log.info("saved adapter -> %s", out)

    def train(self, samples, rollout_fn, save_dir: str | None = None, save_every: int = 0) -> None:
        """Main loop. ``rollout_fn(sample) -> (inputs, group)`` produces a GRPO group.

        Periodically (and at the end) saves the policy's LoRA adapter to ``save_dir``.
        """
        import torch

        self.optimizer.zero_grad()
        last_step = -1
        for step, sample in enumerate(samples):
            if step >= self.config.max_steps:
                break
            last_step = step
            inputs, group = rollout_fn(sample)
            metrics = self.step(inputs, group)

            if (step + 1) % self.config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in self.policy.model.parameters() if p.requires_grad),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.logger is not None:
                self.logger.log_metrics(metrics, step=step)

            if save_dir and save_every and (step + 1) % save_every == 0:
                self._save(save_dir, step + 1)

        if save_dir and last_step >= 0:
            self._save(save_dir, "final")
