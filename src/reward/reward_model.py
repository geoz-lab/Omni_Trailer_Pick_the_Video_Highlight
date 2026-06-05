"""Reward model: an external VLM (Gemini / GPT-4o) that scores candidate clips.

The judge is kept strictly larger than and frozen relative to the policy, so the
policy cannot trivially reward-hack it. Scores are cached by clip hash to avoid
paying for the same clip twice across GRPO rollouts.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import reward_prompts


@dataclass
class RewardConfig:
    provider: str = "gemini"               # gemini | openai
    model: str = "gemini-2.5-pro"
    api_key_env: str = "GEMINI_API_KEY"
    frame_rate: float = 1.0
    temperature: float = 0.0
    max_retries: int = 4
    weights: dict[str, float] = field(default_factory=dict)
    cache_dir: str | None = "data/processed/reward_cache"
    length_penalty: float = 0.05
    target_clip_seconds: float = 45.0


class RewardModel:
    """Scores a candidate clip and returns a scalar reward + per-axis breakdown."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._client = None
        if self.config.cache_dir:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

    # --- client / API ----------------------------------------------------
    def _ensure_client(self):
        if self._client is not None:
            return
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(f"Set ${self.config.api_key_env} for the judge API.")
        # TODO: construct the Gemini / OpenAI client from `key`.
        raise NotImplementedError("Initialize the judge API client here.")

    def _call_judge(self, clip_path: str, video_summary: str) -> dict[str, float]:
        """Send the clip + prompt to the judge and parse per-axis scores."""
        self._ensure_client()
        # user_prompt = reward_prompts.build_user_prompt(video_summary)
        # TODO: upload sampled frames + audio of `clip_path`; call the model with
        #       reward_prompts.SYSTEM_PROMPT; retry up to max_retries.
        # return reward_prompts.parse_scores(raw_response)
        raise NotImplementedError

    # --- caching ---------------------------------------------------------
    def _cache_path(self, clip_path: str) -> Path | None:
        if not self.config.cache_dir:
            return None
        digest = hashlib.sha1(Path(clip_path).read_bytes()).hexdigest()
        return Path(self.config.cache_dir) / f"{digest}.json"

    # --- public API ------------------------------------------------------
    def score(self, clip_path: str, video_summary: str, duration_s: float) -> dict:
        """Return {reward: float, axes: {axis: score}} for one candidate clip."""
        cache_path = self._cache_path(clip_path)
        if cache_path and cache_path.exists():
            axes = json.loads(cache_path.read_text())
        else:
            axes = self._call_judge(clip_path, video_summary)
            if cache_path:
                cache_path.write_text(json.dumps(axes))

        weights = self.config.weights or {a: 1 / len(axes) for a in axes}
        reward = sum(weights.get(a, 0.0) * s for a, s in axes.items())
        reward -= self.config.length_penalty * abs(
            duration_s - self.config.target_clip_seconds
        ) / max(self.config.target_clip_seconds, 1.0)
        return {"reward": reward, "axes": axes}
