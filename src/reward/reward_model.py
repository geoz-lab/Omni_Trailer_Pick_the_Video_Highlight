"""Reward model: an external VLM (Gemini / GPT-4o) that scores candidate clips.

Default judge is Gemini 2.5 Flash: it ingests the clip's *video and audio*
natively (crowd roar / commentary are part of the highlight signal), and is cheap
enough to call inside the GRPO loop. GPT-4o is a config-swappable fallback that
scores sampled frames + transcript instead. The judge is kept larger than and
frozen relative to the policy, and scores are cached by clip hash so the same
clip is never paid for twice.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import reward_prompts


@dataclass
class RewardConfig:
    provider: str = "gemini"               # gemini | openai
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    frame_rate: float = 1.0                # frames/sec sampled for the openai fallback
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

    # --- client ----------------------------------------------------------
    def _api_key(self) -> str:
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise RuntimeError(f"Set ${self.config.api_key_env} for the judge API.")
        return key

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.config.provider == "gemini":
            from google import genai

            self._client = genai.Client(api_key=self._api_key())
        elif self.config.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key())
        else:
            raise ValueError(f"Unknown provider {self.config.provider!r}")

    # --- provider calls --------------------------------------------------
    def _judge_gemini(self, clip_path: str, user_prompt: str) -> str:
        """Upload the clip (video+audio) and ask Gemini for the JSON scores."""
        from google.genai import types

        client = self._client
        myfile = client.files.upload(file=clip_path)
        # Files need to reach ACTIVE state before generation.
        while getattr(myfile.state, "name", myfile.state) == "PROCESSING":
            time.sleep(1)
            myfile = client.files.get(name=myfile.name)
        resp = client.models.generate_content(
            model=self.config.model,
            contents=[myfile, user_prompt],
            config=types.GenerateContentConfig(
                system_instruction=reward_prompts.SYSTEM_PROMPT,
                temperature=self.config.temperature,
            ),
        )
        return resp.text

    def _judge_openai(self, clip_path: str, user_prompt: str) -> str:
        """Fallback: send sampled frames + prompt to a GPT-4o-class model."""
        from ..utils.video_utils import sample_frames
        from PIL import Image
        import io

        frames = sample_frames(clip_path, fps=self.config.frame_rate)
        content = [{"type": "text", "text": user_prompt}]
        for fr in frames[:8]:                       # cap frames to control cost
            buf = io.BytesIO()
            Image.fromarray(fr).save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": reward_prompts.SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content

    def _call_judge(self, clip_path: str, video_summary: str) -> dict[str, float]:
        self._ensure_client()
        user_prompt = reward_prompts.build_user_prompt(video_summary)
        call = self._judge_gemini if self.config.provider == "gemini" else self._judge_openai
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                return reward_prompts.parse_scores(call(clip_path, user_prompt))
            except Exception as exc:                 # noqa: BLE001 - retry any API/parse error
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Judge failed after {self.config.max_retries} retries: {last_err}")

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
