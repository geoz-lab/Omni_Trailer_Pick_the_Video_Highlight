"""The omni "thinker": Qwen2.5-Omni-7B wrapped as the reasoning backbone.

It natively ingests video + audio + text and performs event detection, scene
understanding, emotion analysis, story modeling and cross-modal reasoning. The
same model also acts as the policy network during GRPO training (a LoRA adapter
is trained on top; the base stays frozen).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fusion_model import FusionInputs, OmniFusion


@dataclass
class ThinkerConfig:
    backbone: str = "Qwen/Qwen2.5-Omni-7B"
    frame_rate: float = 2.0
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    device: str = "cuda"


class OmniThinker:
    """Loads the omni backbone and exposes generate / log-prob hooks for RL."""

    def __init__(self, config: ThinkerConfig | None = None) -> None:
        self.config = config or ThinkerConfig()
        self.model: Any = None
        self.processor: Any = None
        self.fusion: OmniFusion | None = None

    def load(self) -> None:
        """Load Qwen2.5-Omni weights + processor; init the fusion adapter."""
        # TODO: AutoModelForCausalLM / Qwen2_5OmniForConditionalGeneration + processor.
        self.fusion = OmniFusion(self.processor, self.config.frame_rate)
        raise NotImplementedError("Load Qwen2.5-Omni here.")

    def reason(self, sample: FusionInputs, prompt: str) -> str:
        """Run a single forward pass and return the model's textual reasoning."""
        assert self.fusion is not None, "call load() first"
        # inputs = self.fusion.build_inputs(sample, prompt)
        # TODO: self.model.generate(**inputs) -> decode.
        raise NotImplementedError

    def generate(self, inputs: dict, **gen_kwargs) -> dict:
        """Sample a completion. Returns {text, token_ids, logprobs} for RL rollouts."""
        # TODO: generate with output_scores=True; gather per-token logprobs.
        raise NotImplementedError

    def logprobs_of(self, inputs: dict, token_ids) -> Any:
        """Teacher-force `token_ids` and return their per-token log-probs.

        Used by GRPO to compute the policy/reference ratio.
        """
        raise NotImplementedError
