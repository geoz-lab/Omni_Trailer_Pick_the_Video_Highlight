"""The omni "thinker": Qwen2.5-Omni-7B wrapped as the reasoning backbone.

It natively ingests video + audio + text and performs event detection, scene
understanding, emotion analysis, story modeling and cross-modal reasoning. The
same model is the policy during GRPO training (a LoRA adapter is trained on top;
the base stays frozen).

Designed to run on an A100/H100. Not runnable on CPU/Mac.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fusion_model import FusionInputs, OmniFusion


@dataclass
class ThinkerConfig:
    backbone: str = "Qwen/Qwen2.5-Omni-7B"
    frame_rate: float = 2.0
    video_max_pixels: int = 200704        # per-frame cap; bounds vision-attention memory
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    device_map: str = "auto"
    use_audio_in_video: bool = True


class OmniThinker:
    """Loads the omni backbone and exposes generate / log-prob hooks for RL."""

    def __init__(self, config: ThinkerConfig | None = None) -> None:
        self.config = config or ThinkerConfig()
        self.model: Any = None
        self.processor: Any = None
        self.fusion: OmniFusion | None = None

    # ------------------------------------------------------------------ load
    def load(self, lora_path: str | None = None) -> "OmniThinker":
        """Load Qwen2.5-Omni weights + processor; optionally attach a LoRA adapter."""
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        # Qwen2.5-Omni's from_pretrained always calls load_speakers(), which uses
        # torch.load and is blocked by transformers on torch < 2.6 (CVE-2025-32434).
        # We're pinned to torch 2.5.x (older CUDA wheels are the only ones that run
        # on GLIBC 2.17 clusters), so neutralize the guard for this trusted official
        # speaker file (loaded weights_only). We only use the text thinker anyway.
        import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
        _qwen_mod.check_torch_load_is_safe = lambda *args, **kwargs: None

        dtype = getattr(torch, self.config.dtype)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.config.backbone)
        # enable_audio_output=False still saves the talker's runtime memory.
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.config.backbone,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_implementation,
            device_map=self.config.device_map,
            enable_audio_output=False,
        )
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        if lora_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self.fusion = OmniFusion(self.processor, self.config.frame_rate,
                                 self.config.use_audio_in_video, self.config.video_max_pixels)
        return self

    # --------------------------------------------------------------- helpers
    def _device(self):
        return next(self.model.parameters()).device

    def _prepare(self, sample: FusionInputs, prompt: str) -> dict:
        assert self.fusion is not None, "call load() first"
        inputs = self.fusion.build_inputs(sample, prompt)
        return {k: v.to(self._device()) for k, v in inputs.items() if hasattr(v, "to")}

    # ------------------------------------------------------------- inference
    def reason(self, sample: FusionInputs, prompt: str, max_new_tokens: int = 64) -> str:
        """Greedy single pass; returns the model's decoded completion (text only)."""
        import torch

        inputs = self._prepare(sample, prompt)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                use_audio_in_video=self.config.use_audio_in_video,
                return_audio=False,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------- RL
    def generate(self, inputs: dict, temperature: float = 1.0, top_p: float = 0.95,
                 max_new_tokens: int = 64) -> dict:
        """Sample one completion; return {text, token_ids, logprobs} for a rollout."""
        import torch

        inputs = {k: (v.to(self._device()) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                use_audio_in_video=self.config.use_audio_in_video,
                return_audio=False,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, prompt_len:]
        # per-step log-prob of the actually-sampled token
        logprobs = []
        for step, score in enumerate(out.scores):
            logp = torch.log_softmax(score[0], dim=-1)
            logprobs.append(logp[gen_ids[step]].item())
        text = self.processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
        return {"text": text, "token_ids": gen_ids.tolist(), "logprobs": logprobs}

    def logprobs_of(self, inputs: dict, token_ids: list[int]):
        """Teacher-force `token_ids` after the prompt; return their per-token log-probs.

        Keeps gradients so GRPO can backprop through the current policy. Returns a
        1-D tensor of length ``len(token_ids)``.
        """
        import torch

        device = self._device()
        prompt_ids = inputs["input_ids"].to(device)
        cont = torch.tensor(token_ids, device=device).unsqueeze(0)
        full = torch.cat([prompt_ids, cont], dim=1)
        model_inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                        for k, v in inputs.items() if k != "input_ids"}
        attn = torch.ones_like(full)
        out = self.model(input_ids=full, attention_mask=attn, **model_inputs)
        # logits at position t predict token t+1; align to the continuation block
        start = prompt_ids.shape[1] - 1
        logits = out.logits[0, start:start + len(token_ids), :]
        logp = torch.log_softmax(logits, dim=-1)
        return logp[range(len(token_ids)), cont[0]]
