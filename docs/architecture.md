# Architecture

## Chosen design: unified omni backbone

We use a single pretrained omni model as both the fusion module and the reasoning
"thinker", rather than training a custom fusion transformer over separate
encoders. Qwen2.5-Omni already aligns video, audio and text in pretraining, so we
inherit that alignment instead of relearning it.

```
                 ┌─────────────────────────────────────────────┐
 raw video ─────►│  proposal encoders (cheap pre-filter)        │
                 │   InternVideo2 (motion)  + Whisper/CLAP (audio)│
                 │   + BGE-m3 (text)  → top-k candidate windows  │
                 └───────────────────────┬─────────────────────┘
                                         │ (narrows the search)
                                         ▼
   video + audio + captions ──►  Qwen2.5-Omni THINKER (fusion + reasoning)
                                         │  event / scene / emotion / story /
                                         │  cross-modal reasoning
                                         ▼
                            TRAILER SELECTOR (policy head)
                            emits <start>MM:SS</start><end>MM:SS</end>
                                         │
                                         ▼
                     boundary snap → ffmpeg export → candidate clip
                                         │
                                         ▼
                            REWARD MODEL  (Gemini / GPT-4o judge)
                       6 axes → weighted scalar reward (cached)
                                         │
                                         ▼
                              GRPO update (LoRA on thinker)
```

## Components → code

| Stage | Module |
| ----- | ------ |
| Motion proposal | `src/encoders/visual_encoder.py` (InternVideo2) |
| Audio proposal | `src/encoders/audio_encoder.py` (Whisper + CLAP) |
| Text proposal | `src/encoders/text_encoder.py` (BGE-m3, optional) |
| Window proposal | `src/video_cut/segment_proposal.py` |
| Fusion adapter | `src/omni_model/fusion_model.py` |
| Thinker | `src/omni_model/llm_thinker.py` (Qwen2.5-Omni) |
| Policy / selector | `src/omni_model/trailer_selector.py` |
| Boundary snap | `src/video_cut/boundary_detection.py` |
| Export | `src/video_cut/video_exporter.py` |
| Judge | `src/reward/reward_model.py`, `reward_prompts.py` |
| RL | `src/rl/grpo_trainer.py` (+ `ppo_trainer.py`, `rollout.py`) |

## Why proposal encoders *and* a unified omni model

The omni thinker is expensive to run over a full long video at every RL step.
The proposal encoders are cheap and motion/audio-aware, so they cut the search
space to a handful of windows before the omni model reasons in detail. They are
inference-only (not trained by GRPO).

## Policy as text tokens

The selector outputs timestamps as literal tokens. This means the *same* model
that reasons also selects, the completion is a normal token sequence, and GRPO
can score it with one scalar — no regression/pointer head, no separate decoder.
