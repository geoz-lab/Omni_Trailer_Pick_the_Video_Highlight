# Idea

## Problem

Manually finding the single most compelling moment in a long video — the clip
that would make a great trailer — is slow and subjective. Supervised highlight
detection needs large labeled datasets of "the highlight," which barely exist
and don't generalize across genres.

## Insight

We don't need ground-truth highlight labels. A strong **VLM judge** can already
tell whether a *candidate* clip is exciting, emotionally moving, story-complete,
relevant, and well-aligned across audio and video. If the judge can *score*
clips, we can **search** for the best clip with reinforcement learning — the
judge is the reward.

## Approach

1. A **unified omni model** (Qwen2.5-Omni) ingests video + audio + captions and
   reasons about events, emotion and story.
2. A **selector policy** emits highlight start/end timestamps as text tokens.
3. A frozen **VLM judge** (Gemini / GPT-4o) scores each candidate clip on six
   axes; the weighted score is the reward.
4. **GRPO** trains the policy from groups of candidates — no value network, no
   human labels.

## Why this can work

- The judge generalizes across genres (sports, film, vlogs) far better than a
  narrow supervised detector.
- Emitting timestamps as tokens keeps the policy a plain LM, so RL is simple.
- Group-relative advantages (GRPO) are stable with a single scalar reward.

## Open questions

- Reward hacking: does the policy learn to exploit judge biases? (Mitigations:
  frozen + larger judge, axis ensembling, periodic judge rotation.)
- Cost: API judging inside the RL loop is the main expense — caching + proposal
  pre-filtering keep calls down.
- Long videos: proposal pre-filtering vs. native long-context omni input.
