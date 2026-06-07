# Training

## Recipe: GRPO

GRPO trains the selector policy without a value network. For each video:

1. **Rollout a group.** Sample `group_size` (default 4) highlight completions
   from the policy for the same video (`rl/rollout.py`).
2. **Cut + judge.** Export each candidate clip and score it with the VLM judge
   on six axes → weighted scalar reward (`reward/reward_model.py`).
3. **Group-relative advantage.** `adv_i = reward_i − mean(group)`, optionally
   `/ std(group)` (`rl/grpo_trainer.py::group_advantages`).
4. **Update.** Clipped surrogate on the policy/old-policy log-prob ratio, plus a
   KL penalty toward a frozen reference policy. Only a **LoRA** adapter on the
   omni thinker is trained; the base is frozen.

```bash
python scripts/train_rl.py --config configs/train_rl.yaml
```

## Key hyperparameters (`configs/train_rl.yaml`)

| Param | Default | Notes |
| ----- | ------- | ----- |
| `grpo.group_size` | 4 | candidates per video; larger = lower-variance advantages, more judge cost |
| `grpo.kl_coeff` | 0.04 | raise if the policy drifts / degenerates |
| `optim.lr` | 1e-6 | small LR over a LoRA adapter |
| `grpo.sampling.temperature` | 1.0 | needs diversity within a group |
| `peft.lora_r` | 16 | adapter capacity |

## Cost control

API judging is the dominant cost. Mitigations:

- **Cache** judge scores by clip hash (`reward.cache.enabled`).
- **Proposal pre-filter** so only promising windows reach the judge.
- Consider an **open 7B judge** for early iteration, switching to the API judge
  for final runs (swap `configs/reward.yaml::judge`).

## Avoiding reward hacking

- Keep the judge **frozen and larger** than the policy.
- **Length / duplicate penalties** (`reward.yaml::shaping`) stop degenerate clips.
- Watch per-axis scores during training — a spike on one axis with a drop in
  others usually signals exploitation.
- Hold out a **validation set** and eval every `eval.every_steps`.

## Warm start (optional)

Bootstrap with **DPO** on preference pairs (clip A ≻ clip B from the judge)
before online GRPO to speed convergence and reduce early API spend.

## Monitoring

`utils/logging_utils.py` logs to console and optionally W&B
(`logging.backend: wandb`). Track: mean reward, per-axis means, KL to reference,
fraction of malformed completions, mean clip duration.

## Multi-GPU (data-parallel GRPO)

`train_rl.py` supports data parallelism via `torchrun` — each GPU holds one full
7B replica, trains on a **different shard** of the manifest, and gradients are
**averaged across ranks** (manual all-reduce; not `nn.DistributedDataParallel`,
which doesn't play well with `generate()` + multiple backwards). This gives ~Nx
throughput for the rollout-bound loop.

```bash
# N GPUs on one node (each needs ~one 40-80GB card for a full replica)
torchrun --standalone --nproc_per_node=4 scripts/train_rl.py --config configs/train_rl.yaml
# or via Slurm:
sbatch slurm/train_ddp.sbatch        # set #SBATCH -G N and NGPU=N to match
```

Notes:
- `--max-steps` is **per rank**; total videos seen ≈ `max_steps × N`. The loop runs
  a fixed step count (cycling the shard) so all ranks stay in lockstep for the
  all-reduce.
- Rank 0 alone logs (`checkpoints/metrics.jsonl`) and saves adapters.
- LoRA weights are broadcast from rank 0 at startup so all ranks begin identical.
- Each rank calls the Gemini judge for its own clips, so API traffic scales with N
  (the retry/backoff + skip-on-failure keep it robust).
