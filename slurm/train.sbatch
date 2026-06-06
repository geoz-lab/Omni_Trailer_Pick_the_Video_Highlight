#!/bin/bash
# Omni Trailer — GRPO training on Sherlock (Stanford HPC).
#   mkdir -p logs && sbatch slurm/train.sbatch
#
# Requirements before submitting:
#   - Model weights pre-downloaded to $HF_HOME on a login node (docs/sherlock.md).
#   - .env contains GEMINI_API_KEY (auto-loaded), or export it below.
#   - The reward judge calls the Gemini API every step -> the compute node MUST
#     have outbound HTTPS. If your partition has no egress, set the Sherlock
#     proxy (see PROXY note) or host the judge locally (configs/reward.yaml).
#   - data/metadata/train.jsonl exists (one {"video": ..., "summary": ...} per line).
#SBATCH -J omni-grpo
#SBATCH -p serc
#SBATCH -G 1                     # one 80GB H100 fits single-model GRPO; use -G 2 for headroom
#SBATCH -C GPU_MEM:80GB          # device_map=auto shards across both if you request -G 2
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00              # 50 steps is ~1-2h; raise if you increase max_steps
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err

set -euo pipefail

module load cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omni_trailer

export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Gemini key: prefer .env (auto-loaded by train_rl.py). Or uncomment:
# export GEMINI_API_KEY=...

# PROXY: if compute nodes lack direct internet, the Gemini API needs a proxy.
# Check the current Sherlock docs for the exact host, then uncomment:
# export https_proxy="http://<sherlock-proxy-host>:<port>"
# export http_proxy="$https_proxy"

nvidia-smi
python scripts/train_rl.py --config configs/train_rl.yaml
