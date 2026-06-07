#!/bin/bash
# Data-parallel GRPO on Sherlock: N GPUs each train on a different shard of videos,
# gradients averaged across ranks (manual all-reduce). ~Nx throughput.
#   mkdir -p logs && sbatch slurm/train_ddp.sbatch
#   resume/limit:  sbatch --export=ALL,MAX_STEPS=150,RESUME=checkpoints/adapter_150 slurm/train_ddp.sbatch
#
# Set NGPU to the number of GPUs (each holds one full 7B replica -> needs ~one 40-80GB
# card per GPU). Keep it on ONE node (this launcher is single-node, multi-GPU).
#SBATCH -J omni-grpo-ddp
#SBATCH -p serc
#SBATCH -G 4                     # <-- number of GPUs (match NGPU below)
#SBATCH -C GPU_MEM:80GB
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o logs/train_ddp_%j.out
#SBATCH -e logs/train_ddp_%j.err

set -euo pipefail
NGPU=4                           # must match #SBATCH -G

module load cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omni_trailer

export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# GEMINI_API_KEY auto-loads from .env (each rank calls the judge for its own clips)

nvidia-smi
# use the env's python explicitly ($CONDA_PREFIX is reliable even if PATH isn't)
"$CONDA_PREFIX/bin/python" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
    scripts/train_rl.py --config configs/train_rl.yaml \
    ${RESUME:+--resume "$RESUME"} ${MAX_STEPS:+--max-steps "$MAX_STEPS"}
