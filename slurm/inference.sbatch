#!/bin/bash
# Omni Trailer — inference on Sherlock (Stanford HPC).
#   mkdir -p logs && sbatch slurm/inference.sbatch
# Override the video:  sbatch slurm/inference.sbatch /scratch/.../my_clip.mp4
#SBATCH -J omni-infer
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:80GB          # 40GB also works for 7B inference; verify with `sinfo`
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err

set -euo pipefail

VIDEO="${1:-demo_video/Ronaldo_goal_demo.mp4}"

module load cuda/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omni_trailer

# Weights were pre-downloaded on a login node (see docs/sherlock.md); run offline.
export HF_HOME="${HF_HOME:-$SCRATCH/hf}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

nvidia-smi
python scripts/run_inference.py --video "$VIDEO" --output output
