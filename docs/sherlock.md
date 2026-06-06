# Running on Sherlock (Stanford HPC)

Sherlock is a Slurm cluster with a few quirks that matter here:

- **Login nodes have no GPU; compute nodes have no direct internet.** You
  download weights/data on a login node and compute on a GPU node.
- **`$HOME` is ~15 GB.** Qwen2.5-Omni-7B is ~20 GB — keep the repo, conda env,
  HF cache and checkpoints on **`$SCRATCH`** (or `$GROUP_SCRATCH`/`$OAK`).
- Exact GPU constraint strings and the HTTP proxy host change over time — verify
  against the current [Sherlock docs](https://www.sherlock.stanford.edu/docs/).

---

## 0. Gotcha: the demo video is not in the repo

`demo_video/Ronaldo_goal_demo.mp4` is gitignored (only the GIF is committed), so
`git pull` won't bring it. Copy it over, or use your own clip:

```bash
scp demo_video/Ronaldo_goal_demo.mp4 \
  <sunet>@login.sherlock.stanford.edu:$SCRATCH/Omni_Trailer_Pick_the_Video_Highlight/demo_video/
```

## 1. One-time setup (login node — has internet)

> ⚠️ Do **not** use `conda env create -f environment.yml` here — Sherlock is
> CentOS 7 (GLIBC 2.17) and conda's `pytorch-cuda` needs GLIBC ≥ 2.27, while the
> system GCC 4.8.5 can't build native wheels. Use the verified recipe below
> (also in the README "Environment setup" section).

```bash
cd $SCRATCH
git clone https://github.com/geoz-lab/Omni_Trailer_Pick_the_Video_Highlight.git
cd Omni_Trailer_Pick_the_Video_Highlight

conda create -y -n omni_trailer python=3.11
conda activate omni_trailer

# native deps from conda-forge (GCC 4.8.5 can't build them; wandb pip-build needs Go)
conda install -y -c conda-forge av scipy librosa numba wandb

# torch from pip cu121 wheels (manylinux2014 = GLIBC 2.17; conda pytorch-cuda won't run)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# rest as wheels; transformers pinned 4.52.4 (>=4.53 needs torch>=2.7 -> GLIBC>=2.27)
pip install "transformers==4.52.4" accelerate peft qwen-omni-utils \
    google-genai openai imageio imageio-ffmpeg moviepy opencv-python-headless
pip install --force-reinstall --no-cache-dir pillow   # bundles libtiff; avoids conda mismatch
```

(`attn_implementation: sdpa` in `configs/model.yaml` means **no flash-attn build
needed**. Build it later only if you want the speedup.)

**Pre-download the model** (compute nodes can't reach HuggingFace):

```bash
export HF_HOME=$SCRATCH/hf
huggingface-cli download Qwen/Qwen2.5-Omni-7B
```

**API key** for the reward judge:

```bash
cp .env.example .env       # edit .env -> GEMINI_API_KEY=...
```

`.env` is gitignored and auto-loaded by the scripts. Keep it on `$SCRATCH` so
batch jobs pick it up without pasting the key into sbatch files.

## 2. Quick interactive test

```bash
sh_dev -p gpu -G 1 -C GPU_MEM:80GB -t 1:00:00   # verify constraint with `sinfo`/`sh_part`
module load cuda/12.1.1
conda activate omni_trailer
export HF_HOME=$SCRATCH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_inference.py        # -> output/Ronaldo_goal_demo_highlight.mp4 + .gif
```

## 3. Batch inference

```bash
mkdir -p logs
sbatch slurm/inference.sbatch                       # demo video
sbatch slurm/inference.sbatch $SCRATCH/my_clip.mp4  # your own video
```

## 4. Training (GRPO)

Prepare a manifest first — `data/metadata/train.jsonl`, one JSON per line:

```json
{"video": "data/raw_videos/match1.mp4", "summary": "Champions League final, last-minute winner"}
```

Then:

```bash
mkdir -p logs
sbatch slurm/train.sbatch
```

### The reward API needs outbound internet

Every GRPO step calls Gemini to score clips. If the GPU node has no egress,
training stalls on the first reward call. Options:

1. **Set the Sherlock HTTP(S) proxy** in `slurm/train.sbatch` (uncomment the
   `https_proxy`/`http_proxy` lines; get the host from Sherlock docs).
2. The per-clip **reward cache** (`data/processed/reward_cache`) dedupes repeat
   clips, but a fresh run still needs live access.
3. Or **host the judge on-cluster** (a local VLM) and edit `configs/reward.yaml`.

W&B logging also needs internet — start with `logging.backend: none` in
`configs/train_rl.yaml`, switch to `wandb` once egress works.

### Memory (GRPO)

`train_rl.py` trains **one** 7B model: the KL reference reuses the policy base
with the LoRA adapter disabled (no second copy), so it fits a single **80 GB**
H100 with the default token caps (`frame_rate: 1`, `video_max_pixels` in
`configs/model.yaml`).

If you still OOM (longer clips, bigger `group_size`):

- Lower `frame_rate` / `video_max_pixels` further.
- Reduce `grpo.group_size`.
- **Use two GPUs** — request `-G 2` (e.g. `-C GPU_SKU:H100_SXM5`); the model loads
  with `device_map="auto"`, which automatically shards the one model across both
  cards. No code change needed:

  ```bash
  srun --ntasks=1 -G 2 --constraint="GPU_SKU:H100_SXM5" \
       --cpus-per-task=16 --mem-per-cpu=8g --time=8:00:00 --partition=serc --pty bash
  # or in slurm/train.sbatch: #SBATCH -G 2
  ```

## 5. Common pitfalls

| Symptom | Cause / fix |
| --- | --- |
| `ImportError ... libm.so.6: version GLIBC_2.27 not found` (torch) | conda `pytorch-cuda` needs GLIBC ≥ 2.27; install pip cu121 wheels (`torch==2.5.1`, manylinux2014) instead (§1) |
| `module 'torch' has no attribute 'float8_e8m0fnu'` | transformers too new for torch ≤ 2.5; pin `transformers==4.52.4` |
| `ImportError: libtiff.so.5: cannot open shared object file` (Pillow) | conda Pillow vs libtiff mismatch; `pip install --force-reinstall --no-cache-dir pillow` |
| `scipy`/`av` build fails / "NumPy requires GCC >= 9.3" | system GCC 4.8.5 too old; install those from conda-forge, don't build from source |
| `cannot import name 'Qwen2_5OmniForConditionalGeneration'` | wrong env is active (env-stacking PATH shadow); `which python` → `conda deactivate; conda activate omni_trailer; hash -r` |
| `CUDA out of memory` in the vision encoder (huge alloc) | too many vision tokens; lower `frame_rate` and `video_max_pixels` in `configs/model.yaml` |
| `OSError ... can't reach huggingface.co` | Pre-download on login node; set `HF_HUB_OFFLINE=1` |
| Reward call hangs / times out | Compute node has no internet → set the proxy (§4) |
| `CUDA out of memory` in training | Use 80 GB GPU, or adapter-disable reference (§4) |
| `wandb` build fails: "Did not find the 'go' binary" | Install from conda-forge: `conda install -n omni_trailer -c conda-forge wandb` (pip builds wandb-core from source). It's optional anyway. |
| `$HOME` quota exceeded | Move repo/env/`HF_HOME`/checkpoints to `$SCRATCH` |
| Job killed at time limit | `gpu` partition caps ~2 days; checkpoint + resume |

## 6. Fill in the README trailer GIF

After a successful run, `output/Ronaldo_goal_demo_highlight.gif` exists on Sherlock.
Copy it back and commit so the README's right-hand demo cell renders:

```bash
scp <sunet>@login.sherlock.stanford.edu:$SCRATCH/.../output/Ronaldo_goal_demo_highlight.gif output/
git add -f output/Ronaldo_goal_demo_highlight.gif && git commit -m "Add trailer GIF" && git push
```
