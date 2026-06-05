# Omni Trailer: Pick the Video Highlight

> Omni Model + RL for Automatic Video Highlight Generation

Omni Trailer is a multimodal reinforcement learning project for automatic
highlight and trailer generation.

Given a video with **visual frames**, **voice/audio**, and **captions/ASR text**,
the model learns to select the most engaging segment. Instead of relying on
human-labeled ground-truth highlights, the system uses a **reward model** to judge
generated candidate clips and trains the trailer selector through
**reinforcement learning**.

The reward model evaluates whether a candidate clip is emotionally engaging,
semantically important, visually clear, coherent as a short trailer, and aligned
with the full video context.

![Workflow](Omni_Trailer_workflow.png)

---

## Goal

Automatically pick the most touching, exciting, or representative highlight
section from a video, using audio, visual frames, and captions **together**.

## Pipeline

```
Video Input
    │
    ├── Visual Frames ──► Visual Encoder ──► visual embeddings
    ├── Voice / Audio ──► Audio Encoder  ──► audio embeddings
    └── Caption / ASR ──► Text Encoder   ──► text embeddings
                                  │
                                  ▼
                         Omni Fusion Model          (joint video + voice + caption)
                                  │
                                  ▼
                         Trailer Omni Model         (decides highlight start/end)
                                  │
                                  ▼
                       Candidate Highlight Clip
                                  │
                                  ▼
                            Reward Model            (large VLM/omni evaluator)
                                  │                  - emotional impact
                                  │                  - story completeness
                                  │                  - excitement
                                  │                  - relevance to full video
                                  │                  - audio-visual alignment
                                  │                  - trailer quality
                                  ▼
                   Reinforcement Learning Training  (PPO / GRPO / DPO-style)
                                  │
                                  ▼
                      Update Trailer Omni Model
                                  │
                                  ▼
                       Better Highlight Selection
```

### 1. Multimodal encoding
Each modality is encoded independently:
- **Visual Encoder** — frames → visual embeddings (e.g. CLIP/ViT/video backbone).
- **Audio Encoder** — waveform/spectrogram → audio embeddings (e.g. Whisper encoder / wav2vec).
- **Text Encoder** — captions or ASR transcript → text embeddings (tokenizer + LM).

### 2. Multimodal token fusion
The `OmniFusionModel` projects all modalities into a shared token space and
produces a **unified multimodal context** aligned along the video timeline.

### 3. Trailer Omni Model (LLM Thinker)
A reasoning module performs **event detection, scene understanding, emotion
analysis, story modeling, and cross-modal reasoning**, then a **Highlight Segment
Selector** acts as the policy that predicts the highlight **start/end
timestamps**.

### 4. Video cutting
`video_cut/` proposes candidate segments, refines boundaries, and exports the
final clip with ffmpeg/moviepy.

### 5. Reward model
A large VLM/omni model scores each candidate clip on six axes (excitement,
emotional impact, story completeness, relevance, AV alignment, trailer quality)
and returns a scalar reward.

### 6. RL training loop
The policy is optimized with **PPO / GRPO** (DPO-style preference training is
also supported) using the reward model's score as the training signal — no
ground-truth highlight labels required.

---

## Project structure

```
Omni_Trailer_Pick_the_Video_Highlight/
├── README.md
├── requirements.txt
├── configs/                # model / reward / RL hyperparameters (YAML)
├── data/                   # raw_videos, processed features, metadata
├── src/
│   ├── encoders/           # visual / audio / text encoders
│   ├── omni_model/         # fusion model, LLM thinker, trailer selector (policy)
│   ├── reward/             # reward model + scoring prompts
│   ├── rl/                 # PPO / GRPO trainers + rollout
│   ├── video_cut/          # segment proposal, boundary detection, export
│   └── utils/              # video / audio / logging helpers
├── scripts/                # preprocess, train, inference, evaluate
├── examples/               # demo video + outputs
└── docs/                   # idea, architecture, training notes
```

---

## Installation

```bash
git clone https://github.com/geoz-lab/Omni_Trailer_Pick_the_Video_Highlight.git
cd Omni_Trailer_Pick_the_Video_Highlight
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Preprocess raw videos into frames / audio / ASR features
python scripts/preprocess_videos.py --input data/raw_videos --output data/processed

# 2. Train the trailer selector with RL
python scripts/train_rl.py --config configs/train_rl.yaml

# 3. Run inference: pick the highlight from a new video
python scripts/run_inference.py --video examples/demo_video.mp4 --output examples/demo_output

# 4. Evaluate the reward model on candidate clips
python scripts/evaluate_reward.py --clips examples/demo_output
```

## Configuration

| File                    | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `configs/model.yaml`    | Encoder backbones, fusion + selector hyperparameters |
| `configs/reward.yaml`   | Reward model id, scoring weights, prompt settings    |
| `configs/train_rl.yaml` | RL algorithm (PPO/GRPO), rollout, optimization       |

## Documentation

- [`docs/idea.md`](docs/idea.md) — motivation and problem framing
- [`docs/architecture.md`](docs/architecture.md) — model and data flow
- [`docs/training.md`](docs/training.md) — RL training recipe

## Status

🚧 Early scaffold. Module interfaces are defined with documented stubs and
`TODO` markers; implementations are in progress.

## License

Released under the [MIT License](LICENSE).
