# Datasets

Task-aligned video sources for training/evaluating the highlight selector. Pexels
stock footage was dropped — much of it is silent and off-task. These sources have
**real audio** (commentary/crowd) and genuine highlight structure.

## Roles

| Source | Role | Audio |
| --- | --- | --- |
| **SoccerNet** | main training data (soccer broadcasts) | commentary + crowd |
| **Mr. HiSum** | large-scale weak supervision / pretraining (most-replayed labels) | YouTube audio |
| **TVSum + YouTube Highlights** | benchmark with *human* highlight labels | YouTube audio |
| **Internet Archive / Wikimedia / Pixabay / Mixkit** | shareable demo clips | mixed |

## ⚠️ Licensing & storage (read first)
- **SoccerNet, Mr.HiSum, TVSum, YouTube Highlights are research-only, copyrighted
  videos.** Keep them on `$SCRATCH` or `$OAK` — **never commit them** (the repo
  `.gitignore` excludes `data/`). Only CC/public-domain demo clips may be committed.
- `yt-dlp`-based downloads (Mr.HiSum, TVSum, YouTube Highlights) are your
  responsibility under each platform's terms — use for research only.

## SoccerNet (main training)
1. Register and sign the NDA at <https://www.soccer-net.org/> to get the download
   password.
2. `pip install SoccerNet`
3. Download (Python):
   ```python
   from SoccerNet.Downloader import SoccerNetDownloader as SND
   d = SND(LocalDirectory="$SCRATCH/omni_data/soccernet")
   d.password = "<from the NDA>"
   d.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"])
   ```
   Each game is two ~45-min halves with audio. (224p keeps size manageable;
   720p available.)
4. Cut into clips and build a manifest:
   ```bash
   python scripts/build_manifest.py --videos-dir $SCRATCH/omni_data/soccernet \
       --out data/metadata/soccernet.jsonl --segment 90 --require-audio \
       --summary "Soccer match broadcast (SoccerNet)"
   ```
   (Optionally use SoccerNet's action-spotting labels to target segments around
   goals/cards instead of uniform 90s cuts — a future enhancement.)

## Mr. HiSum (weak supervision / pretraining) + YouTube soccer (interim)
- Repo: <https://github.com/MRHiSum/MR.HiSum> — provides **most-replayed** highlight
  scores + features for ~31k YouTube videos (frame-level "ground truth" highlights).
  It ships *metadata + labels*, **not the videos** — fetch those by YouTube ID.
- Soccer subset: filter the Mr.HiSum metadata to its football/soccer category
  (YouTube-8M labels), save the IDs to a file, then download:
  ```bash
  python scripts/download_youtube.py --ids soccer_ids.txt --out $SCRATCH/omni_data/yt_soccer
  ```
- **Quick interim (no metadata needed)** — while SoccerNet's NDA clears, grab
  soccer clips directly by search (we reward with Gemini, so we don't strictly
  need Mr.HiSum's labels to start training):
  ```bash
  pip install yt-dlp
  python scripts/download_youtube.py --search "soccer goals highlights" --limit 60 \
      --out $SCRATCH/omni_data/yt_soccer --max-duration 1200
  python scripts/build_manifest.py --videos-dir $SCRATCH/omni_data/yt_soccer \
      --out data/metadata/all.jsonl --segment 90 --require-audio \
      --summary "Soccer highlights (YouTube)"
  python scripts/split_manifest.py --input data/metadata/all.jsonl --test-frac 0.1
  ```
  `yt-dlp` muxes audio in, so these clips have commentary/crowd. Research use only;
  some IDs may be unavailable; keep videos on `$SCRATCH`, never commit.

## TVSum & YouTube Highlights (benchmark)
- **TVSum** (50 videos, 10 categories, importance scores): `ydata-tvsum50`
  package/site provides annotations + video IDs.
- **YouTube Highlights** (Sun et al. 2014; 6 domains: surfing, skating, …):
  per-domain train/test lists + highlight labels.
- Fetch videos via `yt-dlp` into `$SCRATCH/omni_data/{tvsum,yt_highlights}/`,
  `build_manifest.py --max-seconds 90 --require-audio`.
- These have **human** labels → use them as a held-out benchmark (compare the
  policy's pick to the annotated highlight, not just the Gemini reward).

## Open / demo clips (shareable)
- **Pixabay** (free API, like Pexels), **Mixkit** (direct downloads), **Internet
  Archive** (public domain), **Wikimedia Commons** (CC). Use for README demo GIFs
  that *can* be committed. `scripts/collect_videos.py --source samples` already
  pulls a few CC clips; a Pixabay path can be added on request.

## Ingest → train (common flow)
```bash
# 1. put videos on scratch:  $SCRATCH/omni_data/<dataset>/
# 2. build a manifest (segment long videos / require audio as needed)
python scripts/build_manifest.py --videos-dir $SCRATCH/omni_data/<dataset> \
    --out data/metadata/all.jsonl --segment 90 --require-audio --summary "..."
# 3. split into train/test
python scripts/split_manifest.py --input data/metadata/all.jsonl --test-frac 0.1
# 4. point configs/train_rl.yaml data.train_manifest at it (already data/metadata/train.jsonl)
```

> The Gemini reward judge still works the same way regardless of source. With
> SoccerNet's real crowd/commentary audio, the audio-alignment axis becomes
> meaningful (it was mostly inert on silent Pexels clips).
