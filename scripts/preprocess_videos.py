"""Preprocess raw videos into frames, audio, ASR transcript and proposal features.

Writes per-video artifacts under data/processed/ and a manifest row into
data/metadata/ that downstream training/inference consume.

Usage:
    python scripts/preprocess_videos.py --input data/raw_videos --output data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/raw_videos", help="dir of raw videos")
    p.add_argument("--output", default="data/processed", help="dir for processed features")
    p.add_argument("--metadata", default="data/metadata", help="dir for manifests")
    p.add_argument("--config", default="configs/model.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.metadata).mkdir(parents=True, exist_ok=True)
    # TODO: for each video -> extract_waveform, transcribe (Whisper), CLAP events,
    #       InternVideo2 windows, write features + manifest row.
    raise NotImplementedError("Implement the preprocessing pipeline.")


if __name__ == "__main__":
    main()
