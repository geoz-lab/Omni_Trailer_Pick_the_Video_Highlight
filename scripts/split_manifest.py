"""Split a manifest (JSONL of {"video", "summary"}) into train / test sets.

The test split is held out from GRPO training and used for evaluation (to measure
the policy's lift on unseen clips). Deterministic given --seed.

Usage:
    # 10% held out for test (default), shuffled with seed 42
    python scripts/split_manifest.py --input data/metadata/train.jsonl

    # or a fixed test size
    python scripts/split_manifest.py --input data/metadata/train.jsonl --test-size 100
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/metadata/train.jsonl", help="full manifest to split")
    p.add_argument("--train-out", default="data/metadata/train.jsonl")
    p.add_argument("--test-out", default="data/metadata/test.jsonl")
    p.add_argument("--test-frac", type=float, default=0.1, help="fraction held out for test")
    p.add_argument("--test-size", type=int, default=None, help="absolute test size (overrides --test-frac)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lines = [ln for ln in Path(args.input).read_text().splitlines() if ln.strip()]
    if not lines:
        print(f"No entries in {args.input}")
        return

    random.seed(args.seed)
    random.shuffle(lines)

    n_test = args.test_size if args.test_size is not None else max(1, round(len(lines) * args.test_frac))
    n_test = min(n_test, len(lines) - 1)          # keep at least 1 for train
    test, train = lines[:n_test], lines[n_test:]

    # write test first in case train-out == input (we already read it fully)
    Path(args.test_out).write_text("\n".join(test) + "\n")
    Path(args.train_out).write_text("\n".join(train) + "\n")
    print(f"{len(lines)} total -> {len(train)} train ({args.train_out}), "
          f"{len(test)} test ({args.test_out})  [seed={args.seed}]")


if __name__ == "__main__":
    main()
