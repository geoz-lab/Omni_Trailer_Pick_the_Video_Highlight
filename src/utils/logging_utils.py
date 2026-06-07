"""Lightweight logging: console + a metrics JSONL file + optional Weights & Biases."""
from __future__ import annotations

import json
import logging
from pathlib import Path


def get_logger(name: str = "omni_trailer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False    # avoid duplicate lines via the root logger
    return logger


class MetricLogger:
    """Thin wrapper that logs to console and optionally to W&B."""

    def __init__(self, backend: str = "none", project: str = "omni-trailer",
                 run_name: str | None = None, metrics_file: str | None = None) -> None:
        self.backend = backend
        self.log = get_logger()
        self._wandb = None
        # always append metrics here (one JSON object per line) for later plotting
        self.metrics_file = Path(metrics_file) if metrics_file else None
        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        if backend == "wandb":
            import wandb  # imported lazily so the dep is optional
            self._wandb = wandb.init(project=project, name=run_name)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.log.info("step=%s %s", step, metrics)
        if self.metrics_file:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
