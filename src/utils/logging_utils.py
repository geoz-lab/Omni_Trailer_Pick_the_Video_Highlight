"""Lightweight logging: rich console + optional Weights & Biases."""
from __future__ import annotations

import logging


def get_logger(name: str = "omni_trailer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class MetricLogger:
    """Thin wrapper that logs to console and optionally to W&B."""

    def __init__(self, backend: str = "none", project: str = "omni-trailer", run_name: str | None = None) -> None:
        self.backend = backend
        self.log = get_logger()
        self._wandb = None
        if backend == "wandb":
            import wandb  # imported lazily so the dep is optional
            self._wandb = wandb.init(project=project, name=run_name)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self.log.info("step=%s %s", step, metrics)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
