"""Tiny zero-dependency loader for a `.env` file.

Reads simple ``KEY=VALUE`` lines from a `.env` at the repo root and puts them in
``os.environ`` (without overwriting variables already set in the shell, so an
explicit ``export`` still wins). Avoids a python-dotenv dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env_file(path: str | os.PathLike | None = None, override: bool = False) -> None:
    """Load ``KEY=VALUE`` pairs from ``path`` (default: repo-root ``.env``)."""
    env_path = Path(path) if path else _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
