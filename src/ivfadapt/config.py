"""Load configs/*.yaml and resolve paths relative to the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/sift1m.yaml") -> dict[str, Any]:
    """Read a YAML config. Relative paths inside it are resolved against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = str(REPO_ROOT / cfg["data_dir"])
    cfg.setdefault("results_dir", str(REPO_ROOT / "results"))
    return cfg
