"""Shared fixtures. Tests need the cached index from experiments/01_build.py; else they skip."""

import numpy as np
import pytest

from ivfadapt.env import load_env
from ivfadapt.index import index_paths
from ivfadapt.config import load_config


@pytest.fixture(scope="session")
def env():
    cfg = load_config()
    if not all(p.exists() for p in index_paths(cfg).values()):
        pytest.skip("index not built; run experiments/01_build.py")
    return load_env()


@pytest.fixture(scope="session")
def xq_small(env):
    """500 tune queries: tests never touch the test split."""
    return env.xq[env.tune[:500]]


@pytest.fixture(scope="session")
def gt_small(env):
    return env.gt[env.tune[:500]]
