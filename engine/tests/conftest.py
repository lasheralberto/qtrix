from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qtrix_engine.config import Settings
from qtrix_engine.context import EngineContext, set_context
from qtrix_engine.workspace import Workspace


def make_wages(seed: int = 0) -> pd.DataFrame:
    """Synthetic wage panel with known coefficients (educ 0.08, exper 0.03, union 0.10)."""
    rng = np.random.default_rng(seed)
    n_id, t = 60, 8
    ids = np.repeat(np.arange(n_id), t)
    year = np.tile(np.arange(2000, 2000 + t), n_id)
    alpha = rng.normal(size=n_id)[ids]
    educ = rng.integers(8, 20, ids.size)
    exper = rng.integers(0, 30, ids.size)
    union = (rng.random(ids.size) < 0.3).astype(int)
    lwage = 0.5 + 0.08 * educ + 0.03 * exper + 0.10 * union + alpha + rng.normal(scale=0.3, size=ids.size)
    return pd.DataFrame({"id": ids, "year": year, "educ": educ, "exper": exper, "union": union, "lwage": lwage,
                         "high": (lwage > lwage.mean()).astype(int)})


@pytest.fixture
def ctx(tmp_path: Path) -> EngineContext:
    settings = Settings(home=tmp_path, token=None, local_only=False)
    return set_context(EngineContext(settings=settings, workspace=Workspace(tmp_path / "ws.duckdb")))


@pytest.fixture
def wages(ctx: EngineContext):
    return ctx.workspace.add_dataframe(make_wages(), "wages", "synthetic")


def make_macro(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """Quarterly AR(1) series from 1990Q1 with period labels as strings."""
    rng = np.random.default_rng(seed)
    gdp, rate = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        gdp[i] = 0.7 * gdp[i - 1] + rng.normal()
        rate[i] = 0.5 * rate[i - 1] + 0.2 * gdp[i - 1] + rng.normal()
    quarters = [f"{1990 + i // 4}Q{i % 4 + 1}" for i in range(n)]
    return pd.DataFrame({"quarter": quarters, "gdp": gdp, "rate": rate})


@pytest.fixture
def macro(ctx: EngineContext):
    info = ctx.workspace.add_dataframe(make_macro(), "macro", "synthetic")
    return ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")


@pytest.fixture
def wages_csv(tmp_path: Path) -> Path:
    path = tmp_path / "wages.csv"
    make_wages().to_csv(path, index=False)
    return path
