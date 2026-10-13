"""Page structure (ADR 0002 §1): detection, regularisation, structure-aware tools and migration."""

import asyncio

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from qtrix_engine.tools import all_tools
from qtrix_engine.workfile import structure as st
from qtrix_engine.workspace import Workspace

from .conftest import make_macro, make_wages


def tools():
    return {t.name: t.fn for t in all_tools()}


# ---- detection ----------------------------------------------------------------------

@pytest.mark.parametrize(("values", "freq"), [
    (list(range(1960, 2020)), "A"),
    ([f"{1990 + i // 4}Q{i % 4 + 1}" for i in range(40)], "Q"),
    ([f"{2000 + i // 12}M{i % 12 + 1:02d}" for i in range(36)], "M"),
    (pd.date_range("2020-01-31", periods=24, freq="ME").strftime("%Y-%m-%d").tolist(), "M"),
    (pd.date_range("2021-01-04", periods=30, freq="W-MON").tolist(), "W"),
    (pd.bdate_range("2022-01-03", periods=60).tolist(), "D5"),
    (pd.date_range("2022-01-01", periods=60, freq="D").tolist(), "D7"),
])
def test_suggests_time_series_with_frequency(values, freq):
    df = pd.DataFrame({"date" if freq != "A" else "year": values, "y": np.arange(len(values), dtype=float)})
    s = st.suggest(df)
    assert s is not None and (s.kind, s.freq) == ("time_series", freq)


def test_suggests_panel_for_entity_by_year():
    s = st.suggest(make_wages())
    assert s is not None
    assert (s.kind, s.freq, s.id_col, s.date_col) == ("panel", "A", "id", "year")


def test_surveys_are_never_mistaken_for_time():
    rng = np.random.default_rng(0)
    survey = pd.DataFrame({
        "person": np.arange(5000), "year": 2019, "age": rng.integers(18, 65, 5000),
        "wage": rng.lognormal(3, 0.5, 5000), "female": rng.integers(0, 2, 5000),
    })
    assert st.suggest(survey) is None
    # A unique 1990..2049 integer column only counts as years when its name says so.
    assert st.suggest(pd.DataFrame({"x": range(1990, 2050), "y": range(60)})) is None
    pooled = pd.DataFrame({"year": np.repeat([2018, 2019], 500), "wage": rng.normal(size=1000)})
    assert st.suggest(pooled) is None


@pytest.mark.parametrize(("df", "expected"), [
    (pd.DataFrame({"periodo": [f"{2015 + i // 12}-{i % 12 + 1:02d}" for i in range(48)], "v": range(48)}),
     ("time_series", "M", "periodo", None)),
    (pd.DataFrame({"fecha": pd.date_range("2020-01-31", periods=30, freq="ME").strftime("%d/%m/%Y"),
                   "v": range(30)}), ("time_series", "M", "fecha", None)),
    (pd.DataFrame({"q": [f"Q{i % 4 + 1} {1990 + i // 4}" for i in range(20)], "v": range(20)}),
     ("time_series", "Q", "q", None)),
    (pd.DataFrame({"iso3": np.repeat(["ESP", "FRA", "DEU"], 20), "year": np.tile(range(2000, 2020), 3),
                   "gdp": np.arange(60.0)}), ("panel", "A", "year", "iso3")),
    (pd.DataFrame({"firm": np.repeat(list("abcd"), 24),
                   "date": np.tile(pd.date_range("2019-01-01", periods=24, freq="MS"), 4),
                   "sales": np.arange(96.0)}), ("panel", "M", "date", "firm")),
    (pd.DataFrame({"hh_id": range(300), "income": np.linspace(1, 9, 300), "edad": np.arange(300) % 60 + 18}),
     None),
])
def test_detection_on_realistic_layouts(df, expected):
    s = st.suggest(df)
    assert (s and (s.kind, s.freq, s.date_col, s.id_col)) == expected


def test_import_keeps_cross_section_and_records_suggestion(ctx, tmp_path):
    path = tmp_path / "macro.csv"
    make_macro().to_csv(path, index=False)
    info = ctx.workspace.import_file(path)
    assert info.kind == "cross_section" and info.n_rows == 100
    assert info.suggestion["kind"] == "time_series" and info.suggestion["freq"] == "Q"


# ---- regularisation -------------------------------------------------------------------

def test_time_series_fills_gaps_and_indexes_by_date(ctx):
    df = make_macro(20).drop(index=[5]).reset_index(drop=True)  # 1991Q2 missing
    info = ctx.workspace.add_dataframe(df, "gappy", "t")
    info = ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")
    assert (info.kind, info.freq, info.start, info.end, info.gaps, info.n_rows) == \
        ("time_series", "Q", "1990Q1", "1994Q4", 1, 20)
    frame = ctx.workspace.frame("gappy")
    assert isinstance(frame.index, pd.DatetimeIndex) and frame.index.freqstr.startswith("QS")
    assert np.isnan(frame.loc["1991-04-01", "gdp"])
    assert ctx.workspace.df("gappy")["quarter"].tolist()[5] == "1991Q2"  # label filled in
    assert "__obs" not in ctx.workspace.df("gappy").columns


def test_time_series_from_start_period(ctx):
    info = ctx.workspace.add_dataframe(pd.DataFrame({"y": np.arange(8.0)}), "rows", "t")
    info = ctx.workspace.set_structure(info.id, "time_series", freq="M", start="2001m11")
    assert (info.start, info.end) == ("2001M11", "2002M06")


def test_panel_is_balanced_with_missing_years(ctx):
    df = make_wages()
    df = df[~((df["id"] == 3) & (df["year"] == 2004))]
    info = ctx.workspace.add_dataframe(df, "unbalanced", "t")
    info = ctx.workspace.set_panel_structure(info.id, "id", "year")
    assert (info.kind, info.freq, info.gaps, info.n_rows) == ("panel", "A", 1, 480)
    frame = ctx.workspace.frame("unbalanced")
    assert frame.index.names == ["id", "obs"]
    assert ctx.workspace.describe_structure("unbalanced")["n_entities"] == 60


def test_duplicates_are_rejected_with_examples(ctx, wages):
    with pytest.raises(ValueError, match="share the same"):
        ctx.workspace.set_structure("wages", "time_series", date_col="year")


def test_restructure_back_to_cross_section_restores_rows(ctx):
    df = make_macro(20).drop(index=[5]).reset_index(drop=True)
    info = ctx.workspace.add_dataframe(df, "back", "t")
    ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")
    info = ctx.workspace.set_structure(info.id, "cross_section")
    assert info.n_rows == 19 and info.kind == "cross_section"
    assert ctx.workspace.df("back")["quarter"].tolist() == df["quarter"].tolist()


def test_undated_panel(ctx):
    df = pd.DataFrame({"student": np.repeat([1, 2, 3], 3), "wave": np.tile([1, 2, 3], 3), "score": np.arange(9.0)})
    info = ctx.workspace.add_dataframe(df, "waves", "t")
    info = ctx.workspace.set_structure(info.id, "panel_undated", id_col="student", t_col="wave")
    assert (info.entity_col, info.time_col) == ("student", "wave")
    assert ctx.workspace.frame("waves").index.names == ["student", "wave"]


# ---- structure-aware tools ---------------------------------------------------------------

def test_tools_declare_the_structures_they_apply_to():
    specs = {t.name: t for t in all_tools()}
    assert specs["ts_arima"].kinds == {"time_series"}
    assert specs["panel_fit"].kinds == {"panel", "panel_undated"}
    assert specs["ols_fit"].kinds is None
    assert not specs["ts_arima"].applies_to({"cross_section"})
    assert specs["ts_arima"].applies_to({"cross_section", "time_series"})


def test_agent_only_sees_tools_that_apply(ctx, wages):
    from pydantic_ai.tools import ToolDefinition

    from qtrix_engine.agents.lead import _only_for_structures

    spec = next(t for t in all_tools() if t.name == "ts_arima")
    prepare = _only_for_structures(spec)
    tool_def = ToolDefinition(name="ts_arima")
    assert asyncio.run(prepare(None, tool_def)) is None
    macro = ctx.workspace.add_dataframe(make_macro(), "m", "t")
    ctx.workspace.set_structure(macro.id, "time_series", date_col="quarter")
    assert asyncio.run(prepare(None, tool_def)) is tool_def


def test_suggestion_appears_in_actionable_errors(ctx, tmp_path):
    path = tmp_path / "macro.csv"
    make_macro().to_csv(path, index=False)
    ctx.workspace.import_file(path)
    with pytest.raises(ValueError, match=r"confirm with data_set_structure\('macro', kind='time_series'"):
        tools()["ts_arima"]("macro", "gdp", order=[1, 0, 0])


def test_panel_fit_uses_declared_dated_panel(ctx, wages):
    t = tools()
    t["data_set_structure"]("wages", "panel", date_col="year", id_col="id")
    out = t["panel_fit"]("wages", "lwage ~ 1 + exper + union", estimator="fe")
    assert ctx.results.get(out["result_id"]).summary["n_entities"] == 60


def test_transform_keeps_structure_and_line_chart_uses_dates(ctx, macro, wages):
    t = tools()
    out = t["data_transform"]("macro", {"gdp2": "gdp * 2"}, save_as="macro2")
    derived = ctx.workspace.get(out["dataset_id"])
    assert (derived.kind, derived.freq, derived.start) == ("time_series", "Q", "1990Q1")
    chart = t["viz_line"]("macro2", ["gdp", "gdp2"])
    assert ctx.panel.get(chart["block_id"]).data["data"][0]["x"][0].startswith("1990-01-01")
    with pytest.raises(ValueError, match="pass x"):
        t["viz_line"]("wages", ["lwage"])


# ---- persistence ----------------------------------------------------------------------

def test_legacy_catalog_is_migrated(tmp_path):
    db = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE _qtrix_datasets (id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL, source VARCHAR "
                "NOT NULL, table_name VARCHAR NOT NULL, created_at TIMESTAMP NOT NULL, entity_col VARCHAR, "
                "time_col VARCHAR)")
    con.register("w", make_wages())
    con.execute("CREATE TABLE ds_w AS SELECT * FROM w")
    con.execute("CREATE TABLE ds_x AS SELECT 1 AS x")
    con.execute("INSERT INTO _qtrix_datasets VALUES ('a1', 'wages', 'csv', 'ds_w', now(), 'id', 'year'), "
                "('b2', 'x', 'csv', 'ds_x', now(), NULL, NULL)")
    con.close()
    ws = Workspace(db)
    assert {d.name: d.kind for d in ws.list()} == {"wages": "panel", "x": "cross_section"}
    assert ws.get("wages").n_rows == 480


def test_structure_survives_reopen(tmp_path):
    ws = Workspace(tmp_path / "ws.duckdb")
    info = ws.add_dataframe(make_macro(), "macro", "t")
    ws.set_structure(info.id, "time_series", date_col="quarter")
    del ws
    again = Workspace(tmp_path / "ws.duckdb")
    assert again.get("macro").structure_text == "time series quarterly 1990Q1–2014Q4 from 'quarter'"


def test_structure_endpoint(tmp_path):
    from qtrix_engine.api import create_app
    from qtrix_engine.config import Settings

    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    path = tmp_path / "macro.csv"
    make_macro().to_csv(path, index=False)
    info = c.post("/datasets/import", json={"path": str(path)}).json()
    assert info["kind"] == "cross_section" and info["suggestion"]["freq"] == "Q"
    r = c.put(f"/datasets/{info['id']}/structure", json={"kind": "time_series", "date_col": "quarter"})
    assert r.status_code == 200 and r.json()["start"] == "1990Q1"
    bad = c.put(f"/datasets/{info['id']}/structure", json={"kind": "panel", "date_col": "quarter"})
    assert bad.status_code == 400 and "id_col" in bad.json()["detail"]
