import pandas as pd
import pytest

from qtrix_engine.workspace import DatasetNotFound

from .conftest import make_wages


def test_import_csv_parquet_excel(ctx, tmp_path, wages_csv):
    ws = ctx.workspace
    csv = ws.import_file(wages_csv)
    assert csv.name == "wages" and csv.n_rows == 480
    pq = tmp_path / "w.parquet"
    make_wages().to_parquet(pq)
    xl = tmp_path / "w.xlsx"
    make_wages().head(20).to_excel(xl, index=False)
    assert ws.import_file(pq).n_rows == 480
    assert ws.import_file(xl, name="excel_sample").n_rows == 20
    assert {d.name for d in ws.list()} == {"wages", "w", "excel_sample"}


def test_duplicate_names_get_suffix(ctx):
    a = ctx.workspace.add_dataframe(pd.DataFrame({"x": [1]}), "d", "t")
    b = ctx.workspace.add_dataframe(pd.DataFrame({"x": [2]}), "d", "t")
    assert (a.name, b.name) == ("d", "d_2")


def test_lookup_by_name_and_missing(ctx, wages):
    assert ctx.workspace.get("wages").id == wages.id
    with pytest.raises(DatasetNotFound):
        ctx.workspace.get("nope")


def test_sql_is_read_only(ctx, wages):
    assert len(ctx.workspace.sql("select id, avg(lwage) from wages group by id")) == 60
    for bad in ("drop table wages", "select 1; drop table wages", "insert into x values (1)"):
        with pytest.raises(ValueError):
            ctx.workspace.sql(bad)


def test_sql_error_lists_valid_names(ctx, wages):
    with pytest.raises(ValueError, match=r"Available views and columns.*wages \(.*lwage"):
        ctx.workspace.sql('select count(*) from "wages-2026"')


def test_panel_structure_validation(ctx, wages):
    info = ctx.workspace.set_panel_structure("wages", "id", "year")
    assert (info.entity_col, info.time_col) == ("id", "year")
    with pytest.raises(ValueError):
        ctx.workspace.set_panel_structure("wages", "firm", "year")
