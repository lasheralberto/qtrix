"""Local data workspace: every imported dataset becomes a DuckDB table (a *page*) plus a catalog row.

Data never leaves this file; tools receive datasets by ``dataset_id`` and load them on demand. Each page
declares the structure its data actually have (ADR 0002 §1): a cross-section by default, or a time series /
panel once declared. Dated pages are regularised and carry hidden ``__obs`` / ``__id`` columns; the original
rows are kept in an immutable ``src_*`` snapshot so a page can always be re-structured.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from . import expr
from .callctx import current_recording
from .journal import BEFORE_DDL, JOURNAL_DDL
from .objects.store import OBJECTS_DDL, VERSIONS_DDL
from .workfile import sample as sp
from .workfile import structure as st

log = logging.getLogger(__name__)

# 4: named objects (_qtrix_objects, _qtrix_object_versions); 5: journal (_qtrix_journal, _qtrix_journal_before)
# and an immutable src_<id> snapshot for every imported page
SCHEMA_VERSION = 5

PAGES_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_pages (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    table_name VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    kind VARCHAR NOT NULL DEFAULT 'cross_section',
    freq VARCHAR NOT NULL DEFAULT 'U',
    start_label VARCHAR,
    end_label VARCHAR,
    id_col VARCHAR,
    date_col VARCHAR,
    t_col VARCHAR,
    gaps INTEGER NOT NULL DEFAULT 0,
    src_table VARCHAR,
    suggestion VARCHAR,
    data_version INTEGER NOT NULL DEFAULT 1,
    sample_spec VARCHAR
)
"""
META_DDL = "CREATE TABLE IF NOT EXISTS _qtrix_meta (key VARCHAR PRIMARY KEY, value VARCHAR)"
SERIES_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_series (
    page_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    expression VARCHAR NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (page_id, name)
)
"""
PAGE_COLS = ("id, name, source, table_name, created_at, kind, freq, start_label, end_label, id_col, date_col, "
             "t_col, gaps, src_table, suggestion, data_version, sample_spec")

FILE_FORMATS = {
    ".csv": "csv", ".tsv": "csv", ".txt": "csv",
    ".parquet": "parquet", ".pq": "parquet",
    ".xlsx": "excel", ".xls": "excel", ".xlsm": "excel",
}
FREQ_ALIASES = {"Y": "A", "D": "D7", "B": "D5", "H": "S"}


@dataclass
class DatasetInfo:
    id: str
    name: str
    source: str
    table_name: str
    created_at: str
    kind: str = "cross_section"
    freq: str = "U"
    start: str | None = None
    end: str | None = None
    id_col: str | None = None
    date_col: str | None = None
    t_col: str | None = None
    gaps: int = 0
    data_version: int = 1
    suggestion: dict[str, Any] | None = None
    sample: str | None = None  # active sample in text form (None = @all)
    src_table: str | None = None
    n_rows: int = 0
    columns: list[dict[str, str]] | None = None

    @property
    def dated(self) -> bool:
        return self.kind in st.DATED_KINDS

    # Pre-ADR-0002 names, still used by the UI and older callers.
    @property
    def entity_col(self) -> str | None:
        return self.id_col if self.kind in st.PANEL_KINDS else None

    @property
    def time_col(self) -> str | None:
        if self.kind not in st.PANEL_KINDS:
            return None
        return self.t_col if self.kind == "panel_undated" else (None if self.date_col == st.OBS else self.date_col)

    @property
    def structure_text(self) -> str:
        return st.describe(self.kind, self.freq, self.start, self.end, self.id_col, self.date_col, self.t_col)

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("src_table")
        out.update(entity_col=self.entity_col, time_col=self.time_col, structure=self.structure_text)
        return out


class DatasetNotFound(KeyError):
    pass


def _slug(name: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower() or "dataset"
    return slug if not slug[0].isdigit() else f"d_{slug}"


def _quote(cols: Iterable[str]) -> str:
    return ", ".join('"' + c.replace('"', '""') + '"' for c in cols)


_QUOTED = re.compile(r"""('(?:[^']|'')*'|"(?:[^"]|"")*")""")


def _quote_names(expr: str, names: Iterable[str]) -> str:
    """Quote column names in a SQL expression so columns called ``union`` or ``order`` just work.

    String literals and identifiers the user already quoted are left alone.
    """
    names = sorted({n for n in names if n}, key=len, reverse=True)
    if not names:
        return expr
    pattern = re.compile(r"(?<![\w.])(" + "|".join(re.escape(n) for n in names) + r")(?![\w(])", re.I)
    exact = {n.lower(): n for n in names}
    parts = _QUOTED.split(expr)
    for i in range(0, len(parts), 2):  # even parts are outside quotes
        parts[i] = pattern.sub(lambda m: '"' + exact[m.group(1).lower()] + '"', parts[i])
    return "".join(parts)


class Workspace:
    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._con = duckdb.connect(str(db_path))
        self._lock = threading.RLock()
        with self._lock:
            self._con.execute(META_DDL)
            self._con.execute(PAGES_DDL)
            self._con.execute(SERIES_DDL)
            self._con.execute(OBJECTS_DDL)
            self._con.execute(VERSIONS_DDL)
            self._con.execute("ALTER TABLE _qtrix_pages ADD COLUMN IF NOT EXISTS sample_spec VARCHAR")
            self._migrate()

    def checkpoint(self) -> None:
        """Flush the write-ahead log into the database file (before copying it)."""
        with self._lock:
            self._con.execute("CHECKPOINT")

    def copy_to(self, path: Path | str) -> None:
        """Write a complete copy of this database to ``path`` (an open DuckDB file cannot be copied on Windows)."""
        with self._lock:
            source = self._con.execute("SELECT current_database()").fetchone()[0]
            self._con.execute(f"ATTACH '{Path(path).as_posix()}' AS qtrix_copy")
            try:
                self._con.execute(f'COPY FROM DATABASE "{source}" TO qtrix_copy')
            finally:
                self._con.execute("DETACH qtrix_copy")

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # ---- schema ------------------------------------------------------------------
    def _migrate(self) -> None:
        """Forward-only migrations, tracked in ``_qtrix_meta.schema_version``."""
        row = self._con.execute("SELECT value FROM _qtrix_meta WHERE key = 'schema_version'").fetchone()
        version = int(row[0]) if row else 0
        if version < 5:  # the journal arrives; pages that already exist simply have no snapshot (src_table NULL)
            self._con.execute(JOURNAL_DDL)
            self._con.execute(BEFORE_DDL)
        legacy = self._con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '_qtrix_datasets'").fetchone()[0]
        if not legacy:
            self._set_meta("schema_version", str(SCHEMA_VERSION))
            return
        rows = self._con.execute(
            "SELECT id, name, source, table_name, created_at, entity_col, time_col FROM _qtrix_datasets").fetchall()
        for r in rows:
            self._con.execute("INSERT INTO _qtrix_pages (id, name, source, table_name, created_at) VALUES "
                              "(?, ?, ?, ?, ?)", list(r[:5]))
        self._con.execute("DROP TABLE _qtrix_datasets")
        self._set_meta("schema_version", str(SCHEMA_VERSION))
        for r in rows:
            if r[5] and r[6]:
                try:
                    self.set_panel_structure(r[0], r[5], r[6])
                except ValueError as exc:  # keep the data as a cross-section rather than fail startup
                    log.warning("Dataset %s: panel structure not migrated (%s)", r[1], exc)
        log.info("Migrated %d datasets to the page catalog (schema %d)", len(rows), SCHEMA_VERSION)

    def _set_meta(self, key: str, value: str) -> None:
        self._con.execute("INSERT OR REPLACE INTO _qtrix_meta VALUES (?, ?)", [key, value])

    # ---- catalog -----------------------------------------------------------------
    def list(self) -> list[DatasetInfo]:
        with self._lock:
            ids = [r[0] for r in self._con.execute("SELECT id FROM _qtrix_pages ORDER BY created_at").fetchall()]
        return [self.get(i) for i in ids]

    def page_data_version(self, dataset_id: str) -> int | None:
        """Cheap ``data_version`` of a page (None if it no longer exists); objects compare it to go stale."""
        with self._lock:
            row = self._con.execute("SELECT data_version FROM _qtrix_pages WHERE id = ? OR name = ?",
                                    [dataset_id, dataset_id]).fetchone()
        return None if row is None else int(row[0])

    def kinds(self) -> set[str]:
        """Structures present in the workspace (used to offer only the tools that apply)."""
        with self._lock:
            return {r[0] for r in self._con.execute("SELECT DISTINCT kind FROM _qtrix_pages").fetchall()}

    def get(self, dataset_id: str) -> DatasetInfo:
        """Look up a dataset by id or by name."""
        with self._lock:
            row = self._con.execute(f"SELECT {PAGE_COLS} FROM _qtrix_pages WHERE id = ? OR name = ?",
                                    [dataset_id, dataset_id]).fetchone()
            if row is None:
                raise DatasetNotFound(f"Unknown dataset '{dataset_id}'")
            table = row[3]
            n_rows = self._con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            cols = self._con.execute(f'DESCRIBE "{table}"').fetchall()
        return DatasetInfo(
            id=row[0], name=row[1], source=row[2], table_name=table, created_at=row[4].isoformat(),
            kind=row[5], freq=row[6], start=row[7], end=row[8], id_col=row[9], date_col=row[10], t_col=row[11],
            gaps=row[12], src_table=row[13], suggestion=json.loads(row[14]) if row[14] else None,
            data_version=row[15], sample=row[16], n_rows=n_rows,
            columns=[{"name": c[0], "type": c[1]} for c in cols if c[0] not in st.HIDDEN],
        )

    def _user_columns(self, info: DatasetInfo, columns: list[str] | None) -> list[str]:
        names = [c["name"] for c in info.columns or []]
        if columns is None:
            return names
        missing = [c for c in columns if c not in names]
        if missing:
            raise ValueError(f"Columns {missing} not in dataset '{info.name}'. Available: {names}")
        return list(dict.fromkeys(columns))

    def df(self, dataset_id: str, columns: list[str] | None = None, sample: str | None = None) -> pd.DataFrame:
        """Plain table (user columns, stored row order) restricted to the sample.

        ``sample`` overrides the page's active sample for this call (``"@all"`` = every row).
        """
        info = self.get(dataset_id)
        cols = _quote(self._user_columns(info, columns))
        with self._lock:
            out = self._con.execute(f'SELECT {cols} FROM "{info.table_name}"').df()
        res = self.resolve_sample(info, sample)
        return out if res.mask is None else out[res.mask].reset_index(drop=True)

    def frame(self, dataset_id: str, columns: list[str] | None = None, sample: str | None = None) -> pd.DataFrame:
        """Table indexed by its structure: DatetimeIndex for time series, (entity, time) for panels.

        Rows outside the sample are dropped, except on time series, which keep a regular index: the range
        is cut to the sample's first and last period and any period inside it that is excluded (several
        ranges, a condition) becomes missing.
        """
        info = self.get(dataset_id)
        wanted = self._user_columns(info, columns)
        if info.kind == "panel_undated" and info.t_col not in wanted:
            wanted.append(info.t_col)
        with self._lock:
            hidden = [c for c in st.HIDDEN if c in self._table_columns(info.table_name)]
            df = self._con.execute(f'SELECT {_quote([*hidden, *wanted])} FROM "{info.table_name}"').df()
        res = self.resolve_sample(info, sample)
        if res.mask is not None:
            if info.kind == "time_series":
                keep = np.flatnonzero(res.mask)
                df[wanted] = df[wanted].where(pd.Series(res.mask, index=df.index))
                df = df.iloc[keep[0]:keep[-1] + 1]
            else:
                df = df[res.mask].reset_index(drop=True)
        return st.index_frame(df, kind=info.kind, freq=info.freq, id_col=info.id_col, t_col=info.t_col)

    # ---- generated series -----------------------------------------------------------
    def genr(self, dataset_id: str, name: str, expression: str, overwrite: bool = False) -> dict[str, Any]:
        """Add a series computed from an expression over every row of the page (ADR 0002 §3).

        The expression may use the time operators of ``expr`` (lags, differences, trends…). Its definition is
        kept, so the series is recomputed if the page is re-structured.
        """
        info = self.get(dataset_id)
        replaced = name in {c["name"] for c in info.columns or []}
        with self._lock:
            series = self._compute_series(info, name, expression, overwrite)
            self._con.execute(
                "INSERT INTO _qtrix_series VALUES (?, ?, ?, ?) ON CONFLICT (page_id, name) "
                "DO UPDATE SET expression = excluded.expression", [info.id, name, expression, datetime.now(UTC)])
        values = series.to_numpy(dtype=float)
        return {"dataset": info.name, "series": name, "expression": expression, "n_rows": len(values),
                "n_missing": int(np.isnan(values).sum()), "replaced": replaced}

    def generated_series(self, dataset_id: str) -> dict[str, str]:
        with self._lock:
            return dict(self._con.execute(
                "SELECT name, expression FROM _qtrix_series WHERE page_id = ? ORDER BY created_at",
                [self.get(dataset_id).id]).fetchall())

    def set_series(self, dataset_id: str, name: str, values: list[float | None],
                   overwrite: bool = False) -> dict[str, Any]:
        """Add or replace a series with values computed outside the engine (``qx.series_from``, E7·07b).

        ``values`` follow the stored row order of the page (every row, not the sample). A series defined by an
        expression loses its definition: it now holds plain values.
        """
        info = self.get(dataset_id)
        replaced = name in {c["name"] for c in info.columns or []}
        self._check_series_name(info, name, overwrite)
        if len(values) != info.n_rows:
            raise ValueError(f"'{name}' has {len(values)} values but '{info.name}' has {info.n_rows} rows: pass one "
                             "value per row of the page (every row, not just the sample)")
        array = np.array([np.nan if v is None else v for v in values], dtype=float)
        with self._lock:
            df = self._con.execute(f'SELECT * FROM "{info.table_name}"').df()
            df[name] = array
            self._replace_table(info.table_name, df)
            self._con.execute("DELETE FROM _qtrix_series WHERE page_id = ? AND name = ?", [info.id, name])
            if replaced:
                self._con.execute("UPDATE _qtrix_pages SET data_version = data_version + 1 WHERE id = ?", [info.id])
        return {"dataset": info.name, "series": name, "n_rows": len(array),
                "n_missing": int(np.isnan(array).sum()), "replaced": replaced}

    @staticmethod
    def _check_series_name(info: DatasetInfo, name: str, overwrite: bool) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name.startswith("__"):
            raise ValueError(f"'{name}' is not a valid series name: use letters, digits and underscores, "
                             "starting with a letter")
        if name in {c["name"] for c in info.columns or []} and not overwrite:
            raise ValueError(f"'{name}' already exists in '{info.name}'; choose another name or pass "
                             "overwrite=True to replace it")

    def _compute_series(self, info: DatasetInfo, name: str, expression: str, overwrite: bool) -> pd.Series:
        self._check_series_name(info, name, overwrite)
        existing = {c["name"] for c in info.columns or []}
        frame = self.frame(info.id, sample="@all")
        ctx = expr.Context(info.kind, info.freq, frame.index, tuple(map(str, frame.columns)))
        series = expr.evaluate(expression, frame, ctx)
        if len(series) != info.n_rows:
            raise ValueError(f"The expression produced {len(series)} values for {info.n_rows} rows")
        df = self._con.execute(f'SELECT * FROM "{info.table_name}"').df()
        df[name] = series.to_numpy()
        self._replace_table(info.table_name, df)
        if name in existing:  # the data of an existing column changed
            self._con.execute("UPDATE _qtrix_pages SET data_version = data_version + 1 WHERE id = ?", [info.id])
        return series

    def _regenerate(self, page_id: str) -> None:
        """Recompute generated series after the page was rebuilt (re-structured)."""
        for name, expression in self.generated_series(page_id).items():
            try:
                self._compute_series(self.get(page_id), name, expression, overwrite=True)
            except ValueError as exc:  # e.g. a lag on a page that is no longer a time series
                log.warning("Generated series '%s' dropped after re-structuring: %s", name, exc)
                self._con.execute("DELETE FROM _qtrix_series WHERE page_id = ? AND name = ?", [page_id, name])

    # ---- sample --------------------------------------------------------------------
    def resolve_sample(self, dataset: str | DatasetInfo, sample: str | None = None) -> sp.ResolvedSample:
        """Evaluate a sample over the page: the row mask, how many rows it keeps and a readable description.

        ``sample=None`` uses the page's active sample. Raises when the sample is invalid or selects nothing.
        """
        info = dataset if isinstance(dataset, DatasetInfo) else self.get(dataset)
        spec = sp.parse(info.sample if sample is None else sample)
        n = info.n_rows
        if spec.is_all:
            text = f"{info.start}–{info.end}" if info.kind != "cross_section" else f"all {n} observations"
            return sp.ResolvedSample(spec, None, n, n, info.start, info.end, text)
        key = st.OBS if info.dated else info.t_col if info.kind == "panel_undated" else None
        select = [f'"{key}" AS k' if key else "1 AS k"]
        if spec.where:
            where = _quote_names(spec.where, [c["name"] for c in info.columns or []])
            select.append(f"COALESCE(CAST(({where}) AS BOOLEAN), FALSE) AS m")
        try:
            with self._lock:
                rows = self._con.execute(f'SELECT {", ".join(select)} FROM "{info.table_name}"').df()
        except duckdb.Error as exc:
            msg = str(exc).splitlines()[0]
            raise sp.SampleError(f"Cannot evaluate the sample condition '{spec.where}': {msg}. Use column names "
                                 f"of '{info.name}' and SQL syntax, e.g. age >= 25 and female = 1") from exc
        keys = pd.Series(rows["k"]) if key else pd.Series(np.arange(1, n + 1))
        mask = sp.range_mask(spec, keys, kind=info.kind, freq=info.freq)
        if spec.where:
            mask &= rows["m"].to_numpy(dtype=bool)
        selected = int(mask.sum())
        if selected == 0:
            span = (f"The data have {n} rows." if info.kind == "cross_section"
                    else f"The data span {info.start}–{info.end}.")
            raise sp.SampleError(f"The sample '{spec.text}' selects no observations in '{info.name}'. {span}")
        first = last = None
        if info.dated:
            first, last = (st.label(pd.Timestamp(keys[mask].min()), info.freq),
                           st.label(pd.Timestamp(keys[mask].max()), info.freq))
        elif info.kind == "panel_undated":
            first, last = str(keys[mask].min()), str(keys[mask].max())
        if info.kind == "cross_section":
            text = f"{spec.text} ({selected} of {n} observations)"
        else:
            text = f"{spec.text} ({first}–{last}; {selected} of {n} rows)"
        return sp.ResolvedSample(spec, None if selected == n else mask, selected, n, first, last, text)

    def set_sample(self, dataset_id: str, sample: str | None) -> sp.ResolvedSample:
        """Set the page's active sample (``None`` or ``"@all"`` resets it). Validated before it is stored."""
        info = self.get(dataset_id)
        res = self.resolve_sample(info, sample if sample is not None else "@all")
        with self._lock:
            self._con.execute("UPDATE _qtrix_pages SET sample_spec = ? WHERE id = ?",
                              [None if res.spec.is_all else res.spec.text, info.id])
        return res

    def preview(self, dataset_id: str, limit: int = 50, offset: int = 0) -> pd.DataFrame:
        info = self.get(dataset_id)
        cols = _quote(self._user_columns(info, None))
        with self._lock:
            return self._con.execute(f'SELECT {cols} FROM "{info.table_name}" LIMIT ? OFFSET ?',
                                     [limit, offset]).df()

    def require(self, dataset_id: str, kinds: Iterable[str], what: str) -> DatasetInfo:
        """Fail with an actionable message when an operation does not apply to the page's structure."""
        info = self.get(dataset_id)
        kinds = set(kinds)
        if info.kind in kinds:
            return info
        need = " or ".join(k.replace("_", " ") for k in sorted(kinds))
        msg = f"{what} needs a {need}, but '{info.name}' is a {info.structure_text}."
        if info.suggestion and info.suggestion["kind"] in kinds:
            s = info.suggestion
            args = f"kind={s['kind']!r}, freq={s['freq']!r}, date_col={s['date_col']!r}"
            args += f", id_col={s['id_col']!r}" if s.get("id_col") else ""
            msg += f" It looks like one ({s['reason']}): confirm with data_set_structure({info.name!r}, {args})."
        elif kinds & st.DATED_KINDS:
            msg += (" If the data are dated, declare it with data_set_structure(dataset_id, kind='time_series', "
                    "freq='Q', date_col='<date column>'), or start='1990Q1' for undated rows in time order.")
        raise st.StructureError(msg)

    # ---- structure -----------------------------------------------------------------
    def set_structure(self, dataset_id: str, kind: str, freq: str | None = None, date_col: str | None = None,
                      id_col: str | None = None, t_col: str | None = None, start: str | None = None) -> DatasetInfo:
        """Declare what the page is. Dated structures regularise the table; the source rows are kept."""
        info = self.get(dataset_id)
        if kind not in st.KINDS:
            raise st.StructureError(f"Unknown structure '{kind}'. Use one of {', '.join(st.KINDS)}")
        freq = FREQ_ALIASES.get(freq.upper(), freq.upper()) if freq else None
        if kind in {"cross_section", "panel_undated"}:
            freq = "U"
        with self._lock:
            source_cols = self._table_columns(info.src_table or info.table_name)
            for col in (date_col, id_col, t_col):
                if col and col not in source_cols:
                    raise ValueError(f"Column '{col}' not in dataset '{info.name}'")
            if kind == "cross_section" and info.src_table is None:
                n = info.n_rows
                reg = st.Regularized(pd.DataFrame(), "1" if n else None, str(n) if n else None, 0)
            else:
                src = info.src_table or self._snapshot(info)
                base = self._con.execute(f'SELECT * FROM "{src}"').df()
                if kind in st.DATED_KINDS and not freq:
                    freq = self._infer_freq(base, date_col, kind)
                reg = st.regularize(base, kind=kind, freq=freq, date_col=date_col, id_col=id_col, t_col=t_col,
                                    start=start)
                self._replace_table(info.table_name, reg.frame)
            self._con.execute(
                "UPDATE _qtrix_pages SET kind = ?, freq = ?, start_label = ?, end_label = ?, id_col = ?, "
                "date_col = ?, t_col = ?, gaps = ?, suggestion = NULL, sample_spec = NULL, "
                "data_version = data_version + 1 WHERE id = ?",
                [kind, freq or "U", reg.start, reg.end, id_col, date_col if kind in st.DATED_KINDS else None,
                 t_col if kind == "panel_undated" else None, reg.gaps, info.id],
            )
            self._regenerate(info.id)
        return self.get(info.id)

    def set_panel_structure(self, dataset_id: str, entity_col: str | None, time_col: str | None) -> DatasetInfo:
        """Declare entity and time columns; dated when the time column holds dates/years, undated otherwise."""
        info = self.get(dataset_id)
        names = {c["name"] for c in info.columns or []}
        for col in (entity_col, time_col):
            if col and col not in names:
                raise ValueError(f"Column '{col}' not in dataset '{info.name}'")
        if not entity_col or not time_col:
            return self.set_structure(dataset_id, "cross_section")
        times = self.df(dataset_id, [time_col], sample="@all")[time_col]
        parsed = st.as_dates(times, time_col)
        freq = (parsed[1] or st.infer_freq(parsed[0])) if parsed is not None else None
        if freq:
            return self.set_structure(dataset_id, "panel", freq=freq, date_col=time_col, id_col=entity_col)
        return self.set_structure(dataset_id, "panel_undated", id_col=entity_col, t_col=time_col)

    def describe_structure(self, dataset_id: str) -> dict[str, Any]:
        info = self.get(dataset_id)
        out: dict[str, Any] = {
            "dataset": info.name, "kind": info.kind, "freq": info.freq, "start": info.start, "end": info.end,
            "id_col": info.id_col, "date_col": info.date_col, "t_col": info.t_col, "n_rows": info.n_rows,
            "filled_gaps": info.gaps, "description": info.structure_text, "suggestion": info.suggestion,
            "active_sample": self.resolve_sample(info).to_dict(),
            "generated_series": self.generated_series(info.id),
        }
        if info.kind in st.PANEL_KINDS:
            with self._lock:
                out["n_entities"] = self._con.execute(
                    f'SELECT COUNT(DISTINCT "{st.ID}") FROM "{info.table_name}"').fetchone()[0]
        return out

    def _infer_freq(self, base: pd.DataFrame, date_col: str | None, kind: str) -> str:
        if date_col:
            parsed = st.as_dates(base[date_col], date_col)
            if parsed is not None:
                freq = parsed[1] or st.infer_freq(parsed[0])
                if freq:
                    return freq
        raise st.StructureError(f"Cannot infer the frequency of a {kind.replace('_', ' ')}: pass freq "
                                f"(one of {', '.join(st.DATED_FREQS)})")

    def _snapshot(self, info: DatasetInfo) -> str:
        src = f"src_{info.id}"
        cols = [c for c in self._table_columns(info.table_name) if c not in st.HIDDEN]
        self._con.execute(f'CREATE TABLE "{src}" AS SELECT {_quote(cols)} FROM "{info.table_name}"')
        self._con.execute("UPDATE _qtrix_pages SET src_table = ? WHERE id = ?", [src, info.id])
        return src

    def _replace_table(self, table: str, df: pd.DataFrame) -> None:
        self._con.register("_qtrix_incoming", df)
        try:
            self._con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _qtrix_incoming')
        finally:
            self._con.unregister("_qtrix_incoming")

    def _table_columns(self, table: str) -> list[str]:
        return [c[0] for c in self._con.execute(f'DESCRIBE "{table}"').fetchall()]

    def delete(self, dataset_id: str, keep_source: bool = False) -> None:
        """Drop a page. ``keep_source`` keeps its ``src_`` snapshot, which a journal replay still needs."""
        info = self.get(dataset_id)
        with self._lock:
            self._con.execute(f'DROP TABLE IF EXISTS "{info.table_name}"')
            if info.src_table and not keep_source:
                self._con.execute(f'DROP TABLE IF EXISTS "{info.src_table}"')
            self._con.execute("DELETE FROM _qtrix_series WHERE page_id = ?", [info.id])
            self._con.execute("DELETE FROM _qtrix_pages WHERE id = ?", [info.id])

    # ---- journal support: page state for undo and replay (E7·06) ---------------------------------
    PAGE_STATE_COLS = ("kind", "freq", "start_label", "end_label", "id_col", "date_col", "t_col", "gaps",
                       "suggestion", "data_version", "sample_spec")

    def page_rows(self) -> dict[str, list[Any]]:
        """Raw catalog rows by page id (compared before/after a call to see which pages it touched)."""
        with self._lock:
            rows = self._con.execute(f"SELECT {PAGE_COLS} FROM _qtrix_pages").fetchall()
        return {r[0]: [v.isoformat() if isinstance(v, datetime) else v for v in r] for r in rows}

    def capture_page_state(self, dataset_id: str, undo_table: str | None) -> dict[str, Any]:
        """Before-image of a page: its structure/sample state, series definitions and (optionally) its data."""
        info = self.get(dataset_id)
        with self._lock:
            cols = ", ".join(self.PAGE_STATE_COLS)
            row = self._con.execute(f"SELECT {cols} FROM _qtrix_pages WHERE id = ?", [info.id]).fetchone()
            series = [[r[0], r[1], r[2].isoformat()] for r in self._con.execute(
                "SELECT name, expression, created_at FROM _qtrix_series WHERE page_id = ? ORDER BY created_at",
                [info.id]).fetchall()]
            if undo_table:
                self._con.execute(f'CREATE OR REPLACE TABLE "{undo_table}" AS SELECT * FROM "{info.table_name}"')
        return {"id": info.id, "name": info.name, "table_name": info.table_name, "table": undo_table,
                "catalog": dict(zip(self.PAGE_STATE_COLS, row, strict=True)), "series": series}

    def restore_page_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            sets = ", ".join(f"{c} = ?" for c in self.PAGE_STATE_COLS)
            self._con.execute(f"UPDATE _qtrix_pages SET {sets} WHERE id = ?",
                              [state["catalog"][c] for c in self.PAGE_STATE_COLS] + [state["id"]])
            if state["table"]:
                target, source = state["table_name"], state["table"]
                self._con.execute(f'CREATE OR REPLACE TABLE "{target}" AS SELECT * FROM "{source}"')
            self._con.execute("DELETE FROM _qtrix_series WHERE page_id = ?", [state["id"]])
            for name, expression, created in state["series"]:
                self._con.execute("INSERT INTO _qtrix_series VALUES (?, ?, ?, ?)",
                                  [state["id"], name, expression, datetime.fromisoformat(created)])

    def restore_page(self, page_id: str, name: str, source: str, table_name: str, src_table: str,
                     created_at: str | None = None) -> DatasetInfo:
        """Rebuild an imported page from its immutable source snapshot (journal replay)."""
        with self._lock:
            if not self._con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = ?",
                                     [src_table]).fetchall():
                raise FileNotFoundError(f"The source snapshot '{src_table}' of page '{name}' is gone (compacted?): "
                                        "replay needs the original file again")
            self._con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM "{src_table}"')
            sample = self._con.execute(f'SELECT * FROM "{table_name}" LIMIT 200000').df()
            self._insert_catalog(page_id, name, source, table_name, sample, src_table=src_table,
                                 created_at=datetime.fromisoformat(created_at) if created_at else None)
        return self.get(page_id)

    def wipe_pages(self) -> None:
        """Drop every page table and catalog row but keep the ``src_`` snapshots (journal replay)."""
        for info in self.list():
            self.delete(info.id, keep_source=True)

    # ---- import ------------------------------------------------------------------
    def new_page_id(self) -> str:
        """A fresh page id; a journal replay hands back the ids the original run produced."""
        rec = current_recording()
        return rec.page_ids.pop(0) if rec and rec.page_ids else uuid.uuid4().hex[:8]

    def add_dataframe(self, df: pd.DataFrame, name: str, source: str) -> DatasetInfo:
        dataset_id = self.new_page_id()
        table = f"ds_{_slug(name)}_{dataset_id}"
        with self._lock:
            self._replace_table(table, df)
            self._insert_catalog(dataset_id, name, source, table, df)
        return self.get(dataset_id)

    def add_derived(self, df: pd.DataFrame, name: str, source: str, like: DatasetInfo) -> DatasetInfo:
        """New page computed from another one (SQL, transforms), keeping the parent's structure."""
        date_col = like.date_col
        if like.dated and (not date_col or date_col not in df.columns) and st.OBS in df.columns:
            # The parent was dated by a start period, not a column: keep its dates as a visible column.
            date_col = "date" if "date" not in df.columns else "obs_date"
            df = df.rename(columns={st.OBS: date_col})
        df = df.drop(columns=[c for c in st.HIDDEN if c in df.columns])
        new = self.add_dataframe(df, name, source)
        if like.kind == "cross_section":
            return new
        try:
            if like.kind == "panel_undated":
                return self.set_structure(new.id, like.kind, id_col=like.id_col, t_col=like.t_col)
            return self.set_structure(new.id, like.kind, freq=like.freq, date_col=date_col, id_col=like.id_col)
        except ValueError as exc:
            log.info("Derived page %s kept as cross-section: %s", name, exc)
            return new

    def import_file(self, path: str | Path, name: str | None = None, sheet: str | int | None = None) -> DatasetInfo:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        fmt = FILE_FORMATS.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {sorted(FILE_FORMATS)}")
        name = name or path.stem
        if fmt == "excel":
            df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
            return self.add_dataframe(df, name, f"excel:{path.name}")
        dataset_id = self.new_page_id()
        table = f"ds_{_slug(name)}_{dataset_id}"
        reader = "read_csv_auto" if fmt == "csv" else "read_parquet"
        with self._lock:
            self._con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM {reader}(?)', [str(path)])
            sample = self._con.execute(f'SELECT * FROM "{table}" LIMIT 200000').df()
            self._insert_catalog(dataset_id, name, f"{fmt}:{path.name}", table, sample)
        return self.get(dataset_id)

    def sql(self, query: str, limit: int = 1000) -> pd.DataFrame:
        """Read-only SQL over the workspace; each dataset is exposed as a view named after it."""
        body = query.strip().rstrip(";")
        if not re.match(r"^(select|with)\b", body, re.IGNORECASE) or ";" in body:
            raise ValueError("Only a single SELECT/WITH query is allowed")
        with self._lock:
            views = {}
            for info in self.list():
                views[_slug(info.name)] = info
                self._con.execute(
                    f'CREATE OR REPLACE TEMP VIEW "{_slug(info.name)}" AS SELECT * FROM "{info.table_name}"'
                )
            try:
                return self._con.execute(f"SELECT * FROM ({body}) LIMIT {int(limit)}").df()
            except duckdb.Error as exc:
                # Give the caller what it needs to fix the query in one more attempt: the exact view names and columns.
                catalog = "; ".join(f"{v} ({', '.join(str(c['name']) for c in (i.columns or [])[:40])})"
                                    for v, i in views.items()) or "(no datasets)"
                raise ValueError(f"{type(exc).__name__}: {exc}\nAvailable views and columns (use these exact names, "
                                 f"quoted if they contain unusual characters): {catalog}") from exc

    def _insert_catalog(self, dataset_id: str, name: str, source: str, table: str, sample: pd.DataFrame, *,
                        src_table: str | None = None, created_at: datetime | None = None) -> None:
        existing = {r[0] for r in self._con.execute("SELECT name FROM _qtrix_pages").fetchall()}
        unique, i = name, 2
        while unique in existing:
            unique, i = f"{name}_{i}", i + 1
        try:
            suggestion = st.suggest(sample)
        except Exception as exc:  # noqa: BLE001 - a failed guess must never block an import
            log.warning("Structure detection failed for %s: %s", name, exc)
            suggestion = None
        n = len(sample)
        if src_table is None:  # immutable snapshot of what was imported: the page is derived from it (ADR §1)
            src_table = f"src_{dataset_id}"
            self._con.execute(f'CREATE OR REPLACE TABLE "{src_table}" AS SELECT * FROM "{table}"')
        self._con.execute(
            "INSERT INTO _qtrix_pages (id, name, source, table_name, created_at, start_label, end_label, suggestion, "
            "src_table) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [dataset_id, unique, source, table, created_at or datetime.now(UTC), "1" if n else None,
             str(n) if n else None, json.dumps(suggestion.to_dict()) if suggestion else None, src_table],
        )
