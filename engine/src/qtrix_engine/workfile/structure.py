"""Page structure: what a dataset *is* (cross-section, time series, panel) and how dated pages are indexed.

Nothing here assumes time. A page is a cross-section unless its structure is declared; import only
*suggests* a structure when the evidence is strong (ADR 0002 §1). Dated pages are regularised: one row per
period (or entity × period) over the full range, gaps filled with NA, plus hidden canonical columns
``__obs`` (period start) and ``__id`` (entity) from which every tool builds its index.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

PageKind = Literal["cross_section", "time_series", "panel", "panel_undated"]
Freq = Literal["U", "A", "S", "Q", "M", "W", "D5", "D7"]

KINDS: tuple[str, ...] = ("cross_section", "time_series", "panel", "panel_undated")
DATED_KINDS = frozenset({"time_series", "panel"})
PANEL_KINDS = frozenset({"panel", "panel_undated"})
DATED_FREQS = ("A", "S", "Q", "M", "W", "D5", "D7")
FREQ_NAMES = {"U": "undated", "A": "annual", "S": "semiannual", "Q": "quarterly", "M": "monthly",
              "W": "weekly", "D5": "daily (5-day)", "D7": "daily (7-day)"}
OBS, ID = "__obs", "__id"
HIDDEN = (OBS, ID)

_TIME_NAME = re.compile(r"^(year|yr|anio|año|ano|fecha|date|dates|period|periodo|time|tiempo|quarter|"
                        r"trimestre|month|mes|week|semana|day|dia|día|obs|t)$|(_date|_year|_fecha)$", re.I)
_ID_NAME = re.compile(r"(^|_)(id|code|cod|codigo|iso|iso3|country|pais|país|firm|empresa|company|entity|"
                      r"region|state|estado|province|provincia|ticker|isin|cusip|permno|gvkey|person|hh|"
                      r"household|individual|school|county|city|municipio)($|_)", re.I)
_WEEKDAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


class StructureError(ValueError):
    """Invalid structure declaration (duplicate periods, unknown frequency…); message is actionable."""


@dataclass
class Suggestion:
    kind: str
    freq: str
    date_col: str
    id_col: str | None
    start: str
    end: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---- periods ---------------------------------------------------------------------

def offset(freq: str, anchor: pd.Timestamp | None = None) -> str:
    """pandas offset alias for the period starts of a frequency."""
    if freq == "W":
        return f"W-{_WEEKDAYS[(anchor or pd.Timestamp('2000-01-03')).weekday()]}"
    try:
        return {"A": "YS", "S": "6MS", "Q": "QS", "M": "MS", "D5": "B", "D7": "D"}[freq]
    except KeyError:
        raise StructureError(f"Unknown frequency '{freq}'. Use one of {', '.join(DATED_FREQS)}") from None


def period_start(ts: pd.Series, freq: str, anchor: pd.Timestamp | None = None) -> pd.Series:
    """Snap timestamps to the start of the period that contains them."""
    ts = pd.to_datetime(ts).dt.normalize()
    if freq == "A":
        return pd.to_datetime(pd.DataFrame({"year": ts.dt.year, "month": 1, "day": 1}), errors="coerce")
    if freq in {"S", "Q", "M"}:
        months = {"S": 6, "Q": 3, "M": 1}[freq]
        month = ((ts.dt.month - 1) // months) * months + 1
        return pd.to_datetime(pd.DataFrame({"year": ts.dt.year, "month": month, "day": 1}), errors="coerce")
    if freq == "W":
        anchor_day = (anchor if anchor is not None else ts.dropna().min()).weekday()
        return ts - pd.to_timedelta((ts.dt.weekday - anchor_day) % 7, unit="D")
    return ts


def label(ts: pd.Timestamp, freq: str) -> str:
    """Human/EViews-style label of a period start: 1990, 1990S1, 1990Q1, 1990M03, 1990-01-15."""
    if freq == "A":
        return f"{ts.year}"
    if freq == "S":
        return f"{ts.year}S{1 if ts.month <= 6 else 2}"
    if freq == "Q":
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    if freq == "M":
        return f"{ts.year}M{ts.month:02d}"
    return ts.strftime("%Y-%m-%d")


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Q", re.compile(r"^(?P<y>\d{4})\s*[-:/ ]?\s*[Qq](?P<p>[1-4])$")),
    ("Q", re.compile(r"^[Qq](?P<p>[1-4])\s*[-:/ ]?\s*(?P<y>\d{4})$")),
    ("S", re.compile(r"^(?P<y>\d{4})\s*[-:/ ]?\s*[SsHh](?P<p>[12])$")),
    ("M", re.compile(r"^(?P<y>\d{4})\s*[Mm](?P<p>\d{1,2})$")),
    ("A", re.compile(r"^(?P<y>\d{4})$")),
)


def _parse_tagged(values: pd.Series) -> tuple[pd.Series, str] | None:
    """Parse '1990Q1', 'Q1 1990', '1990S2', '1990M03', '1990' style labels (all values must match)."""
    if values.empty:
        return None
    text = values.astype(str).str.strip()
    for freq, pattern in _PATTERNS:
        parts = text.str.extract(pattern)
        if not parts["y"].notna().all():
            continue
        year = parts["y"].astype(int)
        p = parts["p"].astype(int) if "p" in parts else pd.Series(1, index=parts.index)
        if freq == "M" and not p.between(1, 12).all():
            continue
        month = {"Q": (p - 1) * 3 + 1, "S": (p - 1) * 6 + 1, "M": p, "A": 1}[freq]
        out = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": 1}), errors="coerce")
        return pd.Series(out.to_numpy(), index=values.index), freq
    return None


def parse_period(text: str, freq: str) -> pd.Timestamp:
    """Parse a period label typed by a user or agent ('1990q1', '1990', '1990m3', '2001-05-07')."""
    tagged = _parse_tagged(pd.Series([str(text)]))
    ts = tagged[0].iloc[0] if tagged is not None else pd.to_datetime(str(text), errors="coerce")
    if pd.isna(ts):
        raise StructureError(f"Cannot read '{text}' as a period (examples: 1990, 1990Q1, 1990M03, 1990-01-15)")
    return period_start(pd.Series([ts]), freq).iloc[0]


def as_dates(values: pd.Series, name: str = "") -> tuple[pd.Series, str | None] | None:
    """Interpret a column as dates. Returns (timestamps, freq hint) or None when it is not a date column.

    Integer columns only count as years when their *name* says so (``year``, ``anio``, ``fecha``…), so a
    survey's numeric variables are never mistaken for time.
    """
    s = values.dropna()
    if s.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values), None
    if pd.api.types.is_bool_dtype(values):
        return None
    if pd.api.types.is_numeric_dtype(values):
        if not _TIME_NAME.search(name or ""):
            return None
        if not (np.mod(s, 1) == 0).all() or not s.between(1800, 2200).all():
            return None
        years = pd.DataFrame({"year": values, "month": 1, "day": 1})
        return pd.to_datetime(years, errors="coerce"), "A"
    text = values.astype(str).where(values.notna())
    sample = text.dropna()
    tagged = _parse_tagged(sample)
    if tagged is not None:
        parsed, freq = tagged
        return parsed.reindex(values.index), freq
    if not sample.str.contains(r"\d", regex=True).all():
        return None
    for kwargs in ({"format": "ISO8601"}, {"format": "mixed", "dayfirst": True}):
        try:
            parsed = pd.to_datetime(text, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if parsed.notna().sum() >= 0.95 * len(sample):
            return parsed, None
    return None


def infer_freq(dates: pd.Series) -> str | None:
    """Guess the frequency of a set of dates from the spacing of their distinct values."""
    uniq = pd.Series(pd.to_datetime(dates.dropna()).unique()).sort_values()
    if len(uniq) < 3:
        if len(uniq) and (uniq.dt.month == 1).all() and (uniq.dt.day == 1).all():
            return "A"
        return None
    days = uniq.diff().dropna().dt.days
    step = float(days.median())
    for freq, low, high in (("A", 360, 370), ("S", 178, 186), ("Q", 88, 93), ("M", 27, 32), ("W", 7, 7)):
        if low <= step <= high:
            return freq
    if step == 1 or (step <= 3 and days.min() >= 1):
        return "D7" if (uniq.dt.weekday >= 5).any() else "D5"
    return None


# ---- suggestion -------------------------------------------------------------------

def suggest(df: pd.DataFrame, max_rows: int = 200_000) -> Suggestion | None:
    """Propose a structure only when the evidence is strong; otherwise the page stays a cross-section."""
    if len(df) < 3:
        return None
    frame = df if len(df) <= max_rows else df.head(max_rows)
    candidates: list[tuple[str, pd.Series, str]] = []
    for col in frame.columns:
        if col in HIDDEN:
            continue
        parsed = as_dates(frame[col], str(col))
        if parsed is None:
            continue
        dates, hint = parsed
        if dates.notna().mean() < 0.95:
            continue
        freq = hint or infer_freq(dates)
        if freq is None:
            continue
        candidates.append((str(col), dates, freq))
    # A column literally named like time wins over other date-like columns.
    candidates.sort(key=lambda c: 0 if _TIME_NAME.search(c[0]) else 1)
    for col, dates, freq in candidates:
        starts = period_start(dates, freq)
        bounds = (label(starts.min(), freq), label(starts.max(), freq))
        if starts.notna().all() and starts.is_unique:
            return Suggestion("time_series", freq, col, None, *bounds,
                              f"'{col}' holds one {FREQ_NAMES[freq]} date per row")
        id_col = _panel_id(frame, col, starts)
        if id_col is not None:
            n_ids = frame[id_col].nunique()
            return Suggestion("panel", freq, col, id_col, *bounds,
                              f"each ({id_col}, {col}) pair is unique: {n_ids} entities × "
                              f"{FREQ_NAMES[freq]} periods")
    return None


def _panel_id(df: pd.DataFrame, date_col: str, starts: pd.Series) -> str | None:
    n = len(df)
    options = []
    for col in df.columns:
        if col in (date_col, *HIDDEN):
            continue
        s = df[col]
        if pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s) or s.isna().any():
            continue
        k = s.nunique()
        if not 2 <= k <= n / 2:
            continue
        if pd.DataFrame({"i": s.to_numpy(), "t": starts.to_numpy()}).duplicated().any():
            continue
        options.append((0 if _ID_NAME.search(str(col)) else 1, k, str(col)))
    return min(options)[2] if options else None


# ---- regularisation -----------------------------------------------------------------

@dataclass
class Regularized:
    frame: pd.DataFrame
    start: str | None
    end: str | None
    gaps: int


def regularize(df: pd.DataFrame, *, kind: str, freq: str | None = None, date_col: str | None = None,
               id_col: str | None = None, t_col: str | None = None, start: str | None = None) -> Regularized:
    """Return the page table for a structure. Cross-sections are returned untouched."""
    df = df.drop(columns=[c for c in HIDDEN if c in df.columns and c != date_col])
    if kind not in KINDS:
        raise StructureError(f"Unknown structure '{kind}'. Use one of {', '.join(KINDS)}")
    n = len(df)
    if kind == "cross_section":
        return Regularized(df.reset_index(drop=True), "1" if n else None, str(n) if n else None, 0)
    if kind == "panel_undated":
        if not id_col or not t_col:
            raise StructureError("An undated panel needs id_col (entity) and t_col (integer time index)")
        if df[[id_col, t_col]].isna().any().any():
            raise StructureError(f"'{id_col}' and '{t_col}' must not have missing values")
        _check_unique(df, [id_col, t_col], f"({id_col}, {t_col})")
        out = df.sort_values([id_col, t_col], kind="stable").reset_index(drop=True)
        out.insert(0, ID, out[id_col])
        return Regularized(out, str(out[t_col].min()), str(out[t_col].max()), 0)

    if freq is None or freq == "U":
        raise StructureError(f"A {kind.replace('_', ' ')} needs a frequency: one of {', '.join(DATED_FREQS)}")
    if kind == "panel" and not id_col:
        raise StructureError("A dated panel needs id_col (the entity column) and date_col")
    if date_col:
        parsed = as_dates(df[date_col], "date" if date_col == OBS else date_col)
        if parsed is None:
            raise StructureError(f"Column '{date_col}' does not look like dates or periods")
        dates = parsed[0]
        if dates.isna().any():
            bad = df.loc[dates.isna(), date_col].head(3).tolist()
            raise StructureError(f"'{date_col}' has values that are not dates: {bad}")
    elif kind == "time_series" and start:
        first = parse_period(start, freq)
        dates = pd.Series(pd.date_range(first, periods=n, freq=offset(freq, first)), index=df.index)
    else:
        raise StructureError("Give date_col (a column with dates/periods) or start (first period, e.g. '1990Q1')")
    anchor = pd.Timestamp(dates.min()) if freq == "W" else None
    obs = period_start(dates, freq, anchor)
    if freq == "D5" and (obs.dt.weekday >= 5).any():
        raise StructureError("Weekend dates found: use freq='D7' for 7-day data")
    work = df.copy()
    work[OBS] = obs.to_numpy()
    if kind == "panel" and work[id_col].isna().any():
        raise StructureError(f"Entity column '{id_col}' has missing values")
    keys = [id_col, OBS] if kind == "panel" else [OBS]
    what = f"({id_col}, {date_col or 'period'})" if kind == "panel" else f"period of '{date_col or 'start'}'"
    _check_unique(work, keys, what)
    full_range = pd.date_range(obs.min(), obs.max(), freq=offset(freq, obs.min()))
    if not obs.isin(full_range).all():
        raise StructureError(f"Dates do not fit a {FREQ_NAMES[freq]} calendar; check the frequency")
    if kind == "panel":
        ids = pd.Index(work[id_col].drop_duplicates().sort_values(), name=id_col)
        target = pd.MultiIndex.from_product([ids, full_range], names=[id_col, OBS])
        out = work.set_index([id_col, OBS]).reindex(target).reset_index()
        out.insert(0, ID, out[id_col])
    else:
        out = work.set_index(OBS).reindex(pd.Index(full_range, name=OBS)).reset_index()
    gaps = len(out) - len(work)
    if date_col and date_col != OBS and gaps:
        missing = out[date_col].isna()
        filler = out.loc[missing, OBS]
        if pd.api.types.is_datetime64_any_dtype(out[date_col]):
            out.loc[missing, date_col] = filler
        elif pd.api.types.is_numeric_dtype(out[date_col]):
            out.loc[missing, date_col] = filler.dt.year
        else:
            out[date_col] = out[date_col].astype(object)
            out.loc[missing, date_col] = filler.map(lambda t: label(t, freq))
    cols = [OBS, *([ID] if kind == "panel" else []), *[c for c in df.columns if c != OBS]]
    return Regularized(out[cols], label(full_range[0], freq), label(full_range[-1], freq), gaps)


def _check_unique(df: pd.DataFrame, keys: list[str], what: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        shown = [k for k in keys if k != OBS] or keys
        sample = df.loc[dup, shown].drop_duplicates().head(3).astype(str).to_dict(orient="records")
        raise StructureError(f"{int(dup.sum())} rows share the same {what} (e.g. {sample}); each must identify "
                             "one row. Aggregate first or choose another structure")


def index_frame(df: pd.DataFrame, *, kind: str, freq: str, id_col: str | None, t_col: str | None) -> pd.DataFrame:
    """Turn a stored page table into the frame tools work on (hidden columns become the index)."""
    if kind == "time_series" and OBS in df.columns:
        idx = pd.DatetimeIndex(df[OBS], name="obs")
        out = df.drop(columns=[OBS]).set_index(idx)
        if len(idx) > 2:
            try:
                out.index.freq = offset(freq, idx[0])
            except ValueError:
                pass
        return out
    if kind == "panel" and OBS in df.columns:
        idx = pd.MultiIndex.from_arrays([df[ID], pd.DatetimeIndex(df[OBS])], names=[id_col, "obs"])
        return df.drop(columns=[OBS, ID]).set_index(idx)
    if kind == "panel_undated" and ID in df.columns:
        idx = pd.MultiIndex.from_arrays([df[ID], df[t_col]], names=[id_col, t_col])
        return df.drop(columns=[ID]).set_index(idx)
    return df.drop(columns=[c for c in HIDDEN if c in df.columns])


def describe(kind: str, freq: str, start: str | None, end: str | None, id_col: str | None = None,
             date_col: str | None = None, t_col: str | None = None) -> str:
    """One-line description used in the agent context and in error messages."""
    if kind == "cross_section":
        return "cross-section" + (f" (id: {id_col})" if id_col else "") + ", no dates"
    if kind == "panel_undated":
        return f"undated panel {id_col} × {t_col} ({start}–{end})"
    by = f" from '{date_col}'" if date_col and date_col != OBS else ""
    what = "time series" if kind == "time_series" else f"panel {id_col} ×"
    return f"{what} {FREQ_NAMES[freq]} {start}–{end}{by}"
