"""Expression language (ADR 0002 §3): EViews-style time operators inside ordinary formulas.

Formulas stay R/patsy style strings, extended with::

    y(-1)  y(-4)  y(+1)        lag / lead of a series (only when y is a column of the dataset)
    L(y, 2)  d(y)  d(y, 2)     lag; difference (1-L)^n; d(y, 1, 4) also takes a seasonal difference
    dlog(y)  pch(y)  log(y)    log difference; percent change; also exp, sqrt, abs, ln
    movav(y, 4)  movsum(y, 4)  moving average / sum of the last n observations
    @trend  @seas(2)           time trend (0 at the first period); seasonal dummy for quarter / month 2
    dummy("2008q3")            1 in a period (dummy("2008q3", "2009q2") for a range)
    recode(x > 3, 1, 0)        element-wise if / else

Terms that use these operators are evaluated here, on the *whole* page, and handed to the estimation
libraries as plain columns. That gives the semantics of EViews: lags are taken within each entity of a
panel, follow the calendar of the dataset, and may look *outside* the estimation sample (the sample
selects which observations are used, not which values exist). The sample is applied afterwards by
blanking the dependent variable outside it, so those rows drop out of the fit.

This is not a sandbox: expressions are Python, evaluated with a small set of names, and a deny-list
catches obvious misuse.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .workfile import structure as st

PREFIX = "_qx_"
LHS = "qx_lhs"
TERM = "qx_t"

HELP: dict[str, str] = {
    "L(x, k=1)": "lag k periods (x(-k) is the same; x(+k) is a lead); within each entity in panels",
    "d(x, n=1, s=0)": "difference: (1-L)^n, and (1-L^s) too when s > 0",
    "dlog(x, n=1, s=0)": "difference of the log",
    "pch(x, k=1)": "percent change over k periods: x / L(x, k) - 1",
    "movav(x, n) / movsum(x, n)": "moving average / sum of the last n observations",
    "log(x) ln(x) exp(x) sqrt(x) abs(x)": "element-wise functions",
    "@trend / trend(base=None)": "time trend, 0 at the first period (or at the period base)",
    "@seas(q) / seas(q)": "dummy for quarter (quarterly data) or month (monthly data) q",
    "dummy(start, end=None)": "1 from start to end (a single period when end is omitted), else 0",
    "recode(cond, a, b)": "a where cond holds, b elsewhere",
}


class ExprError(ValueError):
    """A formula or expression that cannot be evaluated; the message says how to fix it."""


@dataclass(frozen=True)
class Context:
    """What the operators need to know about the page they run on."""

    kind: str
    freq: str
    index: pd.Index
    columns: tuple[str, ...] = field(default_factory=tuple)


# ---- operators ------------------------------------------------------------------------

def _need_time(ctx: Context, name: str) -> None:
    if ctx.kind == "cross_section":
        raise ExprError(
            f"{name}() needs time order, but this dataset is a cross-section (rows have no time). If its rows "
            "are dated, declare it with data_set_structure(kind='time_series' or 'panel', ...); otherwise use "
            "operators that do not need time (log, exp, sqrt, abs, recode, I(x**2))")


def _need_dates(ctx: Context, name: str) -> pd.DatetimeIndex:
    if ctx.kind == "time_series":
        return pd.DatetimeIndex(ctx.index)
    if ctx.kind == "panel":
        return pd.DatetimeIndex(ctx.index.get_level_values(1))
    raise ExprError(f"{name}() needs dates, but this dataset is {st.describe(ctx.kind, ctx.freq, None, None)}. "
                    "Declare dates with data_set_structure(kind='time_series' or 'panel', freq=..., date_col=...)")


def _series(x, ctx: Context, name: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    arr = np.asarray(x)
    if arr.ndim == 0:
        return pd.Series(arr.item(), index=ctx.index)
    if len(arr) != len(ctx.index):
        raise ExprError(f"{name}() received {len(arr)} values for {len(ctx.index)} rows")
    return pd.Series(arr, index=ctx.index)


def _lag(s: pd.Series, k: int, ctx: Context, name: str) -> pd.Series:
    _need_time(ctx, name)
    k = int(k)
    if k == 0:
        return s
    if ctx.kind == "time_series":
        return s.shift(k)
    if ctx.kind == "panel":
        return s.groupby(level=0, sort=False).shift(k)
    # Undated panel: move along the value of the time index, so a missing wave stays missing.
    ids, t = s.index.get_level_values(0), s.index.get_level_values(1)
    try:
        key = pd.MultiIndex.from_arrays([ids, t - k])
    except TypeError:
        raise ExprError(f"{name}() needs an integer time index, found {t.dtype}") from None
    return pd.Series(s.reindex(key).to_numpy(), index=s.index)


def _by_entity(s: pd.Series, ctx: Context, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
    if ctx.kind in st.PANEL_KINDS:
        return s.groupby(level=0, sort=False, group_keys=False).apply(fn).reindex(s.index)
    return fn(s)


def operators(ctx: Context) -> dict[str, Callable]:
    """The operator functions bound to a page, keyed by their public name."""

    def L(x, k=1):  # noqa: N802 - EViews / R spelling
        return _lag(_series(x, ctx, "L"), k, ctx, "L")

    def d(x, n=1, s=0):
        out = _series(x, ctx, "d")
        for _ in range(int(n)):
            out = out - _lag(out, 1, ctx, "d")
        return out - _lag(out, int(s), ctx, "d") if s else out

    def log(x):
        with np.errstate(all="ignore"):
            return np.log(_series(x, ctx, "log"))

    def dlog(x, n=1, s=0):
        return d(log(x), n, s)

    def pch(x, k=1):
        s = _series(x, ctx, "pch")
        return s / _lag(s, k, ctx, "pch") - 1

    def movav(x, n):
        return _by_entity(_series(x, ctx, "movav"), ctx, lambda g: g.rolling(int(n)).mean())

    def movsum(x, n):
        return _by_entity(_series(x, ctx, "movsum"), ctx, lambda g: g.rolling(int(n)).sum())

    def exp(x):
        with np.errstate(all="ignore"):
            return np.exp(_series(x, ctx, "exp"))

    def sqrt(x):
        with np.errstate(all="ignore"):
            return np.sqrt(_series(x, ctx, "sqrt"))

    def abs_(x):
        return _series(x, ctx, "abs").abs()

    def trend(base=None):
        _need_time(ctx, "trend")
        n = len(ctx.index)
        t = (np.arange(n) if ctx.kind == "time_series"
             else pd.Series(0, index=ctx.index).groupby(level=0, sort=False).cumcount().to_numpy())
        if base is not None:
            obs = _need_dates(ctx, "trend")
            first = st.parse_period(str(base), ctx.freq)
            calendar = pd.DatetimeIndex(obs.unique().sort_values())
            if first not in calendar:
                raise ExprError(f"trend('{base}'): that period is outside the data "
                                f"({st.label(calendar[0], ctx.freq)}–{st.label(calendar[-1], ctx.freq)})")
            t = t - calendar.get_loc(first)
        return pd.Series(t, index=ctx.index)

    def seas(q):
        obs = _need_dates(ctx, "seas")
        per = {"Q": 3, "M": 1, "S": 6}.get(ctx.freq)
        if per is None:
            raise ExprError(f"seas() works on semiannual, quarterly and monthly data, not {st.FREQ_NAMES[ctx.freq]}")
        season = (obs.month - 1) // per + 1
        if not 1 <= int(q) <= 12 // per:
            raise ExprError(f"seas({q}): {st.FREQ_NAMES[ctx.freq]} data have seasons 1 to {12 // per}")
        return pd.Series((season == int(q)).astype(int), index=ctx.index)

    def dummy(start, end=None):
        obs = _need_dates(ctx, "dummy")
        lo = st.parse_period(str(start), ctx.freq)
        hi = st.parse_period(str(end), ctx.freq) if end is not None else lo
        return pd.Series(((obs >= lo) & (obs <= hi)).astype(int), index=ctx.index)

    def recode(cond, a, b):
        values = np.where(np.asarray(cond, dtype=bool), a, b)
        return pd.Series(values, index=cond.index) if isinstance(cond, pd.Series) else values

    return {"L": L, "d": d, "dlog": dlog, "pch": pch, "movav": movav, "movsum": movsum, "log": log, "ln": log,
            "exp": exp, "sqrt": sqrt, "abs": abs_, "trend": trend, "seas": seas, "dummy": dummy, "recode": recode}


FUNCTIONS = tuple(operators(Context("cross_section", "U", pd.RangeIndex(0))))
_BY_LOWER = {name.lower(): name for name in FUNCTIONS}

# ---- translation ------------------------------------------------------------------------

_AT = re.compile(r"@([A-Za-z_]\w*)(\s*\()?")
_CALL = re.compile(r"(?<![\w.])(" + "|".join(sorted(FUNCTIONS, key=len, reverse=True)) + r")\s*\(")
_FORBIDDEN = re.compile(r"__|\bimport\b|\bexec\b|\beval\b|\bopen\s*\(|\bcompile\b|\bgetattr\b|\bglobals\b|"
                        r"\bos\.|\bsys\.|subprocess|\blambda\b")
_BUILTINS = {"min": min, "max": max, "len": len, "round": round, "float": float, "int": int, "sum": sum}


def check_safe(text: str) -> None:
    if _FORBIDDEN.search(text):
        raise ExprError(f"'{text}' uses something that is not allowed in formulas (imports, dunder names, "
                        "open/exec/eval…). Use the column names and the operators listed in the tool help")


def translate(piece: str, columns: Iterable[str]) -> str:
    """EViews syntax → Python code: ``y(-1)`` → ``_qx_L(y, 1)``, ``@seas(2)`` → ``_qx_seas(2)``."""
    check_safe(piece)

    def at(m: re.Match[str]) -> str:
        name = _BY_LOWER.get(m.group(1).lower())
        if name is None:
            raise ExprError(f"Unknown function @{m.group(1)}. Available: "
                            + ", ".join(f"@{n}" for n in ("trend", "seas", "dummy", "movav", "recode")))
        return name + ("(" if m.group(2) else "()")

    code = _AT.sub(at, piece)
    names = sorted((c for c in columns if isinstance(c, str) and c.isidentifier()), key=len, reverse=True)
    if names:
        lag = re.compile(r"(?<![\w.'\"])(" + "|".join(re.escape(n) for n in names)
                         + r")\s*\(\s*([+-]?)\s*(\d+)\s*\)")

        def shift(m: re.Match[str]) -> str:  # x(-k) is a lag of k periods, x(+k) and x(k) are leads
            k = int(m.group(3))
            return f"{PREFIX}L({m.group(1)}, {k if m.group(2) == '-' else -k})"

        code = lag.sub(shift, code)
    return _CALL.sub(lambda m: PREFIX + m.group(1) + "(", code)


def _segments(text: str) -> list[tuple[bool, str]]:
    """Split a formula side into (is_piece, text): pieces are the factors between the formula operators
    (+ - * / : | ~ and the [ ] of IV formulas) found outside any parenthesis or quote."""
    out: list[tuple[bool, str]] = []
    buf: list[str] = []
    stack: list[str] = []
    quote: str | None = None

    def flush() -> None:
        if buf:
            out.append((True, "".join(buf)))
            buf.clear()

    for ch in text:
        if quote:
            buf.append(ch)
            quote = None if ch == quote else quote
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in "({" or (ch == "[" and stack):
            stack.append(ch)
            buf.append(ch)
        elif ch in ")}" or (ch == "]" and stack):
            if stack:
                stack.pop()
            buf.append(ch)
        elif not stack and ch in "+-*/:|~[]":
            flush()
            out.append((False, ch))
        else:
            buf.append(ch)
    flush()
    return out


def _first_tilde(text: str) -> int:
    depth, quote = 0, None
    for i, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "~" and depth == 0:
            return i
    return -1


# ---- evaluation ---------------------------------------------------------------------------

def _namespace(frame: pd.DataFrame, ctx: Context) -> dict:
    ns: dict = {c: frame[c] for c in frame.columns if isinstance(c, str) and c.isidentifier()}
    ns.update({PREFIX + name: fn for name, fn in operators(ctx).items()})
    ns.update(np=np, I=lambda x: x, Q=lambda name: frame[name])
    return ns


def _evaluate(code: str, original: str, ns: dict, columns: Iterable[str]):
    try:
        return eval(code, {"__builtins__": _BUILTINS}, ns)  # noqa: S307 - guarded by check_safe, see module doc
    except ExprError:
        raise
    except NameError as exc:
        shown = ", ".join(c for c in columns if isinstance(c, str) and not c.startswith(("__", TERM, LHS)))
        raise ExprError(f"Unknown variable '{exc.name}' in '{original}'. Columns: {shown}") from exc
    except SyntaxError as exc:
        raise ExprError(f"Cannot read '{original}' as an expression ({exc.msg})") from exc
    except Exception as exc:  # noqa: BLE001 - surface any library error with the offending term
        raise ExprError(f"Cannot evaluate '{original}': {type(exc).__name__}: {exc}") from exc


def _as_series(value, ctx: Context, original: str) -> pd.Series:
    s = _series(value, ctx, original)
    if s.dtype == bool:
        s = s.astype(int)
    if not pd.api.types.is_numeric_dtype(s):
        raise ExprError(f"'{original}' is not numeric (got {s.dtype}); use C() for categories")
    return s


def evaluate(expression: str, frame: pd.DataFrame, ctx: Context) -> pd.Series:
    """Evaluate a single expression (used by genr) over every row of the page."""
    code = translate(expression, frame.columns)
    value = _evaluate(code, expression, _namespace(frame, ctx), frame.columns)
    return _as_series(value, ctx, expression)


@dataclass
class Compiled:
    """A formula ready for the estimation libraries: plain columns plus a rewritten formula string."""

    formula: str
    frame: pd.DataFrame
    names: dict[str, str]  # generated column -> the term as the user wrote it
    original: str

    def pretty(self, name: str) -> str:
        return pretty(name, self.names)


def pretty(name: str, names: Mapping[str, str]) -> str:
    """Show generated column names as the user wrote the terms (``qx_t1`` → ``dlog(gdp(-1))``)."""
    for key in sorted(names, key=len, reverse=True):
        name = name.replace(key, names[key])
    return name


def compile_formula(formula: str, frame: pd.DataFrame, ctx: Context, *, mask: np.ndarray | None = None) -> Compiled:
    """Evaluate the time operators of ``formula`` over the full page and blank the dependent variable
    outside the sample (``mask``, aligned with the rows of ``frame``)."""
    check_safe(formula)
    cut = _first_tilde(formula)
    if cut < 0:
        raise ExprError(f"'{formula}' is not a formula: write 'dependent ~ regressors', e.g. dlog(gdp) ~ d(r)")
    lhs_text, rhs_text = formula[:cut], formula[cut + 1:]
    columns = tuple(frame.columns)
    ns: dict | None = None
    out = frame
    names: dict[str, str] = {}
    cache: dict[str, str] = {}

    def lazy() -> dict:
        nonlocal ns
        if ns is None:
            ns = _namespace(frame, ctx)
        return ns

    def own(frame_: pd.DataFrame) -> pd.DataFrame:
        return frame_ if frame_ is not frame else frame_.copy()

    lhs = lhs_text.strip()
    if not lhs:
        raise ExprError("The formula has no dependent variable before '~'")
    code = translate(lhs, columns)
    if PREFIX in code or mask is not None:
        series = _as_series(_evaluate(code, lhs, lazy(), columns), ctx, lhs)
        if mask is not None:
            series = series.where(pd.Series(mask, index=frame.index))
        out = own(out)
        out[LHS] = series
        names[LHS] = lhs
        lhs_out = LHS
    else:
        lhs_out = lhs_text

    pieces: list[str] = []
    for is_piece, text in _segments(rhs_text):
        core = text.strip()
        if not is_piece or not core:
            pieces.append(text)
            continue
        code = translate(core, columns)
        if PREFIX not in code:
            pieces.append(text if code == core else text.replace(core, code))
            continue
        if core not in cache:
            col = f"{TERM}{len(cache) + 1}"
            value = _evaluate(code, core, lazy(), columns)
            out = own(out)
            out[col] = _as_series(value, ctx, core)
            names[col] = core
            cache[core] = col
        pieces.append(text.replace(core, cache[core]))
    return Compiled(f"{lhs_out} ~ {''.join(pieces).strip()}", out, names, formula)


# ---- error translation -------------------------------------------------------------------------

@contextmanager
def readable_errors(columns: Iterable[str] = ()):
    """Turn formula-engine errors (patsy, formulaic) into ExprError with a message an agent can act on."""
    try:
        yield
    except ExprError:
        raise
    except Exception as exc:
        if type(exc).__module__.split(".")[0] not in {"patsy", "formulaic"}:
            raise
        cause = exc.__cause__ or exc.__context__ or exc
        if isinstance(cause, NameError):
            shown = ", ".join(c for c in columns if isinstance(c, str) and not c.startswith(("__", TERM, LHS)))
            raise ExprError(f"Unknown variable '{cause.name}' in the formula. Columns: {shown}") from exc
        raise ExprError(f"Cannot evaluate the formula: {str(cause).splitlines()[0]}") from exc
