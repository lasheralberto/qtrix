"""Declarative object specs (ADR 0002 §3): everything needed to re-estimate an object deterministically.

One Pydantic model per method, joined in a union discriminated by ``method``. Specs are plain data: they are
stored as JSON, diffed between versions and (later) printed as Python by the ``qx`` API.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ObjectType = Literal["equation", "var", "group", "table", "graph", "text"]
DEFAULT_PREFIX: dict[str, str] = {"equation": "eq", "var": "var", "group": "gr", "table": "tab", "graph": "gph",
                                  "text": "txt"}


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ClassVar[ObjectType]

    def refs(self) -> list[str]:
        """Names of other objects this one is built from (a change in them makes it stale)."""
        return []

    @property
    def page_name(self) -> str:
        return getattr(self, "page", "") or ""

    def dump(self) -> dict[str, Any]:
        return {"type": self.type, **self.model_dump()}


class _Fitted(_Spec):
    """Estimated on the rows of a page: ``sample`` is the resolved sample spec ('@all' when unrestricted)."""

    page: str
    sample: str = "@all"


# ---- equations -------------------------------------------------------------------------------------------
class _Equation(_Fitted):
    type: ClassVar[ObjectType] = "equation"
    formula: str


class LsSpec(_Equation):
    method: Literal["ls"] = "ls"
    cov_type: Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC"] = "HC1"
    cluster: str | None = None
    hac_lags: int = 4


class WlsSpec(LsSpec):
    method: Literal["wls"] = "wls"  # type: ignore[assignment]
    weights: str


class GlmSpec(_Equation):
    method: Literal["glm"] = "glm"
    family: Literal["gaussian", "binomial", "poisson", "gamma"] = "gaussian"


class BinarySpec(_Equation):
    method: Literal["binary"] = "binary"
    link: Literal["logit", "probit"] = "logit"
    marginal_effects: bool = True


class CountSpec(_Equation):
    method: Literal["count"] = "count"
    model: Literal["poisson", "negbin", "zip", "zinb"] = "poisson"
    cov_type: Literal["nonrobust", "HC1", "cluster"] = "HC1"
    cluster: str | None = None


class OrderedSpec(_Equation):
    method: Literal["ordered"] = "ordered"
    link: Literal["logit", "probit"] = "logit"


class MultinomialSpec(_Equation):
    method: Literal["multinomial"] = "multinomial"


class QuantileSpec(_Equation):
    method: Literal["quantile"] = "quantile"
    quantile: float = 0.5
    vcov: Literal["robust", "iid"] = "robust"


class _Iv(_Equation):
    cov_type: Literal["unadjusted", "robust", "clustered", "kernel"] = "robust"
    cluster: str | None = None


class TslsSpec(_Iv):
    method: Literal["tsls"] = "tsls"


class LimlSpec(_Iv):
    method: Literal["liml"] = "liml"


class GmmSpec(_Iv):
    method: Literal["gmm"] = "gmm"


class PanelSpec(_Equation):
    method: Literal["panel"] = "panel"
    estimator: Literal["fe", "twfe", "re", "between", "fd", "pooled"] = "fe"
    cov_type: Literal["unadjusted", "robust", "cluster_entity", "cluster_both", "kernel"] = "cluster_entity"
    entity: str | None = None
    time: str | None = None


class ArimaSpec(_Fitted):
    type: ClassVar[ObjectType] = "equation"
    method: Literal["arima"] = "arima"
    column: str
    order: list[int]
    seasonal_order: list[int] | None = None
    exog: list[str] | None = None
    forecast_steps: int = 0


class ArdlSpec(_Fitted):
    type: ClassVar[ObjectType] = "equation"
    method: Literal["ardl"] = "ardl"
    column: str
    exog: list[str] = Field(default_factory=list)
    lags: int | None = None  # None = chosen by information criterion up to max_lag
    exog_lags: int | None = None
    max_lag: int = 4
    trend: Literal["n", "c", "ct"] = "c"
    ic: Literal["aic", "bic"] = "aic"


class MarkovSpec(_Fitted):
    type: ClassVar[ObjectType] = "equation"
    method: Literal["markov"] = "markov"
    column: str
    k_regimes: int = 2
    order: int = 0  # autoregressive order; 0 = Markov-switching mean (MarkovRegression)
    switching_variance: bool = True


class GarchSpec(_Fitted):
    type: ClassVar[ObjectType] = "equation"
    method: Literal["garch"] = "garch"
    column: str
    model: Literal["garch", "egarch", "gjr"] = "garch"
    p: int = 1
    q: int = 1
    o: int = 0  # asymmetry order (forced to 1 for gjr when 0)
    mean: Literal["constant", "zero", "ar"] = "constant"
    dist: Literal["normal", "t", "skewt", "ged"] = "normal"
    forecast_steps: int = 0


class PythonSpec(_Spec):
    """A result computed by a Python cell in the project kernel and registered with ``qx.register`` (E7·07b).

    The cell is the spec: the object is rebuilt by re-running it in a fresh kernel, where ``qx.register`` of
    ``registered_as`` (the name the cell uses, kept across renames) hands back the summary.
    """

    type: ClassVar[ObjectType] = "equation"
    method: Literal["python"] = "python"
    page: str = ""
    code: str
    registered_as: str
    title: str = ""


# ---- other types -----------------------------------------------------------------------------------------
class VarSpec(_Fitted):
    type: ClassVar[ObjectType] = "var"
    method: Literal["var"] = "var"
    columns: list[str]
    maxlags: int = 8
    irf_periods: int = 10


class VecmSpec(_Fitted):
    type: ClassVar[ObjectType] = "var"
    method: Literal["vecm"] = "vecm"
    columns: list[str]
    coint_rank: int | None = None  # None = chosen with the Johansen trace test (5%)
    k_ar_diff: int = 2
    deterministic: Literal["n", "co", "ci", "lo", "li"] = "co"
    irf_periods: int = 10


class GroupSpec(_Spec):
    type: ClassVar[ObjectType] = "group"
    method: Literal["group"] = "group"
    page: str = ""
    members: list[str]


class TableSpec(_Spec):
    """A comparison of other objects' summaries; it goes stale when any of them gets a new version."""

    type: ClassVar[ObjectType] = "table"
    method: Literal["table"] = "table"
    sources: list[str]
    title: str = "Model comparison"

    def refs(self) -> list[str]:
        return list(self.sources)


class GraphSpec(_Spec):
    type: ClassVar[ObjectType] = "graph"
    method: Literal["graph"] = "graph"
    title: str
    figure: dict[str, Any]  # Plotly JSON (frozen)
    source: str | None = None

    def refs(self) -> list[str]:
        return [self.source] if self.source else []


class FrozenTableSpec(_Spec):
    """A view frozen as a table object (``freeze``): a snapshot that keeps its content whatever its source does."""

    type: ClassVar[ObjectType] = "table"
    method: Literal["frozen_table"] = "frozen_table"
    title: str
    kind: str  # block kind of the view: regression_table, diagnostics, dataframe, markdown
    data: dict[str, Any]
    source: str | None = None  # provenance only: a frozen view does not follow (or go stale with) its source
    view: str | None = None


class FrozenGraphSpec(_Spec):
    """A chart view frozen as a graph object (``freeze``)."""

    type: ClassVar[ObjectType] = "graph"
    method: Literal["frozen_graph"] = "frozen_graph"
    title: str
    figure: dict[str, Any]  # Plotly JSON
    source: str | None = None
    view: str | None = None


class TextSpec(_Spec):
    type: ClassVar[ObjectType] = "text"
    method: Literal["text"] = "text"
    title: str = ""
    markdown: str


AnySpec = Annotated[
    LsSpec | WlsSpec | GlmSpec | BinarySpec | CountSpec | OrderedSpec | MultinomialSpec | QuantileSpec
    | TslsSpec | LimlSpec | GmmSpec | PanelSpec | ArimaSpec | ArdlSpec | MarkovSpec
    | GarchSpec | PythonSpec | VarSpec | VecmSpec | GroupSpec | TableSpec | GraphSpec | TextSpec | FrozenTableSpec
    | FrozenGraphSpec,
    Field(discriminator="method"),
]
_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnySpec)


def parse_spec(data: dict[str, Any]) -> _Spec:
    """Spec from its stored JSON (the ``type`` key is informational)."""
    return _ADAPTER.validate_python({k: v for k, v in data.items() if k != "type"})


def with_changes(spec: _Spec, **changes: Any) -> _Spec:
    """A new spec of the same method with some fields replaced (re-estimation with another sample or option)."""
    return parse_spec({**spec.model_dump(), **changes})
