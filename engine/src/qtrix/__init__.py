"""qtrix: the Python API of a qtrix workfile (ADR 0002 §7.1).

    import qtrix as qx

    wages = qx.page("wages")
    wages.generate("lexper", "log(exper + 1)")
    eq_mincer = qx.equation("eq_mincer", page="wages", method="ls", formula="lwage ~ educ + lexper", cov_type="HC1")
    eq_mincer.estimate(sample="if union = 1")       # a new version of the same object
    qx.table("tab01", ["eq_mincer", "eq_iv"])

In a kernel cell, ``qx.page("wages").frame()`` reads data, ``qx.register("eq_q", result)`` makes a statsmodels or
linearmodels result a named object and ``qx.series_from(...)`` adds a series; ``qx.connect(url, token)`` drives a
running engine from any Python. Every function is a registry tool (``qx.tool`` reaches any of them by name), and
every call goes through the engine's dispatcher, so it is journaled and respects locks. Code shown in the chat and
in the journal is written in this API.
"""

from qtrix_engine.qx.runtime import (
    Obj,
    Page,
    connect,
    module_dir,
    module_getattr,
    obj,
    page,
    register,
    report,
    series_from,
    set_backend,
    set_run,
    tool,
)

__all__ = ["Obj", "Page", "connect", "obj", "page", "register", "report", "series_from", "set_backend", "set_run",
           "tool"]


def __getattr__(attr: str):  # qx.equation, qx.sql, qx.import_file…: generated from the registry
    return module_getattr(attr)


def __dir__() -> list[str]:
    return module_dir()
