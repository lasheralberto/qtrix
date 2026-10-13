"""ObjectStore: named, persistent, versioned objects on the workfile's DuckDB (ADR 0002 §3).

Objects persist as declarative specs plus JSON summaries, never pickles. The fitted library object lives in an
in-process LRU keyed by ``(name, version)`` and is rebuilt by re-estimating the spec on a miss. Mutations are
checked against locks (only the user may touch a locked object) and announced to listeners, which is the seam
the dispatcher/journal (E7·06) and the report (E7·08) use. The actor comes from the dispatcher (``callctx``).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..callctx import USER, acting_as, current_actor, current_recording
from . import specs as sp

__all__ = ["ObjectStore", "StoredObject", "acting_as", "current_actor", "USER"]

if TYPE_CHECKING:
    from ..workspace import Workspace
    from .estimators import Estimate

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
STATUSES = ("unchecked", "ok", "warn", "fail", "stale")

OBJECTS_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_objects (
    name VARCHAR PRIMARY KEY,
    type VARCHAR NOT NULL,
    page VARCHAR,
    version INTEGER NOT NULL,
    spec JSON NOT NULL,
    summary JSON NOT NULL,
    status VARCHAR NOT NULL DEFAULT 'unchecked',
    status_detail JSON,
    author VARCHAR,
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    parent VARCHAR,
    data_version INTEGER,
    code VARCHAR,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
"""
VERSIONS_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_object_versions (
    name VARCHAR NOT NULL,
    version INTEGER NOT NULL,
    spec JSON NOT NULL,
    summary JSON NOT NULL,
    status VARCHAR,
    author VARCHAR,
    journal_seq BIGINT,
    data_version INTEGER,
    code VARCHAR,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (name, version)
)
"""
VCOLS = "name, version, spec, summary, status, author, journal_seq, data_version, code, created_at"
COLS = ("name, type, page, version, spec, summary, status, status_detail, author, locked, parent, data_version, "
        "code, created_at, updated_at")


# ---- errors -------------------------------------------------------------------------------------------------
class ObjectError(ValueError):
    """Base of the object-store errors; being a ValueError, agents get it back as a ModelRetry."""


class ObjectNotFound(ObjectError, KeyError):
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class LockedObjectError(ObjectError):
    """A locked object may only be modified, re-estimated or deleted by the user."""


class StaleObjectError(ObjectError):
    """The object's data changed since it was estimated; its fitted model is not rebuilt silently."""


@dataclass
class StoredObject:
    name: str
    type: str
    page: str
    version: int
    spec: dict[str, Any]
    summary: dict[str, Any]
    status: str = "unchecked"
    status_detail: list[dict[str, Any]] = field(default_factory=list)
    author: str = "agent"
    locked: bool = False
    parent: str | None = None
    data_version: int | None = None
    code: str = ""
    created_at: str = ""
    updated_at: str = ""
    extras: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)  # transient by-products
    _store: ObjectStore | None = field(default=None, repr=False, compare=False)

    # Compatibility with the former StoredResult (ResultStore): id, kind and the fitted object.
    @property
    def id(self) -> str:
        return self.name

    @property
    def kind(self) -> str:
        return self.type

    @property
    def method(self) -> str:
        return self.spec.get("method", "")

    @property
    def obj(self) -> Any:
        return self._store.fitted(self.name) if self._store else None

    def to_dict(self, *, detail: bool = True) -> dict[str, Any]:
        out = {"name": self.name, "type": self.type, "method": self.method, "page": self.page,
               "version": self.version, "status": self.status, "locked": self.locked, "author": self.author,
               "parent": self.parent, "formula": self.spec.get("formula"), "data_version": self.data_version,
               "created_at": self.created_at, "updated_at": self.updated_at}
        if detail:
            out.update(spec=self.spec, summary=self.summary, status_detail=self.status_detail, code=self.code)
        return out


def _jsonable(row: tuple | None) -> list[Any] | None:
    return None if row is None else [v.isoformat() if isinstance(v, datetime) else v for v in row]


def _revive(row: list[Any], timestamps: tuple[int, ...]) -> list[Any]:
    return [datetime.fromisoformat(v) if i in timestamps and isinstance(v, str) else v for i, v in enumerate(row)]


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _num(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def diff_summaries(old: dict[str, Any], new: dict[str, Any], from_version: int) -> dict[str, Any]:
    """Coefficient and fit-statistic changes between two versions of an object."""
    a = {c["name"]: c for c in old.get("coefficients", [])}
    b = {c["name"]: c for c in new.get("coefficients", [])}
    coefs: dict[str, Any] = {}
    for key in [*b, *[k for k in a if k not in b]]:
        before, after = a.get(key), b.get(key)
        row: dict[str, Any] = {"before": before and before.get("coef"), "after": after and after.get("coef"),
                               "se_before": before and before.get("se"), "se_after": after and after.get("se")}
        if before and after and _num(before.get("coef")) and _num(after.get("coef")):
            row["delta"] = after["coef"] - before["coef"]
        coefs[key] = row
    stats = {k: {"before": old.get(k), "after": new.get(k)}
             for k in ("n_obs", "r2", "adj_r2", "aic", "bic", "log_likelihood", "cov_type", "formula", "sample")
             if (old.get(k) is not None or new.get(k) is not None) and old.get(k) != new.get(k)}
    return {"from_version": from_version, "coefficients": coefs, "stats": stats}


class ObjectStore:
    """All named objects of a workfile. Shares the workspace's DuckDB connection and lock."""

    def __init__(self, workspace: Workspace, cache_size: int = 32) -> None:
        self._ws = workspace
        self._lock = workspace._lock
        self._cache: OrderedDict[tuple[str, int], Any] = OrderedDict()
        self._cache_size = cache_size
        # By-products of an estimate (figures) and computed views (E7·08), keyed by object version like the fitted LRU.
        self._extras: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._views: OrderedDict[tuple[str, int, str, str], Any] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._aliases: dict[str, str] = {}
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    # ---- events --------------------------------------------------------------------------------------------
    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def _emit(self, event: str, **payload: Any) -> None:
        for listener in list(self._listeners):
            try:
                listener({"event": event, **payload})
            except Exception:  # noqa: BLE001 - a broken listener must never break a mutation
                log.exception("Object listener failed on %s", event)

    # ---- names -----------------------------------------------------------------------------------------------
    @staticmethod
    def valid_name(name: str) -> str:
        norm = name.strip().lower()
        if not NAME_RE.match(norm):
            raise ObjectError(f"Invalid object name '{name}': use a letter followed by letters, digits or '_' "
                              "(at most 32 characters), e.g. eq_taylor")
        return norm

    def add_alias(self, old_id: str, name: str) -> None:
        """Let a former ``r_xxxx`` result id resolve to an object name (compatibility for one release)."""
        self._aliases[old_id.lower()] = self.get(name).name

    def resolve(self, ref: str) -> str:
        key = ref.strip().lower()
        return self._aliases.get(key, key)

    def _taken(self, name: str) -> bool:
        return bool(self._ws._con.execute("SELECT 1 FROM _qtrix_objects WHERE name = ? UNION ALL "
                                          "SELECT 1 FROM _qtrix_object_versions WHERE name = ? LIMIT 1",
                                          [name, name]).fetchall())

    def default_name(self, type_: str) -> str:
        prefix = sp.DEFAULT_PREFIX[type_]
        with self._lock:
            i = 1
            while self._taken(f"{prefix}{i:02d}"):
                i += 1
        return f"{prefix}{i:02d}"

    # ---- reads -----------------------------------------------------------------------------------------------
    def _load(self, row: tuple) -> StoredObject:
        obj = StoredObject(
            name=row[0], type=row[1], page=row[2] or "", version=row[3], spec=json.loads(row[4]),
            summary=json.loads(row[5]), status=row[6], status_detail=json.loads(row[7]) if row[7] else [],
            author=row[8] or "agent", locked=bool(row[9]), parent=row[10], data_version=row[11], code=row[12] or "",
            created_at=row[13].isoformat(), updated_at=row[14].isoformat(), _store=self)
        if obj.status != "stale" and obj.page and obj.data_version is not None:
            current = self._ws.page_data_version(obj.page)
            if current is not None and current != obj.data_version:  # the data changed under the object
                obj.status = "stale"
                obj.status_detail = [{"check": "data", "outcome": "stale", "view": None,
                                      "message": f"data of '{obj.page}' changed since version {obj.version}"}]
                self._ws._con.execute("UPDATE _qtrix_objects SET status = 'stale', status_detail = ? WHERE name = ?",
                                      [json.dumps(obj.status_detail), obj.name])
                self._emit("stale", name=obj.name, version=obj.version, reason="data")
        return obj

    def get(self, ref: str) -> StoredObject:
        """An object by name (case-insensitive) or by former ``r_xxxx`` alias."""
        name = self.resolve(ref)
        with self._lock:
            row = self._ws._con.execute(f"SELECT {COLS} FROM _qtrix_objects WHERE name = ?", [name]).fetchone()
            if row is None:
                raise ObjectNotFound(f"Unknown object '{ref}'")
            return self._load(row)

    def exists(self, ref: str) -> bool:
        with self._lock:
            return bool(self._ws._con.execute("SELECT 1 FROM _qtrix_objects WHERE name = ?",
                                              [self.resolve(ref)]).fetchall())

    def list(self, type_: str | None = None) -> list[StoredObject]:
        with self._lock:
            where = " WHERE type = ?" if type_ else ""
            sql = f"SELECT {COLS} FROM _qtrix_objects{where} ORDER BY created_at, name"
            return [self._load(r) for r in self._ws._con.execute(sql, [type_] if type_ else []).fetchall()]

    def versions(self, ref: str) -> list[dict[str, Any]]:
        """Version history of a name (kept after a soft delete)."""
        name = self.resolve(ref)
        with self._lock:
            rows = self._ws._con.execute(
                "SELECT version, spec, summary, status, author, data_version, code, created_at "
                "FROM _qtrix_object_versions WHERE name = ? ORDER BY version", [name]).fetchall()
        if not rows:
            raise ObjectNotFound(f"Unknown object '{ref}'")
        return [{"version": r[0], "spec": json.loads(r[1]), "summary": json.loads(r[2]), "status": r[3],
                 "author": r[4], "data_version": r[5], "code": r[6], "created_at": r[7].isoformat()} for r in rows]

    def diff(self, ref: str, from_version: int | None = None, to_version: int | None = None) -> dict[str, Any]:
        """Coefficient/fit differences between two versions (default: previous vs current)."""
        history = {v["version"]: v for v in self.versions(ref)}
        last = max(history)
        b = to_version or last
        a = from_version or max((v for v in history if v < b), default=None)
        if a is None or a not in history or b not in history:
            raise ObjectError(f"'{ref}' has versions {sorted(history)}; cannot diff {a} -> {b}")
        return {"name": self.resolve(ref), "to_version": b,
                **diff_summaries(history[a]["summary"], history[b]["summary"], a)}

    def dependants(self, ref: str) -> list[str]:
        """Names of the objects built from ``ref`` (tables comparing it, graphs of it...)."""
        name = self.resolve(ref)
        return [o.name for o in self.list() if name in [r.lower() for r in sp.parse_spec(o.spec).refs()]]

    # ---- fitted cache and rebuild --------------------------------------------------------------------------
    def rebuild(self, ref: str) -> Estimate:
        """Re-estimate the current version from its spec. With unchanged data it reproduces the stored summary."""
        from .estimators import estimate

        return estimate(sp.parse_spec(self.get(ref).spec))

    def fitted(self, ref: str) -> Any:
        """The library result of the current version: from the LRU, else rebuilt (never silently when stale)."""
        obj = self.get(ref)
        key = (obj.name, obj.version)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        if obj.type not in ("equation", "var"):
            raise ObjectError(f"'{obj.name}' is a {obj.type}: it has no fitted model")
        if obj.method == "python":
            raise ObjectError(f"'{obj.name}' was registered from a Python cell: its fitted result lives in the kernel "
                              "(work with it there), and the engine only keeps its summary")
        if obj.status == "stale":
            raise StaleObjectError(f"{obj.name} is stale (its data or inputs changed since version {obj.version}); "
                                   f"re-estimate it with the same name to refresh it")
        fitted = self.rebuild(obj.name).fitted
        self._cache_put(key, fitted)
        return fitted

    def extras(self, ref: str) -> dict[str, Any]:
        """By-products of the current version's estimate (forecast/IRF/volatility figures): kept, else rebuilt."""
        obj = self.get(ref)
        key = (obj.name, obj.version)
        with self._cache_lock:
            if key in self._extras:
                return self._extras[key]
        if obj.status == "stale":
            raise StaleObjectError(f"{obj.name} is stale (its data or inputs changed since version {obj.version}); "
                                   f"re-estimate it with the same name to refresh it")
        est = self.rebuild(obj.name)
        self._cache_put(key, est.fitted, est.extras)
        return est.extras

    def view_cached(self, key: tuple[str, int, str, str]) -> Any:
        with self._cache_lock:
            if key in self._views:
                self._views.move_to_end(key)
                return self._views[key]
        return None

    def view_put(self, key: tuple[str, int, str, str], value: Any) -> None:
        with self._cache_lock:
            self._views[key] = value
            while len(self._views) > self._cache_size * 8:
                self._views.popitem(last=False)

    def _cache_put(self, key: tuple[str, int], fitted: Any, extras: dict[str, Any] | None = None) -> None:
        if extras:
            with self._cache_lock:
                self._extras[key] = extras
                while len(self._extras) > self._cache_size:
                    self._extras.popitem(last=False)
        if fitted is None:
            return
        with self._cache_lock:
            self._cache[key] = fitted
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _cache_drop(self, name: str) -> None:
        with self._cache_lock:
            for cache in (self._cache, self._extras, self._views):
                for key in [k for k in cache if k[0] == name]:
                    del cache[key]

    # ---- before-images (journal undo, E7·06) ---------------------------------------------------------------
    def image(self, name: str) -> dict[str, Any]:
        """JSON-able copy of an object's row and version rows (row is None when the name is not live)."""
        with self._lock:
            con = self._ws._con
            row = con.execute(f"SELECT {COLS} FROM _qtrix_objects WHERE name = ?", [name]).fetchone()
            vers = con.execute(f"SELECT {VCOLS} FROM _qtrix_object_versions WHERE name = ? ORDER BY version",
                               [name]).fetchall()
        return {"row": _jsonable(row), "versions": [_jsonable(v) for v in vers]}

    def _touch(self, *names: str) -> None:
        """Remember the before-image of objects the running dispatched call is about to change."""
        rec = current_recording()
        if rec is None:
            return
        for name in names:
            if name not in rec.images:
                rec.images[name] = self.image(name)

    def _seq(self) -> int | None:
        rec = current_recording()
        return rec.seq if rec else None

    def restore_image(self, name: str, image: dict[str, Any]) -> None:
        """Put an object (live row and version rows) back exactly as the image recorded it."""
        self._touch(name)
        with self._lock:
            con = self._ws._con
            con.execute("DELETE FROM _qtrix_objects WHERE name = ?", [name])
            con.execute("DELETE FROM _qtrix_object_versions WHERE name = ?", [name])
            if image["row"] is not None:
                con.execute(f"INSERT INTO _qtrix_objects ({COLS}) VALUES ({', '.join('?' * 15)})",
                            _revive(image["row"], (13, 14)))
            for v in image["versions"]:
                con.execute(f"INSERT INTO _qtrix_object_versions ({VCOLS}) VALUES ({', '.join('?' * 10)})",
                            _revive(v, (9,)))
        self._cache_drop(name)
        self._emit("undo", name=name)

    def mark_stale(self, name: str, message: str) -> None:
        """Flag a live object stale because something it was built from changed (undo uses it)."""
        if not self.exists(name):
            return
        self._touch(name)
        with self._lock:
            self._ws._con.execute("UPDATE _qtrix_objects SET status = 'stale', status_detail = ? WHERE name = ?",
                                  [json.dumps([{"check": "dependency", "outcome": "stale", "view": None,
                                                "message": message}]), name])
        self._emit("stale", name=name, reason="dependency")

    def clear_stale_data(self, page: str, data_version: int | None) -> None:
        """After page data was restored: objects that only went stale because of it are valid again."""
        with self._lock:
            rows = self._ws._con.execute(
                "SELECT name, data_version, status_detail FROM _qtrix_objects WHERE page = ? AND status = 'stale'",
                [page]).fetchall()
        for name, dv, detail in rows:
            if dv == data_version and '"check": "data"' in str(detail):
                self._touch(name)
                with self._lock:
                    self._ws._con.execute("UPDATE _qtrix_objects SET status = 'unchecked', status_detail = NULL "
                                          "WHERE name = ?", [name])

    def reset(self) -> None:
        """Forget every object and version (journal replay rebuilds them from the sources)."""
        with self._lock:
            self._ws._con.execute("DELETE FROM _qtrix_objects")
            self._ws._con.execute("DELETE FROM _qtrix_object_versions")
        with self._cache_lock:
            self._cache.clear()
        self._aliases.clear()

    # ---- writes ----------------------------------------------------------------------------------------------
    def _check_unlocked(self, obj: StoredObject, what: str, actor: str) -> None:
        if obj.locked and actor != USER:
            raise LockedObjectError(f"'{obj.name}' is locked by the user and cannot be {what}. Ask the user to "
                                    "unlock it, or work on a different name; do not copy it to get around the lock")

    def estimate(self, spec: sp._Spec, name: str | None = None, *, parent: str | None = None) -> StoredObject:
        """Estimate ``spec`` and store it under ``name`` (default eq01, var01...).

        An existing name gets version n+1 with a coefficient diff against the previous version, and objects
        built from it become stale. Locked objects only accept the user.
        """
        from .estimators import estimate as run

        actor = current_actor()
        name = self.valid_name(name) if name else None
        if name and self.exists(name):
            old = self.get(name)
            self._check_unlocked(old, "modified or re-estimated", actor)
            if old.type != spec.type:
                raise ObjectError(f"'{name}' is a {old.type}; it cannot be replaced by a {spec.type}. "
                                  "Pick another name")
        est = run(spec)
        return self._save(spec, est, name, actor, parent)

    def register(self, spec: sp.PythonSpec, summary: dict[str, Any], name: str | None = None) -> StoredObject:
        """Store a result computed in the kernel (its summary comes with it), with the same rules as ``estimate``."""
        from .estimators import Estimate

        actor = current_actor()
        name = self.valid_name(name) if name else None
        if name and self.exists(name):
            old = self.get(name)
            self._check_unlocked(old, "modified or re-estimated", actor)
            if old.type != spec.type:
                raise ObjectError(f"'{name}' is a {old.type}; it cannot be replaced by a {spec.type}. "
                                  "Pick another name")
        return self._save(spec, Estimate(summary, None, spec.code), name, actor, None)

    def reestimate(self, ref: str, **changes: Any) -> StoredObject:
        """New version of an object with some spec fields changed (another sample, cov_type, formula...)."""
        obj = self.get(ref)
        return self.estimate(sp.with_changes(sp.parse_spec(obj.spec), **changes), obj.name)

    def _save(self, spec: sp._Spec, est: Estimate, name: str | None, actor: str, parent: str | None) -> StoredObject:
        now = _now()
        data_version = self._ws.page_data_version(spec.page_name) if spec.page_name else None
        with self._lock:
            name = name or self.default_name(spec.type)
            old = self.get(name) if self.exists(name) else None
            if old:
                self._check_unlocked(old, "modified or re-estimated", actor)
            self._touch(name)
            prev = self._ws._con.execute("SELECT MAX(version) FROM _qtrix_object_versions WHERE name = ?",
                                         [name]).fetchone()[0]
            version = (prev or 0) + 1
            summary = est.summary
            if old:
                summary["diff"] = diff_summaries(old.summary, summary, old.version)
            spec_json, summary_json = json.dumps(spec.dump()), json.dumps(summary)
            args = [spec.type, spec.page_name, version, spec_json, summary_json, "unchecked", None, actor,
                    data_version, est.code]
            if old:
                self._ws._con.execute(
                    "UPDATE _qtrix_objects SET type=?, page=?, version=?, spec=?, summary=?, status=?, "
                    "status_detail=?, author=?, data_version=?, code=?, updated_at=? WHERE name=?",
                    [*args, now, name])
            else:
                self._ws._con.execute(
                    f"INSERT INTO _qtrix_objects ({COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, ?, ?, ?, ?, ?)",
                    [name, spec.type, spec.page_name, version, spec_json, summary_json, "unchecked", None, actor,
                     parent, data_version, est.code, now, now])
            self._ws._con.execute(
                "INSERT INTO _qtrix_object_versions (name, version, spec, summary, status, author, journal_seq, "
                "data_version, code, created_at) VALUES (?, ?, ?, ?, 'unchecked', ?, ?, ?, ?, ?)",
                [name, version, spec_json, summary_json, actor, self._seq(), data_version, est.code, now])
            obj = self.get(name)
        obj.extras = est.extras
        self._cache_drop(name)  # an undo may have reused this version number: nothing cached for it is valid
        self._cache_put((name, version), est.fitted, est.extras)
        self._emit("version", name=name, version=version, type=spec.type, previous=old.version if old else None)
        if old:
            self._mark_dependants_stale(name, version)
        return obj

    def _mark_dependants_stale(self, name: str, version: int) -> None:
        seen = {name}
        queue = [name]
        while queue:
            current = queue.pop()
            for dep in self.dependants(current):
                if dep in seen:
                    continue
                seen.add(dep)
                detail = [{"check": "dependency", "outcome": "stale", "view": None,
                           "message": f"{current} changed (version {version}); re-estimate {dep}"}]
                self._touch(dep)
                with self._lock:
                    self._ws._con.execute("UPDATE _qtrix_objects SET status = 'stale', status_detail = ? "
                                          "WHERE name = ?", [json.dumps(detail), dep])
                self._emit("stale", name=dep, reason="dependency", source=current)
                queue.append(dep)

    def copy(self, ref: str, new_name: str | None = None) -> StoredObject:
        """Duplicate an object (version 1 of the new name, ``parent`` = source). Allowed on locked objects."""
        src = self.get(ref)
        new = self.valid_name(new_name) if new_name else None
        if new and self.exists(new):
            raise ObjectError(f"'{new}' already exists")
        now = _now()
        with self._lock:
            new = new or self.default_name(src.type)
            if self._taken(new):
                raise ObjectError(f"'{new}' was used before; choose a different name")
            self._touch(new)
            spec_json, summary_json = json.dumps(src.spec), json.dumps(src.summary)
            actor = current_actor()
            self._ws._con.execute(
                f"INSERT INTO _qtrix_objects ({COLS}) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, FALSE, ?, ?, ?, ?, ?)",
                [new, src.type, src.page, spec_json, summary_json, src.status, json.dumps(src.status_detail),
                 actor, src.name, src.data_version, src.code, now, now])
            self._ws._con.execute(
                "INSERT INTO _qtrix_object_versions (name, version, spec, summary, status, author, journal_seq, "
                "data_version, code, created_at) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
                [new, spec_json, summary_json, src.status, actor, self._seq(), src.data_version, src.code, now])
            out = self.get(new)
        self._emit("copy", name=new, source=src.name)
        return out

    def rename(self, ref: str, new_name: str) -> StoredObject:
        obj = self.get(ref)
        new = self.valid_name(new_name)
        self._check_unlocked(obj, "renamed", current_actor())
        if new == obj.name:
            return obj
        with self._lock:
            if self._taken(new):
                raise ObjectError(f"'{new}' already exists or was used before; choose a different name")
            con = self._ws._con
            deps = self.dependants(obj.name)
            children = [r[0] for r in con.execute("SELECT name FROM _qtrix_objects WHERE parent = ?",
                                                  [obj.name]).fetchall()]
            self._touch(obj.name, new, *deps, *children)
            # Delete + insert instead of UPDATE of the key: DuckDB cannot update an indexed column in place.
            row = con.execute(f"SELECT {COLS} FROM _qtrix_objects WHERE name = ?", [obj.name]).fetchone()
            con.execute("DELETE FROM _qtrix_objects WHERE name = ?", [obj.name])
            con.execute(f"INSERT INTO _qtrix_objects ({COLS}) VALUES ({', '.join('?' * 15)})", [new, *row[1:]])
            rows = con.execute("SELECT version, spec, summary, status, author, journal_seq, data_version, code, "
                               "created_at FROM _qtrix_object_versions WHERE name = ?", [obj.name]).fetchall()
            con.execute("DELETE FROM _qtrix_object_versions WHERE name = ?", [obj.name])
            for r in rows:
                con.execute("INSERT INTO _qtrix_object_versions (name, version, spec, summary, status, author, "
                            "journal_seq, data_version, code, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [new, *r])
            con.execute("UPDATE _qtrix_objects SET parent = ? WHERE parent = ?", [new, obj.name])
            for dep in deps:  # keep the references of tables/graphs pointing at the renamed object
                self._rewrite_refs(dep, obj.name, new)
        self._cache_drop(obj.name)
        self._aliases = {k: (new if v == obj.name else v) for k, v in self._aliases.items()}
        self._emit("rename", name=new, old_name=obj.name)
        return self.get(new)

    def _rewrite_refs(self, dep: str, old: str, new: str) -> None:
        row = self._ws._con.execute("SELECT spec FROM _qtrix_objects WHERE name = ?", [dep]).fetchone()
        data = json.loads(row[0])
        if "sources" in data:
            data["sources"] = [new if s.lower() == old else s for s in data["sources"]]
        if data.get("source") and data["source"].lower() == old:
            data["source"] = new
        self._ws._con.execute("UPDATE _qtrix_objects SET spec = ? WHERE name = ?", [json.dumps(data), dep])

    def delete(self, ref: str) -> str:
        """Soft delete: the object leaves the workfile, its versions stay in the history table."""
        obj = self.get(ref)
        self._check_unlocked(obj, "deleted", current_actor())
        deps = self.dependants(obj.name)
        self._touch(obj.name, *deps)
        with self._lock:
            self._ws._con.execute("DELETE FROM _qtrix_objects WHERE name = ?", [obj.name])
        self._cache_drop(obj.name)
        for dep in deps:
            with self._lock:
                self._ws._con.execute(
                    "UPDATE _qtrix_objects SET status = 'stale', status_detail = ? WHERE name = ?",
                    [json.dumps([{"check": "dependency", "outcome": "stale", "view": None,
                                  "message": f"{obj.name} was deleted"}]), dep])
            self._emit("stale", name=dep, reason="dependency", source=obj.name)
        self._emit("delete", name=obj.name)
        return obj.name

    def lock(self, ref: str) -> StoredObject:
        return self._set_locked(ref, True)

    def unlock(self, ref: str) -> StoredObject:
        if current_actor() != USER:
            raise LockedObjectError(f"Only the user can unlock '{ref}'")
        return self._set_locked(ref, False)

    def _set_locked(self, ref: str, locked: bool) -> StoredObject:
        obj = self.get(ref)
        self._touch(obj.name)
        with self._lock:
            self._ws._con.execute("UPDATE _qtrix_objects SET locked = ?, updated_at = ? WHERE name = ?",
                                  [locked, _now(), obj.name])
        self._emit("lock" if locked else "unlock", name=obj.name)
        return self.get(obj.name)
