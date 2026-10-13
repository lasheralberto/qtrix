import json
import sys

import pytest

from qtrix_engine import config
from qtrix_engine.config import Settings, migrate_legacy, resolve_home


def test_home_resolution_order(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "bootstrap_dir", lambda: tmp_path / "boot")
    monkeypatch.setattr(config, "default_home", lambda: tmp_path / "qtrix")
    monkeypatch.delenv("QTRIX_HOME", raising=False)
    assert resolve_home() == (tmp_path / "qtrix", "default")

    config.write_location(tmp_path / "elsewhere")
    assert json.loads((tmp_path / "boot" / "location.json").read_text())["home"] == str(tmp_path / "elsewhere")
    assert resolve_home() == (tmp_path / "elsewhere", "location")

    monkeypatch.setenv("QTRIX_HOME", str(tmp_path / "env"))
    assert resolve_home() == (tmp_path / "env", "env")


def test_layout_is_created(tmp_path):
    s = Settings(home=tmp_path / "q", token=None, local_only=False)
    for sub in ("data", "memory", "models", "runtime", "logs", "uploads", "exports"):
        assert (s.home / sub).is_dir()
    assert (s.home / "LEEME.txt").read_text(encoding="utf-8").startswith("qtrix")
    assert s.db_path == s.home / "data" / "workspace.duckdb"


def test_migrates_legacy_hidden_folder(tmp_path):
    legacy = tmp_path / ".qtrix"
    (legacy / "memory").mkdir(parents=True)
    (legacy / "memory" / "items.jsonl").write_text("{}")
    (legacy / "workspace.duckdb").write_bytes(b"db")
    (legacy / "settings.json").write_text("{}")
    home = tmp_path / "qtrix"
    Settings(home=home, token=None, local_only=False)

    moved = migrate_legacy(home, legacy)

    assert set(moved) == {"workspace.duckdb", "settings.json", "memory"}
    assert (home / "data" / "workspace.duckdb").read_bytes() == b"db"
    assert (home / "memory" / "items.jsonl").exists()
    assert not legacy.exists()  # emptied and removed
    assert migrate_legacy(home, legacy) == []  # idempotent


@pytest.mark.skipif(sys.platform != "win32", reason="open files block renames only on Windows")
def test_locked_legacy_workspace_aborts_without_moving_anything(tmp_path):
    legacy = tmp_path / ".qtrix"
    legacy.mkdir()
    (legacy / "workspace.duckdb").write_bytes(b"db")
    (legacy / "settings.json").write_text("{}")
    home = tmp_path / "qtrix"
    Settings(home=home, token=None, local_only=False)
    with (legacy / "workspace.duckdb").open("rb"), pytest.raises(RuntimeError, match="Cierra cualquier otra"):
        migrate_legacy(home, legacy)
    assert (legacy / "settings.json").exists() and not (home / "settings.json").exists()
    assert not (home / "data" / "workspace.duckdb").exists()  # next start retries the migration
