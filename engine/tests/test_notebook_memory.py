"""E8·03: semantic memory per notebook (plus a user-wide one) and idle Python kernels."""

from __future__ import annotations

from qtrix_engine.config import Settings
from qtrix_engine.kernel.manager import KernelPool
from qtrix_engine.memory import NotebookMemory, ProjectMemory
from qtrix_engine.workspace import Workspace

from .test_api import client
from .test_memory import fake_embed


def mem(path) -> ProjectMemory:
    return ProjectMemory(path, fake_embed, 64)


def test_a_notebook_only_recalls_its_own_history_and_the_users_preferences(tmp_path):
    user = mem(tmp_path / "user")
    a, b = NotebookMemory(mem(tmp_path / "a"), user), NotebookMemory(mem(tmp_path / "b"), user)
    a.add("assistant", "the wage regression found a return to education of eight percent")
    a.add("preference", "always report heteroskedasticity robust standard errors")

    assert "eight percent" in a.recall("return to education")
    assert b.recall("return to education") == ""  # b never saw a's analysis
    assert "robust standard errors" in b.recall("which standard errors do you report")  # the user's habits follow
    assert len(a.notebook) == 1 and len(user) == 1  # the preference was written user-wide, not into the notebook


def test_the_notebook_comes_first_in_a_search(tmp_path):
    user = mem(tmp_path / "user")
    nb = NotebookMemory(mem(tmp_path / "nb"), user)
    user.add("preference", "report robust standard errors always")
    nb.add("note", "report robust standard errors in this notebook")
    hits = nb.search("report robust standard errors", k=2, min_score=0)
    assert [h[0].kind for h in hits] == ["note", "preference"]


def start_with_memory(tmp_path):
    c = client(tmp_path)
    engine = c.app.state.engine
    engine._embedder = (fake_embed, 64)
    engine.memory = mem(tmp_path / "memory")
    return c, engine


def test_every_notebook_gets_its_own_index_next_to_its_data(tmp_path):
    c, engine = start_with_memory(tmp_path)
    a = c.post("/notebooks", json={"title": "A"}).json()["id"]
    b = c.post("/notebooks", json={"title": "B"}).json()["id"]
    for nid in (a, b):
        engine._attach_memory(engine.open(nid))
    engine.open(a).memory.add("assistant", "the macro model uses quarterly spanish gdp")
    hit = c.post(f"/notebooks/{a}/tools/memory_search", json={"query": "spanish gdp", "k": 3}).json()
    other = c.post(f"/notebooks/{b}/tools/memory_search", json={"query": "spanish gdp", "k": 3}).json()
    assert any("spanish gdp" in h["text"] for h in hit) and other == []
    assert (tmp_path / "notebooks" / a / "memory" / "items.jsonl").exists()
    assert not (tmp_path / "notebooks" / b / "memory" / "items.jsonl").exists()

    copy = c.post(f"/notebooks/{a}/duplicate", json={}).json()["id"]  # a duplicate keeps what its source remembers
    engine._attach_memory(engine.open(copy))
    assert any("spanish gdp" in h["text"] for h in
               c.post(f"/notebooks/{copy}/tools/memory_search", json={"query": "spanish gdp", "k": 3}).json())


def test_the_memory_of_earlier_versions_moves_into_the_first_notebook(tmp_path):
    settings = Settings(home=tmp_path, token=None, local_only=False)
    Workspace(settings.db_path).close()  # the single workspace of earlier versions
    mem(settings.memory_dir).add("user", "an old question about unemployment")

    c, engine = start_with_memory(tmp_path)
    first = c.get("/notebooks").json()[0]["id"]
    assert (tmp_path / "notebooks" / first / "memory" / "items.jsonl").exists()
    assert not (settings.memory_dir / "items.jsonl").exists()  # the user-wide index starts empty


class _StubKernel:
    def __init__(self, idle: float, busy: bool = False) -> None:
        self.alive, self.busy, self._idle, self.stopped = True, busy, idle, False

    def idle_for(self) -> float:
        return self._idle

    def shutdown(self) -> None:
        self.alive, self.stopped = False, True


def test_idle_kernels_are_stopped_and_busy_ones_are_not():
    pool = KernelPool(lambda: {})
    pool.project, pool._rebuild = _StubKernel(idle=100), _StubKernel(idle=100, busy=True)
    assert pool.shutdown_idle(max_idle=500) is False  # recently used
    assert pool.shutdown_idle(max_idle=60) is True
    assert pool.project.stopped and not pool._rebuild.stopped  # a running cell is never cut off


def test_sweeping_covers_every_open_notebook(tmp_path):
    c = client(tmp_path)
    engine = c.app.state.engine
    nid = c.post("/notebooks", json={"title": "A"}).json()["id"]
    stub = _StubKernel(idle=10_000)
    engine.open(nid).kernels.project = stub
    assert engine.sweep_kernels() == 1 and stub.stopped
    assert engine.sweep_kernels() == 0  # nothing left running
