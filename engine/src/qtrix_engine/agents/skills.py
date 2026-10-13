"""Skills: method guides for the lead agent, with the standard Pydantic AI Harness capabilities.

``Skills`` offers every skill of the library (a child folder of ``SKILLS/`` with a ``SKILL.md``) as a deferred
capability: the model sees its name and description and loads its instructions with ``load_capability``. ``Skills``
does not read a skill's bundled files, so a read-only ``FileSystem`` over the same library lets the model read
the references a guide points to (``read_file``, ``search_files``...; capability tools cannot be renamed). Paths in
the guides are relative to the library. Third-party skills can sit deeper in the library
(``eco-skills/econometrics-skill``): they are not offered on their own, only read through a guide that adapts them.

Both read the library through the run's workspace. Pydantic AI's ``LocalWorkspace`` runs only on POSIX (its command
timeouts kill process groups), so the library is served by ``SkillLibrary``: a read-only, filesystem-only backend,
without commands, that shows the folder as ``/skills`` on any platform.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.workspaces import FileEntry, ReadOnlyWorkspace, Workspace, WorkspaceReadOnlyError, WorkspaceRef
from pydantic_ai_harness import FileSystem, Skills

SKILLS_DIR = Path(__file__).parent / "SKILLS"
ROOT = "/skills"
FILE_TOOLS = ("read_file", "list_directory", "find_files", "search_files")
#: The agent's bookkeeping around skills: never shown as Python cells.
SKILL_TOOLS = frozenset({"load_capability", *FILE_TOOLS})
_READ_ONLY = "The skills library is read-only"


class SkillLibrary:
    """A folder of the host as a read-only workspace backend rooted at ``/skills`` (no commands)."""

    def __init__(self, library: Path) -> None:
        self.library = library.resolve()
        self._ref = WorkspaceRef(provider="qtrix-skills", id=self.library.as_posix())

    @property
    def ref(self) -> WorkspaceRef:
        return self._ref

    async def working_dir(self) -> str:
        return ROOT

    def _host(self, path: str) -> Path:
        """The host path behind a workspace path; anything outside the library does not exist here."""
        path = posixpath.normpath(path)
        if path != ROOT and not path.startswith(ROOT + "/"):
            raise FileNotFoundError(path)
        target = Path(os.path.realpath(self.library / path[len(ROOT) + 1:]))
        if not target.is_relative_to(self.library):
            raise FileNotFoundError(path)
        return target

    def _entry(self, path: str, target: Path) -> FileEntry:
        is_dir = target.is_dir()
        return FileEntry(name=posixpath.basename(path) or "/", path=path, is_dir=is_dir,
                         size=None if is_dir else target.stat().st_size)

    async def read_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(lambda: self._host(path).read_bytes())

    async def stat(self, path: str) -> FileEntry:
        def stat() -> FileEntry:
            target = self._host(path)
            if not target.exists():
                raise FileNotFoundError(path)
            return self._entry(posixpath.normpath(path), target)

        return await asyncio.to_thread(stat)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        def entries() -> list[FileEntry]:
            folder, base = self._host(path), posixpath.normpath(path)
            if not folder.is_dir():
                raise NotADirectoryError(path) if folder.exists() else FileNotFoundError(path)
            return [self._entry(posixpath.join(base, child.name), child)
                    for child in sorted(folder.iterdir(), key=lambda c: c.name)
                    if Path(os.path.realpath(child)).is_relative_to(self.library)]

        return await asyncio.to_thread(entries)

    async def exists(self, path: str) -> bool:
        try:
            return await asyncio.to_thread(lambda: self._host(path).exists())
        except FileNotFoundError:
            return False

    async def realpath(self, path: str) -> str:
        try:
            target = await asyncio.to_thread(self._host, path)
        except FileNotFoundError:
            return "/outside-skills"  # fails every containment check under /skills
        rel = target.relative_to(self.library).as_posix()
        return ROOT if rel == "." else f"{ROOT}/{rel}"

    async def write_bytes(self, path: str, data: bytes) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY)

    async def make_dir(self, path: str) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY)

    async def remove(self, path: str) -> Never:
        raise WorkspaceReadOnlyError(_READ_ONLY)


@dataclass
class SkillWorkspace(AbstractCapability[Any]):
    """Makes the skills library the run's (read-only) workspace."""

    library: Path

    def get_workspace(self, ctx: RunContext[Any], *, ref: WorkspaceRef | None) -> Any:
        backend = SkillLibrary(self.library)
        if ref is not None and ref != backend.ref:
            return None
        return ReadOnlyWorkspace(Workspace(backend))


def skill_capabilities(library: Path | None = SKILLS_DIR) -> list[AbstractCapability[Any]]:
    """The capabilities that give an agent the skills of ``library`` (none without a library or skills in it)."""
    if library is None or not library.is_dir() or not any((d / "SKILL.md").is_file() for d in library.iterdir()):
        return []
    files = FileSystem(read_only=True, tools=FILE_TOOLS, content_hashes=False, max_read_chars=30_000,
                       denied_patterns=("**/.git/**",))
    return [SkillWorkspace(library), Skills("."), files]
