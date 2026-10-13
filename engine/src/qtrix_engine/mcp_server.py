"""qtrix-mcp: the econometrics tool registry exposed as a standalone MCP server.

Lets any MCP client (Claude Desktop, IDEs, other agent frameworks) use the same tools as the qtrix
agents, against a local workspace:

    qtrix-mcp                      # stdio, workspace in QTRIX_HOME
    qtrix-mcp --workspace data.duckdb

Panel blocks produced by the tools are kept in memory and can be listed with ``ui_list_blocks``.
"""

from __future__ import annotations

import argparse
import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import __version__
from .config import Settings
from .context import EngineContext, set_context
from .dispatch import Call, dispatch
from .notebooks import NotebookManager
from .tools.registry import NAMESPACES, ToolSpec, all_tools
from .workspace import Workspace

INSTRUCTIONS = (
    "qtrix exposes econometrics tools over a local data workspace. Start with data_list_datasets or "
    "data_import_file, then estimate with ols_*, panel_*, iv_*, ts_* and visualise with viz_*. "
    "Datasets are referenced by id or name; equations, VARs and tables are named objects "
    "(result_id = object name; re-using a name re-estimates it as a new version). Namespaces: "
    + "; ".join(f"{k}: {v}" for k, v in NAMESPACES.items())
)


def _dispatching(spec: ToolSpec) -> Callable[..., Any]:
    """The tool as MCP clients call it: through the dispatcher, as actor ``mcp`` (locks, journal)."""
    signature = inspect.signature(spec.fn)

    @functools.wraps(spec.fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        return dispatch(Call(spec.name, dict(signature.bind(*args, **kwargs).arguments)), "mcp", "mcp")

    return call


def build_server(namespaces: list[str] | None = None) -> MCPServer:
    server = MCPServer(name="qtrix", title="qtrix econometrics", version=__version__, instructions=INSTRUCTIONS)
    for spec in all_tools(namespaces=namespaces):
        server.add_tool(_dispatching(spec), name=spec.name, description=spec.description)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(prog="qtrix-mcp")
    parser.add_argument("--workspace", help="Notebook id or DuckDB file (defaults to the notebook opened last)")
    parser.add_argument("--namespace", action="append", help="Restrict to these namespaces (repeatable)")
    args = parser.parse_args()
    settings = Settings.from_env()
    notebooks = NotebookManager(settings.notebooks_dir, legacy_db=settings.db_path, legacy_memory=settings.memory_dir)
    if args.workspace and Path(args.workspace).suffix:
        notebook, db = None, Path(args.workspace)
    else:
        notebook = notebooks.get(args.workspace) if args.workspace else notebooks.current()
        db = notebook.db_path
    set_context(EngineContext(settings=settings, workspace=Workspace(db), notebook=notebook))
    build_server(args.namespace).run("stdio")


if __name__ == "__main__":
    main()
