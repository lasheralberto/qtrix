"""Skills: method guides offered with the Pydantic AI Harness `Skills` capability, loaded on demand with
`load_capability`, and their references read through a read-only `FileSystem`."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart

from qtrix_engine.agents import skills as sk
from qtrix_engine.contrib.econometrics import LIB_DIR

from .test_planner import FIT, client, events, scripted, text

DETAILS = "eco-skills/econometrics-skill/references/method_details.md"


def returns(history, tool: str) -> list[str]:
    return [str(p.content) for m in history if isinstance(m, ModelRequest) for p in m.parts
            if isinstance(p, ToolReturnPart) and p.tool_name == tool]


def test_capabilities_only_for_a_library_with_skills(tmp_path):
    assert len(sk.skill_capabilities()) == 3
    assert sk.skill_capabilities(None) == [] and sk.skill_capabilities(tmp_path / "missing") == []
    (tmp_path / "loose").mkdir()
    assert sk.skill_capabilities(tmp_path) == []


@pytest.mark.skipif(not LIB_DIR.exists(), reason="eco-skills not fetched: run `npm run setup:skills`")
def test_the_agent_loads_the_guide_and_reads_its_references_without_cells(tmp_path, wages_csv):
    model = scripted([], [
        ("load_capability", {"id": "qtrix-econometrics"}),
        ("search_files", {"pattern": "^## 14\\. Sharp RDD", "path": "eco-skills"}),
        ("read_file", {"path": DETAILS, "offset": 374, "limit": 30}),
        ("read_file", {"path": "eco-skills/econometrics-skill/.git/config"}),
        ("read_file", {"path": "../lead.py"}),
        ("ols_fit", FIT),
        "Hecho.",
    ])
    c = client(tmp_path, wages_csv, model, planner=False)
    evs = events(c.post("/chat", json={"message": "efecto de la educación"}))
    assert [e["name"] for e in evs if e["type"] in ("tool_call", "tool_result")] == ["ols_fit", "ols_fit"]
    assert text(evs).endswith("Hecho.")

    history = c.app.state.engine.chat.history
    first = next(m for m in history if isinstance(m, ModelRequest))
    assert "qtrix-econometrics" in (first.instructions or "")  # the catalog: name and description only
    assert "Method map" not in (first.instructions or "")
    guide = returns(history, "load_capability")[0]
    assert "# Skill: qtrix-econometrics" in guide and "## 2. Method map" in guide
    assert "method_details.md:374" in returns(history, "search_files")[0]
    [read] = returns(history, "read_file")
    assert "Sharp_Regression_Discontinuity_Design_regression(" in read
    # The library is all it can see: the installed skill's .git and the engine's code stay out of reach.
    git, escape = [str(p.content) for m in history if isinstance(m, ModelRequest) for p in m.parts
                   if isinstance(p, RetryPromptPart) and p.tool_name == "read_file"]
    assert "denied by pattern" in git and "outside the project root `/skills`" in escape
