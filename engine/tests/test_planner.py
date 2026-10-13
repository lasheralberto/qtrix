"""Plan-and-execute chat turns: the planner drafts steps, the ReAct loop settles them, and every turn ends with an
answer (the agent is sent back while steps are pending, and asked for the answer when it stops without one)."""

from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from qtrix_engine.agents.planner import Plan, resumable
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings

FIT = {"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_educ"}


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def last_prompt(messages) -> str:
    return next(p.content for m in reversed(messages) if isinstance(m, ModelRequest) for p in m.parts
                if isinstance(p, UserPromptPart) and isinstance(p.content, str))


def scripted(steps: list[str], script: list, *, final: str | None = "Resumen final.",
             draft: dict | None = None) -> FunctionModel:
    """The planner returns ``steps``; the executor plays ``script`` (one item per model response: a text, or a
    (tool, args) call); the final-answer round says ``final``."""
    script = list(script)

    def plan(messages, info: AgentInfo):
        assert info.output_tools, "only the planner and the explainer run without streaming"
        out = info.output_tools[0]
        if "findings" in out.parameters_json_schema["properties"]:  # the explainer: one finding per cell asked
            ids = re.findall(r"^### Cell (\S+)", last_prompt(messages), re.M)
            return ModelResponse(parts=[ToolCallPart(out.name, {"findings": [
                {"id": i, "finding": f"Hallazgo de {i}"} for i in ids]})])
        return ModelResponse(parts=[ToolCallPart(out.name, {"steps": steps, **(draft or {})})])

    async def run(messages, info: AgentInfo):
        if last_prompt(messages).startswith("[Write now your final answer"):
            if final is None:  # a stubborn model: tries to call a tool instead of answering
                yield {0: DeltaToolCall(name="data_list_datasets", json_args="{}")}
            else:
                yield final
            return
        item = script.pop(0) if script else ""
        if isinstance(item, tuple):
            if len(item) == 3:  # (text, tool, args): says something, then calls
                yield item[0]
                item = item[1:]
            yield {0: DeltaToolCall(name=item[0], json_args=json.dumps(item[1]))}
        else:
            yield item

    return FunctionModel(plan, stream_function=run)


def client(tmp_path, wages_csv, model, **agent):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), model_override=model,
                              with_memory=False))
    for key, value in agent.items():
        setattr(c.app.state.engine.user.agent, key, value)
    c.post("/datasets/import", json={"path": str(wages_csv)})
    return c


def text(evs) -> str:
    return "".join(e["delta"] for e in evs if e["type"] == "text")


def test_plan_is_executed_step_by_step_and_the_loop_resumes_while_steps_are_pending(tmp_path, wages_csv):
    model = scripted(["Estimar lwage sobre educ", "Comprobar heterocedasticidad"], [
        ("ols_fit", FIT), ("plan_update", {"step": 1, "note": "eq_educ"}),
        "Ahora comprobaré la heterocedasticidad.",  # stops early: step 2 is still pending
        ("plan_update", {"step": 2, "status": "skipped", "note": "sin tiempo"}),
        "El retorno de la educación es 0.08.",
    ])
    c = client(tmp_path, wages_csv, model)
    evs = events(c.post("/chat", json={"message": "retorno de la educación"}))

    plans = [e["steps"] for e in evs if e["type"] == "plan"]
    assert [s["status"] for s in plans[0]] == ["pending", "pending"]
    assert [s["status"] for s in plans[-1]] == ["done", "skipped"] and plans[-1][0]["note"] == "eq_educ"
    assert any(e["type"] == "status" and "Continuando" in e["message"] for e in evs)
    # Plan bookkeeping is not shown as Python cells; the estimate is.
    assert [e["name"] for e in evs if e["type"] == "tool_call"] == ["ols_fit"]
    assert text(evs).endswith("El retorno de la educación es 0.08.")
    turn = c.get("/chat/history").json()[-1]
    assert turn["plan"][1]["status"] == "skipped" and turn["cells"][0]["name"] == "ols_fit"


def test_a_turn_without_final_text_is_asked_for_its_answer(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], [("ols_fit", FIT), ""]))
    evs = events(c.post("/chat", json={"message": "estima"}))
    assert not any(e["type"] == "plan" for e in evs)  # nothing to plan: plain ReAct
    assert any(e["type"] == "status" and "Redactando" in e["message"] for e in evs)
    assert text(evs).strip() == "Resumen final."
    assert c.get("/chat/history").json()[-1]["content"].strip() == "Resumen final."


def test_out_of_budget_the_turn_still_ends_with_an_answer(tmp_path, wages_csv):
    loop = [("data_list_datasets", {})] * 10
    c = client(tmp_path, wages_csv, scripted(["Listar datos"], loop), max_tool_calls=3)
    evs = events(c.post("/chat", json={"message": "lista sin parar"}))
    assert evs[-1]["type"] == "done" and not any(e["type"] == "error" for e in evs)
    assert text(evs).strip() == "Resumen final."
    # The cut-short run left no unanswered tool call in the history: the next turn can be sent to the model.
    history = c.app.state.engine.chat.history
    assert history and not any(isinstance(p, ToolCallPart) for p in history[-1].parts)


def test_when_the_model_will_not_answer_the_turn_is_summarised(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], [("ols_fit", FIT), ""], final=None), planner=False)
    evs = events(c.post("/chat", json={"message": "estima"}))
    answer = text(evs)
    assert "No he podido redactar la respuesta final" in answer and "`ols_fit` ✓ → eq_educ" in answer
    # Refused tool calls in the final round are not journaled nor shown as cells.
    assert [e["name"] for e in evs if e["type"] == "tool_call"] == ["ols_fit"]


def test_a_greeting_gets_a_reply_and_no_actions(tmp_path, wages_csv):
    # The planner sees nothing to do; an eager executor still tries to describe the data on its own.
    c = client(tmp_path, wages_csv, scripted([], [("data_summary_stats", {"dataset_id": "wages"}), "¡Hola!"]))
    evs = events(c.post("/chat", json={"message": "hey"}))
    assert not any(e["type"] in ("tool_call", "tool_result") for e in evs)
    assert c.get("/panel").json()["blocks"] == [] and text(evs)
    assert not [e for e in c.app.state.engine.ctx.journal.list() if e["actor"].startswith("agent")]


def test_planner_can_be_switched_off(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted(["Paso"], ["Hecho."]), planner=False)
    evs = events(c.post("/chat", json={"message": "hola"}))
    assert not any(e["type"] in ("plan", "status") for e in evs) and text(evs) == "Hecho."


def test_each_cell_carries_the_finding_the_model_wrote_after_it(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], [
        ("Miro los datos.", "data_list_datasets", {}),
        ("Hay un dataset, wages, con lwage y educ.", "ols_fit", FIT),
        "El retorno de la educación es 0.08.",
    ]), planner=False)
    evs = events(c.post("/chat", json={"message": "retorno de la educación"}))
    calls = [e["id"] for e in evs if e["type"] == "tool_call"]
    found = [e for e in evs if e["type"] == "finding"]
    # Narration before the first call is the intro; text after a result is that cell's finding.
    assert found[0]["ids"] == [] and found[0]["text"] == "Miro los datos."
    assert found[1]["ids"] == [calls[0]] and found[1]["text"].startswith("Hay un dataset")
    # The last cell's reading went into the answer: the explainer gives it its own finding.
    assert found[2] == {"type": "finding", "ids": [calls[1]], "text": f"Hallazgo de {calls[1]}", "strip": 0}
    # What the UI keeps as the answer once findings are moved out of it.
    streamed = ""
    for e in evs:
        if e["type"] == "text":
            streamed += e["delta"]
        elif e["type"] == "finding" and e["strip"]:
            streamed = streamed[: len(streamed) - e["strip"]]
    assert streamed.strip() == "El retorno de la educación es 0.08."
    turn = c.get("/chat/history").json()[-1]
    assert turn["content"].strip() == streamed.strip() and turn["intro"] == "Miro los datos."
    assert [cell.get("finding") for cell in turn["cells"]] == [found[1]["text"], f"Hallazgo de {calls[1]}"]


def test_resumable_drops_unanswered_tool_calls():
    msgs = [ModelRequest(parts=[UserPromptPart("x")]), ModelResponse(parts=[ToolCallPart("f", {})]),
            ModelRequest(parts=[])]
    assert resumable(msgs) == msgs[:1]
    assert Plan.of(["a", " ", "b"]).render() == "1. [ ] a\n2. [ ] b"


def test_the_workspace_collects_goal_alternatives_and_the_agents_notes(tmp_path, wages_csv):
    model = scripted(["Estimar lwage sobre educ"], [
        ("scratch_note", {"section": "idea", "text": "Errores HC1: la varianza puede depender de educ."}),
        ("ols_fit", FIT), ("plan_update", {"step": 1, "note": "eq_educ"}),
        ("scratch_note", {"section": "conclusion", "text": "Cada año de educación eleva el salario un 8%."}),
        ("scratch_note", {"section": "nope", "text": "x"}),  # an unknown section is refused and retried
        ("scratch_note", {"section": "result", "text": "eq_educ estimado."}),
        "El retorno de la educación es 0.08.",
    ], draft={"goal": "Medir el retorno de la educación", "alternatives": ["OLS: datos de corte transversal",
                                                                           "IV: sin instrumento en los datos"]})
    c = client(tmp_path, wages_csv, model)
    evs = events(c.post("/chat", json={"message": "retorno de la educación"}))

    pads = [e for e in evs if e["type"] == "scratch"]
    assert pads[0]["goal"] == "Medir el retorno de la educación"
    assert [n["section"] for n in pads[0]["notes"]] == ["idea", "idea"]
    assert [n["section"] for n in pads[-1]["notes"]] == ["idea", "idea", "idea", "conclusion", "result"]
    # Workspace bookkeeping is not a Python cell and does not leak into the answer.
    assert [e["name"] for e in evs if e["type"] == "tool_call"] == ["ols_fit"]
    assert text(evs).strip().endswith("El retorno de la educación es 0.08.")
    turn = c.get("/chat/history").json()[-1]
    assert turn["scratch"]["goal"] == "Medir el retorno de la educación"
    assert turn["scratch"]["notes"][-1] == {"section": "result", "text": "eq_educ estimado."}


def test_a_turn_without_plan_has_no_workspace_until_the_agent_writes_one(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], ["Hola, ¿en qué te ayudo?"],
                                             draft={"goal": "ignorado sin pasos", "alternatives": ["x"]}))
    evs = events(c.post("/chat", json={"message": "hola"}))
    assert not any(e["type"] == "scratch" for e in evs)
    assert "scratch" not in c.get("/chat/history").json()[-1]


def test_the_planner_asks_before_acting_and_nothing_runs_until_the_user_answers(tmp_path, wages_csv):
    # The executor script would run an estimate: it must never be reached while the plan waits for answers.
    model = scripted(["Estimar"], [("ols_fit", FIT), "No debería llegar aquí."],
                     draft={"questions": ["¿Cuál es la variable de resultado?", "¿Y el tratamiento?"]})
    c = client(tmp_path, wages_csv, model)
    evs = events(c.post("/chat", json={"message": "estima el efecto del programa"}))
    assert not any(e["type"] in ("plan", "tool_call", "tool_result") for e in evs)
    answer = text(evs)
    assert "1. ¿Cuál es la variable de resultado?" in answer and "2. ¿Y el tratamiento?" in answer
    assert evs[-1]["type"] == "done" and not [e for e in c.app.state.engine.ctx.journal.list()
                                              if e["actor"].startswith("agent")]
    assert c.get("/chat/history").json()[-1]["content"].strip() == answer.strip()


def test_ask_user_ends_the_turn_and_blocks_further_tools(tmp_path, wages_csv):
    model = scripted(["Estimar", "Comprobar"], [
        ("ask_user", {"questions": ["¿Qué población te interesa?"]}),
        ("ols_fit", FIT),  # refused: the user has not answered yet
        "¿Qué población te interesa?",
        "No debería llegar aquí.",
    ])
    c = client(tmp_path, wages_csv, model)
    evs = events(c.post("/chat", json={"message": "estima el retorno"}))
    assert not any(e["type"] == "tool_call" for e in evs)  # neither ask_user nor the refused estimate is a cell
    assert not any(e["type"] == "status" and "Continuando" in e["message"] for e in evs)  # pending steps: no loop
    assert text(evs).strip() == "¿Qué población te interesa?"
    assert not [e for e in c.app.state.engine.ctx.journal.list() if e["actor"].startswith("agent")]
