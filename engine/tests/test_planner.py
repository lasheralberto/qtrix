"""Plan-and-execute chat turns: the planner drafts steps, the ReAct loop settles them, and every turn ends with an
answer (the agent is sent back while steps are pending, and asked for the answer when it stops without one)."""

from __future__ import annotations

import json
import re

from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from qtrix_engine.agents.planner import Plan, leaked_call, leaked_note, resumable
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
            concl = ["Conclusión del turno"] if "Answer written for the turn" in last_prompt(messages) else []
            return ModelResponse(parts=[ToolCallPart(out.name, {"findings": [
                {"id": i, "finding": f"Hallazgo de {i}"} for i in ids], "conclusions": concl,
                "issues": ["Heterocedasticidad probable"] if concl else [],
                "next_steps": ["Estimar [[eq_educ]] por IV.", "Revisar [[eq_fantasma]]."] if concl else []})])
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


def test_the_workspace_gets_conclusions_and_objects_even_when_the_model_wrote_none(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], [("ols_fit", FIT), "El retorno de la educación es 0.08."]),
               planner=False)
    evs = events(c.post("/chat", json={"message": "retorno de la educación"}))
    notes = [e for e in evs if e["type"] == "scratch"][-1]["notes"]
    assert {"section": "conclusion", "text": "Conclusión del turno"} in notes
    assert {"section": "issue", "text": "Heterocedasticidad probable"} in notes
    assert any(n["section"] == "result" and "Objetos creados" in n["text"] for n in notes)


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
        ("scratch_note", {"section": "issue", "text": "Posible heterocedasticidad: usar errores robustos."}),
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
    assert [n["section"] for n in pads[-1]["notes"]] == ["idea", "idea", "idea", "conclusion", "issue", "result"]
    # Workspace bookkeeping is not a Python cell and does not leak into the answer.
    assert [e["name"] for e in evs if e["type"] == "tool_call"] == ["ols_fit"]
    assert text(evs).strip().endswith("El retorno de la educación es 0.08.")
    turn = c.get("/chat/history").json()[-1]
    assert turn["scratch"]["goal"] == "Medir el retorno de la educación"
    assert turn["scratch"]["notes"][-1] == {"section": "result", "text": "eq_educ estimado."}


def test_next_steps_are_written_by_the_agent_edited_by_the_user_and_kept_across_turns(tmp_path, wages_csv):
    model = scripted(["Estimar lwage sobre educ"], [
        ("ols_fit", FIT), ("plan_update", {"step": 1}),
        ("next_step", {"text": "Comparar [[eq_educ]] con un modelo IV."}),
        ("next_step", {"text": "Revisar [[eq_fantasma]]."}),  # an object that does not exist is refused and retried
        ("next_step", {"text": "Contrastar heterocedasticidad."}),
        ("next_step_done", {"step": 2}),
        "Hecho.",
    ])
    c = client(tmp_path, wages_csv, model)
    evs = events(c.post("/chat", json={"message": "estima"}))

    lists = [e["items"] for e in evs if e["type"] == "next_steps"]
    assert [s["text"] for s in lists[-1]] == ["Comparar [[eq_educ]] con un modelo IV.",
                                              "Contrastar heterocedasticidad."]
    assert lists[-1][0]["objects"] == ["eq_educ"] and [s["done"] for s in lists[-1]] == [False, True]
    assert [e["name"] for e in evs if e["type"] == "tool_call"] == ["ols_fit"]  # bookkeeping is not a cell
    assert [s["id"] for s in c.get("/chat/next-steps").json()] == ["n1", "n2"]

    added = c.post("/chat/next-steps", json={"text": "Probar [[eq_educ]] con errores HC3"}).json()
    assert added[-1]["author"] == "user" and added[-1]["id"] == "n3" and added[-1]["objects"] == ["eq_educ"]
    assert c.post("/chat/next-steps", json={"text": "  "}).status_code == 400
    edited = c.patch("/chat/next-steps/n2", json={"done": False}).json()
    assert edited[1]["done"] is False
    assert [s["id"] for s in c.delete("/chat/next-steps/n1").json()] == ["n2", "n3"]
    assert c.delete("/chat/next-steps/n9").status_code == 404

    # They survive the session (they are saved with the conversation) and the agent reads them every turn.
    chat = c.app.state.engine.chat
    from qtrix_engine.agents.lead import ChatSession, workspace_context

    saved = ChatSession()
    saved.restore([], [], None, chat.next_steps.dump())
    assert [s["text"] for s in saved.next_steps.items] == [s["text"] for s in chat.next_steps.items]
    assert "## Next steps of this conversation" in workspace_context()
    chat.reset()
    assert c.get("/chat/next-steps").json() == []


def test_the_executor_is_offered_the_next_step_tools(tmp_path, wages_csv):
    offered: list[str] = []

    async def run(messages, info: AgentInfo):
        offered.extend(t.name for t in info.function_tools)
        yield "Hola."

    model = scripted(["Estimar"], [])
    c = client(tmp_path, wages_csv, FunctionModel(model.function, stream_function=run))
    c.post("/chat", json={"message": "estima"})
    assert {"next_step", "next_step_done"} <= set(offered)


def test_a_turn_whose_agent_left_no_next_steps_gets_them_from_its_answer(tmp_path, wages_csv):
    model = scripted(["Estimar lwage sobre educ"], [
        ("ols_fit", FIT), ("plan_update", {"step": 1}), "El retorno de la educación es 0.08 en [[eq_educ]].",
    ])
    c = client(tmp_path, wages_csv, model)
    evs = events(c.post("/chat", json={"message": "estima"}))

    lists = [e["items"] for e in evs if e["type"] == "next_steps"]
    # A reference to an object that does not exist loses its brackets: it would be a dead link.
    assert [s["text"] for s in lists[-1]] == ["Estimar [[eq_educ]] por IV.", "Revisar eq_fantasma."]
    assert lists[-1][0]["objects"] == ["eq_educ"] and lists[-1][1]["objects"] == []
    # The next turn does not add them twice.
    c.post("/chat", json={"message": "estima otra vez"})
    assert len(c.get("/chat/next-steps").json()) == 2


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


PLAN_MD = "## Objetivo\nEstimar el retorno a la educación\n\n## Pasos\n1. OLS"


def drafting(seen: list[str], plan_md: str = PLAN_MD) -> FunctionModel:
    """The plan drafter (plain text) writes ``plan_md``; the planner (an output tool) sees nothing to do; every
    prompt received is appended to ``seen``."""

    def respond(messages, info: AgentInfo):
        seen.append(last_prompt(messages))
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"steps": []})])
        return ModelResponse(parts=[TextPart(plan_md)])

    async def talk(messages, info: AgentInfo):
        yield "Hecho."

    return FunctionModel(respond, stream_function=talk)


def test_plan_mode_drafts_a_plan_and_runs_nothing(tmp_path, wages_csv):
    seen: list[str] = []
    c = client(tmp_path, wages_csv, drafting(seen))
    r = c.post("/chat/plan", json={"message": "estima el retorno a la educación"})
    assert r.status_code == 200 and "## Objetivo" in r.json()["plan"]
    assert "estima el retorno a la educación" in seen[0]
    # The conversation is untouched and the agent did nothing: the plan waits for the user's approval.
    assert c.get("/chat/history").json() == []
    assert not [e for e in c.app.state.engine.ctx.journal.list() if e["actor"].startswith("agent")]


def test_an_approved_plan_goes_with_the_request_but_not_into_the_transcript(tmp_path, wages_csv):
    seen: list[str] = []
    c = client(tmp_path, wages_csv, drafting(seen))
    plan = "1. Estimar OLS de lwage sobre educ con HC1"
    evs = events(c.post("/chat", json={"message": "estima el retorno", "plan": plan}))
    assert evs[-1]["type"] == "done"
    planner_prompt = seen[0]
    assert "Plan approved by the user" in planner_prompt and plan in planner_prompt
    assert c.get("/chat/history").json()[0]["content"] == "estima el retorno"  # the question stays short


def test_without_an_approved_plan_the_request_is_unchanged(tmp_path, wages_csv):
    seen: list[str] = []
    c = client(tmp_path, wages_csv, drafting(seen))
    events(c.post("/chat", json={"message": "hola"}))
    assert "Plan approved by the user" not in seen[0]


LEAK = "scratch_note{section:conclusion,text:El modelo explica un 54.4% de la varianza salarial.}"


def test_a_tool_call_written_as_text_is_not_the_answer():
    name, body = leaked_call(LEAK)
    assert name == "scratch_note" and body.startswith("section:conclusion,text:")
    assert leaked_note(body) == "El modelo explica un 54.4% de la varianza salarial."
    assert leaked_call("scratch_note es una herramienta {pero esto es prosa}") is None
    assert leaked_call("variable_inventada{a:1}") is None


def test_a_final_answer_that_is_a_leaked_note_becomes_the_note_text(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], [LEAK]))
    evs = events(c.post("/chat", json={"message": "hey"}))
    assert any(e["type"] == "finding" and e["strip"] == len(LEAK) and not e["text"] for e in evs)  # raw text removed
    answer = c.get("/chat/history").json()[-1]
    assert answer["content"].strip() == "El modelo explica un 54.4% de la varianza salarial."
    assert {"section": "conclusion", "text": answer["content"].strip()} in answer["scratch"]["notes"]


def test_a_leaked_call_of_another_tool_gives_the_fallback_answer(tmp_path, wages_csv):
    c = client(tmp_path, wages_csv, scripted([], ['data_list_datasets{}']))
    evs = events(c.post("/chat", json={"message": "hey"}))
    assert "data_list_datasets{" not in c.get("/chat/history").json()[-1]["content"]
    assert evs[-1]["type"] == "done"
