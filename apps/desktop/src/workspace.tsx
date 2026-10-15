// The agent's workspace (E8): a sliding side panel where the agent's work in a turn is laid out while it happens —
// the plan, its tasks, the progress of its actions, what it found, the alternatives it weighed and what it produced.
// Plan, tasks and progress come from the plan and the cells; the agent adds short notes with scratch_note.
// The final answer is not here: it closes the turn in the analysis panel, as its own card.
import {
  ChevronLeft,
  ChevronRight,
  Circle,
  CircleAlert,
  CircleCheck,
  CircleMinus,
  Compass,
  TriangleAlert,
  FlaskConical,
  Lightbulb,
  ListChecks,
  ListTodo,
  LoaderCircle,
  Package,
  Route,
  Search,
  X,
} from "lucide-react";
import { Fragment, useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import type { CellInfo, PlanStep } from "./api";
import { Markdown } from "./blocks";
import type { AgentWork, NextStepsApi, Turn, Work } from "./chat";
import { T } from "./i18n";
import { PlanPanel, type PlanDraft } from "./plan";

type Phase = "running" | "done" | "issues" | "error" | "idle";

function phaseOf(turn: Work): Phase {
  if (turn.running) return "running";
  if (turn.error) return "error";
  if (turn.scratch?.notes.some((n) => n.section === "issue")) return "issues";
  return turn.text ? "done" : "idle";
}

const PHASE_LABEL: Record<Phase, string> = {
  running: T("workspace.phaseRunning"),
  done: T("workspace.phaseDone"),
  issues: T("workspace.phaseIssues"),
  error: T("workspace.phaseError"),
  idle: T("workspace.phaseIdle"),
};

/** The index of the step in progress: the first pending one while the agent works. */
function currentStep(turn: Work): number {
  return turn.running ? (turn.plan ?? []).findIndex((s) => s.status === "pending") : -1;
}

function StepIcon({ step, current }: { step: PlanStep; current: boolean }) {
  if (current) return <LoaderCircle size={13} className="spin" />;
  if (step.status === "done") return <CircleCheck size={13} />;
  if (step.status === "skipped") return <CircleMinus size={13} />;
  return <Circle size={13} />;
}

function Section(props: {
  id: string;
  icon: ReactNode;
  title: string;
  count?: number;
  empty: string;
  children?: ReactNode;
  /** Shown whenever the section is open, even while it is empty. */
  footer?: ReactNode;
}) {
  const has = Boolean(props.count);
  const [open, setOpen] = useState(true);
  return (
    <section className={`ws-section ${has ? "" : "empty"}`} aria-labelledby={`ws-${props.id}`}>
      <button className="ws-section-head" id={`ws-${props.id}`} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="ws-icon">{props.icon}</span>
        <span className="ws-section-title">{props.title}</span>
        {has && <span className="ws-count">{props.count}</span>}
        <ChevronRight size={13} className={`ws-caret ${open ? "rot" : ""}`} />
      </button>
      {open && (
        <div className="ws-section-body">
          {has ? props.children : <p className="ws-empty">{props.empty}</p>}
          {props.footer}
        </div>
      )}
    </section>
  );
}

const OBJECT_REF = /\[\[\s*([A-Za-z][A-Za-z0-9_]{0,31})\s*\]\]/g;

/** A step's text with its ``[[object]]`` references as buttons that open the object in the panel. */
function StepText({ text, onOpenObject }: { text: string; onOpenObject(name: string): void }) {
  const parts = text.split(OBJECT_REF); // odd positions are the captured names
  return (
    <span>
      {parts.map((p, i) =>
        i % 2 ? (
          <button key={i} className="ws-object ws-ref" onClick={() => onOpenObject(p.toLowerCase())} title={T("workspace.openInPanel")}>
            <Package size={11} />
            {p}
          </button>
        ) : (
          <Fragment key={i}>{p}</Fragment>
        ),
      )}
    </span>
  );
}

/** What is left to do in the whole conversation: the agent leaves follow-ups, the user ticks, deletes and adds. */
function NextSteps({ steps, onOpenObject }: { steps: NextStepsApi; onOpenObject(name: string): void }) {
  const [draft, setDraft] = useState("");
  const open = steps.items.filter((s) => !s.done).length;
  const add = (e: FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    steps.add(text).catch(() => setDraft(text));
  };
  return (
    <Section
      id="next"
      icon={<ListTodo size={14} />}
      title={T("workspace.nextSteps")}
      count={open}
      empty={T("workspace.noNextSteps")}
      footer={
        <form className="ws-next-add" onSubmit={add}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder={T("workspace.addNextStep")}
            aria-label={T("workspace.addNextStep")}
          />
        </form>
      }
    >
      <ul className="ws-tasks ws-next">
        {steps.items.map((s) => (
          <li key={s.id} className={`ws-task ${s.done ? "done" : ""} ${s.author}`}>
            <button
              className="ws-next-check"
              role="checkbox"
              aria-checked={s.done}
              title={s.done ? T("workspace.reopenStep") : T("workspace.doneStep")}
              onClick={() => steps.toggle(s.id, !s.done)}
            >
              {s.done ? <CircleCheck size={13} /> : <Circle size={13} />}
            </button>
            <StepText text={s.text} onOpenObject={onOpenObject} />
            <button className="ws-next-del ghost" title={T("workspace.deleteStep")} onClick={() => steps.remove(s.id)}>
              <X size={12} />
            </button>
          </li>
        ))}
      </ul>
    </Section>
  );
}

/** One action of the agent in the progress timeline. */
function ProgressItem({ cell }: { cell: CellInfo }) {
  const state = cell.ok === undefined ? "run" : cell.ok ? "ok" : "fail";
  return (
    <li className={`ws-tick ${state}`}>
      <span className="ws-tick-dot">
        {state === "run" ? <LoaderCircle size={11} className="spin" /> : state === "ok" ? <CircleCheck size={11} /> : <CircleAlert size={11} />}
      </span>
      <code>{cell.name}</code>
      {cell.object && <span className="ws-tick-obj">→ {cell.object}</span>}
    </li>
  );
}

export function Workspace(props: {
  open: boolean;
  turns: Turn[];
  /** The turn shown, by key; null follows the latest one. */
  turnKey: string | null;
  onSelect(key: string | null): void;
  onClose(): void;
  onOpenObject(name: string, blockId?: string | null): void;
  /** The conversation's next steps: they stay while the user moves between turns. */
  steps?: NextStepsApi;
  /** Plan mode: the plan waiting for the user's decision; it gets a space of its own beside the agent's work. */
  plan?: {
    draft: PlanDraft;
    /** The agent is working on another turn: the plan can wait but not start yet. */
    busy?: boolean;
    onEdit(text: string): void;
    onApprove(): void;
    onReject(): void;
    onRetry(): void;
  } | null;
  /** The team's default member (the lead), whose tab carries its name. */
  lead?: { name: string; emoji: string };
}) {
  const { turns, open, onClose, plan } = props;
  const index = props.turnKey ? turns.findIndex((t) => t.key === props.turnKey) : turns.length - 1;
  const at = index < 0 ? turns.length - 1 : index;
  const turn = turns[at];
  // "plan", "work" (the lead alone), or, once the team has worked: "lead" and each member's id.
  const [tab, setTab] = useState("work");
  const [full, setFull] = useState(false);
  // Each member's tab accumulates its work and conclusions over every turn up to the one shown.
  const merged = useMemo(() => mergeAgents(turns.slice(0, at + 1)), [turns, at]);
  const team = merged.length ? merged : null;
  const workTabs = team ? ["lead", ...team.map((a) => a.id)] : ["work"];
  const tabs = plan ? ["plan", ...workTabs] : workTabs;
  // A tab of another turn (a member that did not work in this one) falls back to the turn's first tab.
  const active = tabs.includes(tab) ? tab : workTabs[0];
  const showPlan = !!plan && active === "plan";
  const member = team?.find((a) => a.id === active);
  const shown: Work | undefined = member ?? turn;

  const tabLabel = (id: string): ReactNode => {
    if (id === "plan") return T("plan.tabPlan");
    if (id === "work") return T("plan.tabWork");
    if (id === "lead") return props.lead ? `${props.lead.emoji} ${props.lead.name}`.trim() : T("team.leadTab");
    const a = team!.find((m) => m.id === id)!;
    return (
      <>
        {a.running ? <LoaderCircle size={11} className="spin" /> : a.emoji && <span aria-hidden>{a.emoji}</span>}
        {a.name}
      </>
    );
  };

  // A new plan opens its tab; without a plan there is only the agent's work.
  const planKey = plan ? plan.draft.request : null;
  useEffect(() => {
    setTab(planKey ? "plan" : "work");
    if (!planKey) setFull(false);
  }, [planKey]);
  useEffect(() => {
    if (!open) setFull(false);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    // Escape leaves full screen first, then closes the workspace.
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && (full ? setFull(false) : onClose());
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose, full]);

  const go = (i: number) => props.onSelect(i >= turns.length - 1 ? null : turns[i].key);

  return (
    <aside
      className={`workspace ${open ? "open" : ""} ${full && showPlan ? "full" : ""}`}
      aria-hidden={!open}
      inert={!open}
      aria-label={T("workspace.ariaLabel")}
    >
      <header className="ws-head">
        <div className="ws-head-row">
          <span className="ws-title">{T("workspace.title")}</span>
          <span className="spacer" />
          {!showPlan && (
            <>
              <button className="ghost" title={T("workspace.prevTurn")} disabled={at <= 0} onClick={() => go(at - 1)}>
                <ChevronLeft size={15} />
              </button>
              <span className="ws-pos muted small">{turns.length ? `${at + 1}/${turns.length}` : "—"}</span>
              <button className="ghost" title={T("workspace.nextTurn")} disabled={at >= turns.length - 1} onClick={() => go(at + 1)}>
                <ChevronRight size={15} />
              </button>
            </>
          )}
          <button className="ghost" title={T("workspace.close")} onClick={onClose}>
            <X size={15} />
          </button>
        </div>
        {tabs.length > 1 && (
          <div className={`ws-tabs ${tabs.length > 2 ? "many" : ""}`} role="tablist">
            {tabs.map((id) => (
              <button
                key={id}
                role="tab"
                aria-selected={active === id}
                className={active === id ? "on" : ""}
                title={team?.find((a) => a.id === id)?.task}
                onClick={() => setTab(id)}
              >
                {tabLabel(id)}
              </button>
            ))}
          </div>
        )}
        {!showPlan && turn && shown && (
          <>
            <p className="ws-question" title={member ? member.task : turn.question}>
              {member && <span className="ws-label">{T("team.task")}</span>}
              {member ? member.task : turn.question}
            </p>
            <Status turn={shown} />
          </>
        )}
      </header>
      <div className={`ws-body ${showPlan ? "plan" : ""}`}>
        {showPlan ? (
          <PlanPanel
            plan={plan.draft}
            busy={plan.busy}
            fullscreen={full}
            onToggleFullscreen={() => setFull((f) => !f)}
            onEdit={plan.onEdit}
            onApprove={plan.onApprove}
            onReject={plan.onReject}
            onRetry={plan.onRetry}
          />
        ) : (
          <>
            {props.steps && !member && <NextSteps steps={props.steps} onOpenObject={props.onOpenObject} />}
            {member && <MemberConclusions agent={member} />}
            {turn && shown ? (
              <TurnWork key={`${turn.key}/${active}`} turn={shown} onOpenObject={props.onOpenObject} />
            ) : (
              <EmptyWorkspace />
            )}
          </>
        )}
      </div>
    </aside>
  );
}

/** A member's conclusions so far: one stretch per turn it worked in, headed by the question of that turn. */
function MemberConclusions({ agent: a }: { agent: AgentWork }) {
  const text = a.text || (a.ok === false ? T("team.memberFailed") : T("team.noConclusions"));
  return (
    <div className={`ws-card ${a.ok === false ? "issue" : "conclusion"}`}>
      <span className="ws-label">{T("team.conclusionsSoFar")}</span>
      {a.running ? (
        <p className="ws-now">
          <LoaderCircle size={12} className="spin" /> {a.status ?? T("workspace.thinking")}
        </p>
      ) : (
        <Markdown text={text} />
      )}
    </div>
  );
}

/** The team as the turns up to the one shown leave it: each member once, in order of first appearance, with the
 * work of every turn it took part in folded into one — cells and notes appended, the latest plan and task, and its
 * conclusions joined turn by turn under the question that asked for them. */
function mergeAgents(turns: Turn[]): AgentWork[] {
  const out = new Map<string, AgentWork>();
  for (const t of turns) {
    for (const a of t.agents ?? []) {
      // Cell ids are unique within a turn only: prefix them so the merged timeline keeps them apart.
      const tools = a.tools.map((c) => ({ ...c, id: `${t.key}:${c.id}` }));
      const stretch = a.text ? `**${t.question}**\n\n${a.text}` : "";
      const prev = out.get(a.id);
      if (!prev) {
        out.set(a.id, { ...a, tools, text: stretch });
        continue;
      }
      out.set(a.id, {
        ...a,
        tools: [...prev.tools, ...tools],
        text: [prev.text, stretch].filter(Boolean).join("\n\n---\n\n"),
        plan: a.plan ?? prev.plan,
        intro: a.intro ?? prev.intro,
        scratch:
          a.scratch || prev.scratch
            ? { goal: a.scratch?.goal || prev.scratch?.goal || "", notes: [...(prev.scratch?.notes ?? []), ...(a.scratch?.notes ?? [])] }
            : undefined,
      });
    }
  }
  return [...out.values()];
}

function Status({ turn }: { turn: Work }) {
  const phase = phaseOf(turn);
  const steps = turn.plan ?? [];
  const settled = steps.filter((s) => s.status !== "pending").length;
  const pct = steps.length ? (100 * settled) / steps.length : phase === "running" ? 8 : 100;
  return (
    <div className={`ws-status ${phase}`} role="status" aria-live="polite">
      <div className="ws-status-row">
        <span className="ws-dot" />
        <span className="ws-phase">{PHASE_LABEL[phase]}</span>
        {phase === "running" && turn.status && <span className="ws-sub">· {turn.status.replace(/…$/, "")}</span>}
        <span className="spacer" />
        {steps.length > 0 && (
          <span className="ws-sub">
            {T("workspace.stepsProgress", { done: settled, total: steps.length })}
          </span>
        )}
      </div>
      <span className="ws-meter">
        <span style={{ width: `${pct}%` }} />
      </span>
    </div>
  );
}

function EmptyWorkspace() {
  return (
    <div className="ws-blank">
      <Compass size={22} />
      <p>{T("workspace.emptyHint")}</p>
    </div>
  );
}

function TurnWork({ turn, onOpenObject }: { turn: Work; onOpenObject(name: string, blockId?: string | null): void }) {
  const steps = turn.plan ?? [];
  const current = currentStep(turn);
  const notes = turn.scratch?.notes ?? [];
  const ideas = notes.filter((n) => n.section === "idea");
  const conclusions = notes.filter((n) => n.section === "conclusion");
  const deliverables = notes.filter((n) => n.section === "result");
  const found = turn.tools.filter((c) => c.finding);
  // Caveats the analyst must keep in mind (failed assumptions, weak identification...); not the agent's own errors.
  const issues = notes.filter((n) => n.section === "issue");
  // Every object the turn produced, once (the last version it reached).
  const objects = [
    ...new Map(
      turn.tools
        .filter((c) => c.ok)
        .flatMap((c) => [
          ...(c.object ? [{ name: c.object, version: c.version, block_id: c.block_id }] : []),
          ...(c.objects ?? []).map((o) => ({ name: o.name, version: o.version, block_id: null })),
        ])
        .map((o) => [o.name, o]),
    ).values(),
  ];

  const groups: { key: string; label: string; items: [PlanStep, number][] }[] = [
    { key: "now", label: T("workspace.groupNow"), items: [] },
    { key: "pending", label: T("workspace.groupPending"), items: [] },
    { key: "done", label: T("workspace.groupDone"), items: [] },
    { key: "skipped", label: T("workspace.groupSkipped"), items: [] },
  ];
  steps.forEach((s, i) => {
    const g = i === current ? 0 : s.status === "pending" ? 1 : s.status === "done" ? 2 : 3;
    groups[g].items.push([s, i]);
  });

  return (
    <>
      <Section
        id="plan"
        icon={<Route size={14} />}
        title={T("workspace.planning")}
        count={steps.length || (turn.scratch?.goal ? 1 : 0)}
        empty={turn.running ? T("workspace.preparingPlan") : T("workspace.noPlan")}
      >
        {turn.scratch?.goal && (
          <p className="ws-goal">
            <span className="ws-label">{T("workspace.goal")}</span>
            {turn.scratch.goal}
          </p>
        )}
        {steps.length > 0 && (
          <ol className="ws-outline">
            {steps.map((s, i) => (
              <li key={i}>{s.text}</li>
            ))}
          </ol>
        )}
      </Section>

      <Section id="tasks" icon={<ListChecks size={14} />} title={T("workspace.tasks")} count={steps.length} empty={T("workspace.noTasks")}>
        {groups
          .filter((g) => g.items.length)
          .map((g) => (
            <div key={g.key} className={`ws-group ${g.key}`}>
              <span className="ws-label">{g.label}</span>
              <ul className="ws-tasks">
                {g.items.map(([s, i]) => (
                  <li key={i} className={`ws-task ${s.status} ${i === current ? "now" : ""}`}>
                    <StepIcon step={s} current={i === current} />
                    <span>
                      {s.text}
                      {s.note && <span className="ws-note"> · {s.note}</span>}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
      </Section>

      <Section
        id="progress"
        icon={<FlaskConical size={14} />}
        title={T("workspace.progress")}
        count={turn.tools.length}
        empty={turn.running ? turn.status ?? T("workspace.thinking") : T("workspace.noActions")}
      >
        <ul className="ws-timeline">
          {turn.tools.map((c) => (
            <ProgressItem key={c.id} cell={c} />
          ))}
        </ul>
        {turn.running && turn.status && (
          <p className="ws-now">
            <LoaderCircle size={12} className="spin" /> {turn.status}
          </p>
        )}
      </Section>

      {issues.length > 0 && (
        <Section id="issues" icon={<TriangleAlert size={14} />} title={T("workspace.issues")} count={issues.length} empty="">
          {issues.map((n, i) => (
            <div key={i} className="ws-card issue">
              {n.text}
            </div>
          ))}
        </Section>
      )}

      <Section
        id="findings"
        icon={<Search size={14} />}
        title={T("workspace.findings")}
        count={conclusions.length + found.length + (turn.intro ? 1 : 0)}
        empty={T("workspace.noFindings")}
      >
        {conclusions.map((n, i) => (
          <div key={`c${i}`} className="ws-card conclusion">
            <span className="ws-label">{T("workspace.provisionalConclusion")}</span>
            {n.text}
          </div>
        ))}
        {turn.intro && (
          <div className="ws-card">
            <span className="ws-label">{T("workspace.startingPoint")}</span>
            <Markdown text={turn.intro} />
          </div>
        )}
        {found.map((c) => (
          <div key={c.id} className="ws-card">
            <span className="ws-label">
              <code>{c.name}</code>
              {c.object && <> · {c.object}</>}
            </span>
            <Markdown text={c.finding!} />
          </div>
        ))}
      </Section>

      <Section id="ideas" icon={<Lightbulb size={14} />} title={T("workspace.brainstorm")} count={ideas.length} empty={T("workspace.noAlternatives")}>
        <ul className="ws-ideas">
          {ideas.map((n, i) => {
            const cut = n.text.indexOf(":");
            return (
              <li key={i}>
                {cut > 0 && cut < 60 ? (
                  <>
                    <strong>{n.text.slice(0, cut)}</strong>
                    {n.text.slice(cut)}
                  </>
                ) : (
                  n.text
                )}
              </li>
            );
          })}
        </ul>
      </Section>

      <Section
        id="results"
        icon={<Package size={14} />}
        title={T("workspace.results")}
        count={objects.length + deliverables.length}
        empty={turn.running ? T("workspace.noResultsYet") : T("workspace.noObjects")}
      >
        {deliverables.map((n, i) => (
          <div key={`r${i}`} className="ws-card result">
            {n.text}
          </div>
        ))}
        {objects.length > 0 && (
          <div className="ws-objects">
            {objects.map((o) => (
              <button key={o.name} className="ws-object" onClick={() => onOpenObject(o.name, o.block_id)} title={T("workspace.openInPanel")}>
                <Package size={12} />
                {o.name}
                {o.version != null && <span className="muted">v{o.version}</span>}
              </button>
            ))}
          </div>
        )}
      </Section>
    </>
  );
}
