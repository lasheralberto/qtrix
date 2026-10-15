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
import { Fragment, useEffect, useState, type FormEvent, type ReactNode } from "react";
import type { CellInfo, PlanStep } from "./api";
import { Markdown } from "./blocks";
import type { NextStepsApi, Turn } from "./chat";
import { T } from "./i18n";
import { PlanPanel, type PlanDraft } from "./plan";

type Phase = "running" | "done" | "issues" | "error" | "idle";

function phaseOf(turn: Turn): Phase {
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
function currentStep(turn: Turn): number {
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
}) {
  const { turns, open, onClose, plan } = props;
  const index = props.turnKey ? turns.findIndex((t) => t.key === props.turnKey) : turns.length - 1;
  const at = index < 0 ? turns.length - 1 : index;
  const turn = turns[at];
  const [tab, setTab] = useState<"plan" | "work">("work");
  const [full, setFull] = useState(false);
  const showPlan = !!plan && tab === "plan";

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
        {plan && (
          <div className="ws-tabs" role="tablist">
            {(["plan", "work"] as const).map((id) => (
              <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
                {id === "plan" ? T("plan.tabPlan") : T("plan.tabWork")}
              </button>
            ))}
          </div>
        )}
        {!showPlan && turn && (
          <>
            <p className="ws-question" title={turn.question}>
              {turn.question}
            </p>
            <Status turn={turn} />
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
            {props.steps && <NextSteps steps={props.steps} onOpenObject={props.onOpenObject} />}
            {turn ? <TurnWork key={turn.key} turn={turn} onOpenObject={props.onOpenObject} /> : <EmptyWorkspace />}
          </>
        )}
      </div>
    </aside>
  );
}

function Status({ turn }: { turn: Turn }) {
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

function TurnWork({ turn, onOpenObject }: { turn: Turn; onOpenObject(name: string, blockId?: string | null): void }) {
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
