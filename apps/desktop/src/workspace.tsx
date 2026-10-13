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
  FlaskConical,
  Lightbulb,
  ListChecks,
  LoaderCircle,
  Package,
  Route,
  Search,
  X,
} from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import type { CellInfo, PlanStep } from "./api";
import { Markdown } from "./blocks";
import type { Turn } from "./chat";

type Phase = "running" | "done" | "error" | "idle";

function phaseOf(turn: Turn): Phase {
  if (turn.running) return "running";
  if (turn.error || turn.tools.some((c) => c.ok === false)) return "error";
  return turn.text ? "done" : "idle";
}

const PHASE_LABEL: Record<Phase, string> = {
  running: "Trabajando",
  done: "Completado",
  error: "Completado con incidencias",
  idle: "Sin respuesta",
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
      {open && <div className="ws-section-body">{has ? props.children : <p className="ws-empty">{props.empty}</p>}</div>}
    </section>
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
}) {
  const { turns, open, onClose } = props;
  const index = props.turnKey ? turns.findIndex((t) => t.key === props.turnKey) : turns.length - 1;
  const at = index < 0 ? turns.length - 1 : index;
  const turn = turns[at];

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const go = (i: number) => props.onSelect(i >= turns.length - 1 ? null : turns[i].key);

  return (
    <aside className={`workspace ${open ? "open" : ""}`} aria-hidden={!open} inert={!open} aria-label="Espacio de trabajo del agente">
      <header className="ws-head">
        <div className="ws-head-row">
          <span className="ws-title">Espacio de trabajo</span>
          <span className="spacer" />
          <button className="ghost" title="Turno anterior" disabled={at <= 0} onClick={() => go(at - 1)}>
            <ChevronLeft size={15} />
          </button>
          <span className="ws-pos muted small">{turns.length ? `${at + 1}/${turns.length}` : "—"}</span>
          <button className="ghost" title="Turno siguiente" disabled={at >= turns.length - 1} onClick={() => go(at + 1)}>
            <ChevronRight size={15} />
          </button>
          <button className="ghost" title="Cerrar (Esc)" onClick={onClose}>
            <X size={15} />
          </button>
        </div>
        {turn && (
          <>
            <p className="ws-question" title={turn.question}>
              {turn.question}
            </p>
            <Status turn={turn} />
          </>
        )}
      </header>
      <div className="ws-body">
        {turn ? <TurnWork key={turn.key} turn={turn} onOpenObject={props.onOpenObject} /> : <EmptyWorkspace />}
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
            {settled}/{steps.length} pasos
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
      <p>Cuando hagas una pregunta, el agente irá anotando aquí su plan, las tareas, el progreso, los hallazgos y los resultados.</p>
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
  // Every object the turn produced, once (the last version it reached).
  const objects = [...new Map(turn.tools.filter((c) => c.ok && c.object).map((c) => [c.object!, c])).values()];

  const groups: { key: string; label: string; items: [PlanStep, number][] }[] = [
    { key: "now", label: "En curso", items: [] },
    { key: "pending", label: "Pendientes", items: [] },
    { key: "done", label: "Hechas", items: [] },
    { key: "skipped", label: "Descartadas", items: [] },
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
        title="Planificación"
        count={steps.length || (turn.scratch?.goal ? 1 : 0)}
        empty={turn.running ? "Preparando el plan…" : "Este turno no necesitó plan."}
      >
        {turn.scratch?.goal && (
          <p className="ws-goal">
            <span className="ws-label">Objetivo</span>
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

      <Section id="tasks" icon={<ListChecks size={14} />} title="Tareas" count={steps.length} empty="Sin tareas en este turno.">
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
        title="Progreso"
        count={turn.tools.length}
        empty={turn.running ? turn.status ?? "Pensando…" : "El agente respondió sin ejecutar acciones."}
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

      <Section
        id="findings"
        icon={<Search size={14} />}
        title="Hallazgos y conclusiones"
        count={conclusions.length + found.length + (turn.intro ? 1 : 0)}
        empty="Aún no hay hallazgos."
      >
        {conclusions.map((n, i) => (
          <div key={`c${i}`} className="ws-card conclusion">
            <span className="ws-label">Conclusión provisional</span>
            {n.text}
          </div>
        ))}
        {turn.intro && (
          <div className="ws-card">
            <span className="ws-label">Punto de partida</span>
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

      <Section id="ideas" icon={<Lightbulb size={14} />} title="Lluvia de ideas" count={ideas.length} empty="No se plantearon alternativas.">
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
        title="Resultados"
        count={objects.length + deliverables.length}
        empty={turn.running ? "Todavía sin resultados." : "Este turno no produjo objetos."}
      >
        {deliverables.map((n, i) => (
          <div key={`r${i}`} className="ws-card result">
            {n.text}
          </div>
        ))}
        {objects.length > 0 && (
          <div className="ws-objects">
            {objects.map((c) => (
              <button key={c.object!} className="ws-object" onClick={() => onOpenObject(c.object!, c.block_id)} title="Abrir en el panel">
                <Package size={12} />
                {c.object}
                {c.version != null && <span className="muted">v{c.version}</span>}
              </button>
            ))}
          </div>
        )}
      </Section>
    </>
  );
}
