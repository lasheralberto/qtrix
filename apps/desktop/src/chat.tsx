// The conversation with the lead agent, woven into the analysis panel (E7·08c): each turn is a question card with the
// agent's actions as editable Python cells (E7·08b), followed by the blocks it produced and the agent's answer.
// The composer is anchored at the bottom of the panel.
import {
  ArrowUp,
  ChevronRight,
  CircleAlert,
  LoaderCircle,
  PanelRightOpen,
  RotateCcw,
  Sparkles,
  Square,
} from "lucide-react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  chat,
  type Block,
  type CellInfo,
  type ChatEvent,
  type ChatTurn,
  type ContextUsage,
  type PlanStep,
  type Scratch,
} from "./api";
import { Markdown } from "./blocks";
import { describeAction, words } from "./actions";
import { CodeCell, TurnCode } from "./cells";

export interface Turn {
  key: string;
  question: string;
  text: string;
  tools: CellInfo[];
  /** The planner's steps for this turn, settled by the agent as it works (none for turns that needed no tools). */
  plan?: PlanStep[];
  /** The agent's workspace notes in this turn: the plan's goal, alternatives weighed, conclusions, deliverables. */
  scratch?: Scratch;
  /** What the agent said before its first action; what it read in each result goes into that cell's finding. */
  intro?: string;
  error?: string;
  status?: string;
  running: boolean;
  /** The panel block that was last when the turn started: the turn sits right after it (null = top of the panel). */
  after: string | null;
  /** The conversation filled the model's context and was folded into a summary before this turn. */
  compacted?: boolean;
}

export const SUGGESTIONS = [
  "Describe los datos y propón un modelo",
  "Estima el retorno de la educación con errores robustos",
  "¿Hay heterocedasticidad? Comprueba los supuestos",
  "Compara efectos fijos y aleatorios (Hausman)",
];

function turnsFrom(history: ChatTurn[]): Turn[] {
  const turns: Turn[] = [];
  for (const m of history) {
    if (m.role === "user") {
      turns.push({
        key: `h${turns.length}`,
        question: m.content,
        text: "",
        tools: [],
        running: false,
        after: m.after_block ?? null,
        compacted: m.compacted,
      });
      continue;
    }
    const last = turns[turns.length - 1];
    if (!last) continue;
    last.key = m.turn_id ?? last.key;
    last.text = m.content;
    last.tools =
      m.cells ??
      (m.tools ?? []).map((n, i) => ({
        id: `${i}`,
        name: n,
        args: {},
        code: null,
        ok: true,
      }));
    last.plan = m.plan;
    last.scratch = m.scratch;
    last.intro = m.intro;
  }
  return turns;
}

/** Chat state for the unified panel: the turns, whether the agent is working, and how to talk to it. */
export function useChat(
  lastBlock: () => string | null,
  /** The conversation shown ("<notebook>/<thread>"); null while it is not known yet. */
  conversation: string | null,
) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [context, setContext] = useState<ContextUsage | null>(null);
  const [compacting, setCompacting] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const busyRef = useRef(false);

  // The meter's thresholds come with GET /chat/context; turn events only update the counts.
  const refreshContext = useCallback(
    () => api.chatContext().then(setContext).catch(() => {}),
    [],
  );
  const updateContext = (c: ContextUsage) =>
    setContext((prev) => ({ ...prev, used: c.used, limit: c.limit, estimated: c.estimated }));

  // Each conversation of a notebook has its own chat: load it when the conversation changes.
  useEffect(() => {
    setTurns([]);
    setContext(null);
    if (!conversation) return;
    let cancelled = false;
    api
      .history()
      .then((h) => !cancelled && setTurns(turnsFrom(h)))
      .catch(() => {});
    refreshContext();
    return () => {
      cancelled = true;
    };
  }, [conversation, refreshContext]);

  const patchLast = (fn: (t: Turn) => Turn) =>
    setTurns((prev) =>
      prev.length ? [...prev.slice(0, -1), fn(prev[prev.length - 1])] : prev,
    );

  function handle(ev: ChatEvent) {
    switch (ev.type) {
      case "turn":
        patchLast((t) => ({
          ...t,
          key: ev.turn_id,
          after: ev.after_block ?? t.after,
        }));
        break;
      case "text":
        patchLast((t) => ({ ...t, text: t.text + ev.delta }));
        break;
      case "tool_call": {
        const { type: _, ...cell } = ev;
        // A phase message ("Planificando…") is over once the agent acts.
        patchLast((t) => ({ ...t, tools: [...t.tools, cell], status: undefined }));
        break;
      }
      case "tool_result": {
        const { type: _, ...res } = ev;
        // Keep what the call announced when the result has no value for it (the code may only come with the call).
        const filled = Object.fromEntries(
          Object.entries(res).filter(([, v]) => v !== null && v !== undefined),
        );
        patchLast((t) => ({
          ...t,
          tools: t.tools.map((c) => (c.id === ev.id ? { ...c, ...filled } : c)),
        }));
        break;
      }
      case "plan":
        patchLast((t) => ({ ...t, plan: ev.steps, status: undefined }));
        break;
      case "scratch":
        patchLast((t) => ({ ...t, scratch: { goal: ev.goal, notes: ev.notes } }));
        break;
      case "finding":
        // The text was streamed as answer: move it to the cells it explains (or to the intro).
        patchLast((t) => {
          const text = ev.strip
            ? t.text.slice(0, Math.max(0, t.text.length - ev.strip))
            : t.text;
          if (!ev.text) return { ...t, text };
          if (!ev.ids.length)
            return {
              ...t,
              text,
              intro: [t.intro, ev.text].filter(Boolean).join("\n\n"),
            };
          const tools = t.tools.map((c) =>
            ev.ids.includes(c.id)
              ? {
                  ...c,
                  finding: [c.finding, ev.text].filter(Boolean).join("\n\n"),
                }
              : c,
          );
          return { ...t, text, tools };
        });
        break;
      case "status":
        patchLast((t) => ({ ...t, status: ev.message }));
        break;
      case "error":
        patchLast((t) => ({ ...t, error: ev.message }));
        break;
      case "context":
        updateContext(ev);
        break;
      case "compacted":
        updateContext(ev);
        patchLast((t) => ({ ...t, compacted: true }));
        break;
    }
  }

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || busyRef.current) return false;
      busyRef.current = true;
      setBusy(true);
      setTurns((prev) => [
        ...prev,
        {
          key: `live${Date.now()}`,
          question,
          text: "",
          tools: [],
          running: true,
          after: lastBlock(),
        },
      ]);
      abort.current = new AbortController();
      try {
        for await (const ev of chat(question, abort.current.signal)) handle(ev);
      } catch (err) {
        if ((err as Error).name !== "AbortError")
          patchLast((t) => ({ ...t, error: String(err) }));
      } finally {
        patchLast((t) => ({ ...t, running: false }));
        busyRef.current = false;
        setBusy(false);
      }
      return true;
    },
    [lastBlock],
  );

  const stop = () => abort.current?.abort();
  const reset = () =>
    api.resetChat().then(() => {
      setTurns([]);
      refreshContext();
    });
  /** Fold the conversation into a summary now (the meter's button). */
  const compact = async () => {
    if (busyRef.current) return;
    setCompacting(true);
    try {
      setContext(await api.compactChat());
    } finally {
      setCompacting(false);
    }
  };

  return { turns, busy, send, stop, reset, context, compacting, compact, refreshContext };
}

const fmtTokens = (n: number) =>
  n >= 1_000_000 ? `${+(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${+(n / 1000).toFixed(1)}k` : `${n}`;

/** How full the model's context is: a ring next to the model picker. A click compacts the conversation now. */
export function ContextMeter({
  context,
  busy,
  compacting,
  onCompact,
}: {
  context: ContextUsage | null;
  busy: boolean;
  compacting: boolean;
  onCompact(): void;
}) {
  if (!context || !context.limit) return null;
  const share = Math.min(context.used / context.limit, 1);
  const at = context.compact_at ?? 0.8;
  const level = share >= at ? "full" : share >= at * 0.75 ? "high" : "";
  const r = 7;
  const arc = 2 * Math.PI * r;
  const auto = context.auto_compact !== false;
  const title =
    `Contexto: ${context.estimated ? "~" : ""}${context.used.toLocaleString()} de ` +
    `${context.limit.toLocaleString()} tokens (${Math.round(share * 100)} %)\n` +
    (auto
      ? `Se compacta automáticamente al ${Math.round(at * 100)} %.`
      : "La compactación automática está desactivada.") +
    "\nPulsa para compactar ahora.";
  return (
    <button
      type="button"
      className={`ctx-meter ${level}`}
      title={title}
      aria-label={title}
      onClick={onCompact}
      disabled={busy || compacting || context.used === 0}
    >
      {compacting ? (
        <LoaderCircle size={14} className="spin" />
      ) : (
        <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
          <circle className="ctx-track" cx="9" cy="9" r={r} />
          <circle
            className="ctx-fill"
            cx="9"
            cy="9"
            r={r}
            strokeDasharray={arc}
            strokeDashoffset={arc * (1 - share)}
            transform="rotate(-90 9 9)"
          />
        </svg>
      )}
      <span className="ctx-label">
        {context.estimated ? "~" : ""}
        {fmtTokens(context.used)}
        <span className="muted"> / {fmtTokens(context.limit)}</span>
      </span>
    </button>
  );
}

/** Tools whose calls are not listed among a turn's actions in the panel. */
const HIDDEN_TOOLS = new Set(["ui_add_note"]);

/** The panel block a cell produced: the one it returned, else a live view of its object. */
function blockFor(cell: CellInfo, blocks: Block[]) {
  return (
    (cell.block_id ? blocks.find((b) => b.id === cell.block_id) : undefined) ??
    (cell.object
      ? blocks.find(
          (b) => b.ref?.object === cell.object && b.ref?.view === "estimation",
        )
      : undefined)
  );
}

/** The opening card of a turn: what the user asked and, folded, every action of the agent as a Python cell. */
export function TurnHead(props: {
  turn: Turn;
  blocks: Block[];
  onOpenObject(name: string, blockId?: string | null): void;
  /** Open the agent's workspace on this turn. */
  onOpenWorkspace(): void;
}) {
  const { turn } = props;
  // The actions stay folded, live turn included: the ticker below tells what the agent is doing as it goes.
  const [open, setOpen] = useState(false);
  // Notes are already in the feed as their own blocks: their call is not shown as an action.
  const cells = turn.tools.filter((t) => !HIDDEN_TOOLS.has(t.name));
  const n = cells.length;
  return (
    <section className="turn-head">
      {turn.compacted && (
        <div className="turn-compacted" title="El agente sigue a partir de un resumen de lo anterior; aquí ves la conversación entera">
          Conversación compactada
        </div>
      )}
      <div className="turn-question">{turn.question}</div>
      {turn.intro && (
        <div className="turn-intro">
          <Markdown text={turn.intro} />
        </div>
      )}
      <WorkspaceChip turn={turn} onOpen={props.onOpenWorkspace} />
      {n > 0 && (
        <button className="turn-toggle" onClick={() => setOpen((o) => !o)}>
          <ChevronRight size={12} className={open ? "rot" : ""} />
          {n} {n === 1 ? "acción" : "acciones"} del agente · Python
          {cells.some((t) => t.ok === false) && (
            <span className="pill warn">con errores</span>
          )}
        </button>
      )}
      {open && n > 0 && (
        <div className="tools">
          {cells.map((t) => (
            <CodeCell
              key={t.id}
              cell={t}
              block={blockFor(t, props.blocks)}
              onOpenObject={props.onOpenObject}
            />
          ))}
          {!turn.running && n > 1 && <TurnCode cells={cells} />}
        </div>
      )}
      {turn.running && !turn.text && (
        <span className="status-line">
          <LoaderCircle size={14} className="spin" /> <Ticker text={liveMessage(turn, cells)} />
        </span>
      )}
    </section>
  );
}

/** What the agent is doing now, in at most 10 words: the phase, else a plain description of its latest action. */
function liveMessage(turn: Turn, cells: CellInfo[]): string {
  if (turn.status) return words(turn.status);
  const last = cells[cells.length - 1];
  return last ? describeAction(last) : "Pensando en cómo abordarlo…";
}

/** A line of text that changes by blurring out the old message and blurring in the new one. */
function Ticker({ text }: { text: string }) {
  const [shown, setShown] = useState([{ id: 0, text }]);
  const next = useRef(0);
  useEffect(() => {
    setShown((prev) =>
      prev[prev.length - 1].text === text ? prev : [...prev.slice(-1), { id: ++next.current, text }],
    );
  }, [text]);
  useEffect(() => {
    if (shown.length < 2) return;
    const t = setTimeout(() => setShown((prev) => prev.slice(-1)), 500);
    return () => clearTimeout(t);
  }, [shown]);
  return (
    <span className="ticker" aria-live="polite">
      {shown.map((m, i) => (
        <span key={m.id} className={i === shown.length - 1 ? "tick in" : "tick out"}>
          {m.text}
        </span>
      ))}
    </span>
  );
}

/** A turn's link to the agent's workspace: how far its plan has got, at a glance. */
function WorkspaceChip({ turn, onOpen }: { turn: Turn; onOpen(): void }) {
  const steps = turn.plan ?? [];
  const settled = steps.filter((s) => s.status !== "pending").length;
  const notes = turn.scratch?.notes.length ?? 0;
  if (!steps.length && !notes && !turn.running) return null;
  return (
    <button className={`ws-chip ${turn.running ? "live" : ""}`} onClick={onOpen} title="Ver el espacio de trabajo del agente">
      <PanelRightOpen size={13} />
      Espacio de trabajo
      {steps.length > 0 && (
        <span className="ws-chip-meter" aria-label={`${settled} de ${steps.length} pasos`}>
          <span style={{ width: `${(100 * settled) / steps.length}%` }} />
        </span>
      )}
      {steps.length > 0 && (
        <span className="ws-chip-count">
          {settled}/{steps.length}
        </span>
      )}
    </button>
  );
}

/** The agent's answer, after the blocks the turn produced. */
export function TurnAnswer({ turn }: { turn: Turn }) {
  if (!turn.text && !turn.error) return null;
  return (
    <article className={`block answer ${turn.running ? "streaming" : ""}`}>
      <header className="answer-head">
        <Sparkles size={13} /> Respuesta
        {turn.running && <span className="muted small">redactando…</span>}
      </header>
      <div className="block-body">
        {turn.text && <Markdown text={turn.text} />}
        {turn.error && (
          <div className="error">
            <CircleAlert size={14} /> {turn.error}
          </div>
        )}
      </div>
    </article>
  );
}

export interface ComposerHandle {
  /** Put the cursor in the composer, optionally starting the message with ``prefill``. */
  focus(prefill?: string): void;
  /** Replace the message with ``text`` and select its first «…» placeholder, so typing fills it in. */
  fill(text: string): void;
}

export const Composer = forwardRef<
  ComposerHandle,
  {
    busy: boolean;
    onSend(text: string): Promise<boolean>;
    onStop(): void;
    onReset(): void;
    modelPicker: ReactNode;
    /** The context meter, shown next to the model picker. */
    meter?: ReactNode;
  }
>(function Composer({ busy, onSend, onStop, onReset, modelPicker, meter }, ref) {
  const [input, setInput] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);

  useImperativeHandle(ref, () => ({
    focus(prefill) {
      if (prefill) setInput((v) => (v.trim() ? v : prefill));
      requestAnimationFrame(() => {
        const el = area.current;
        if (!el) return;
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
      });
    },
    fill(text) {
      setInput(text);
      requestAnimationFrame(() => {
        const el = area.current;
        if (!el) return;
        el.focus();
        const start = text.indexOf("«");
        const end = start < 0 ? -1 : text.indexOf("»", start);
        if (end < 0) el.setSelectionRange(text.length, text.length);
        else el.setSelectionRange(start, end + 1);
      });
    },
  }));

  const submit = () => {
    const text = input;
    if (!text.trim() || busy) return;
    setInput("");
    onSend(text);
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <textarea
        ref={area}
        value={input}
        placeholder="Pregunta al equipo econométrico… Ej.: estima un modelo de efectos fijos de lwage sobre exper y union"
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        rows={2}
      />
      <div className="composer-bar">
        {modelPicker}
        {meter}
        <span className="spacer" />
        <button
          type="button"
          className="ghost"
          title="Nueva conversación (el panel se conserva)"
          onClick={onReset}
          disabled={busy}
        >
          <RotateCcw size={14} />
        </button>
        {busy ? (
          <button
            type="button"
            className="send stop"
            onClick={onStop}
            title="Detener"
          >
            <Square size={14} />
          </button>
        ) : (
          <button
            type="submit"
            className="send"
            disabled={!input.trim()}
            title="Enviar"
          >
            <ArrowUp size={16} />
          </button>
        )}
      </div>
    </form>
  );
});
