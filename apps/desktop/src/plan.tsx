// Ask and Plan modes of the composer. In Plan mode a request is not run: the engine drafts a detailed plan (Markdown),
// which opens in the workspace for the user to read, edit, approve or reject. An approved plan goes with the request
// into the normal flow (Ask), which follows it.
import {
  Check,
  ChevronDown,
  CircleAlert,
  ClipboardList,
  Eye,
  LoaderCircle,
  Maximize2,
  MessageSquare,
  Minimize2,
  Pencil,
  RotateCcw,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { Components } from "react-markdown";
import { api } from "./api";
import { Markdown } from "./blocks";
import { T } from "./i18n";
import { ToolRef } from "./suggest";

export type Mode = "ask" | "plan";

const MODES: { id: Mode; icon: typeof MessageSquare }[] = [
  { id: "ask", icon: MessageSquare },
  { id: "plan", icon: ClipboardList },
];

const label = (m: Mode) => (m === "ask" ? T("plan.modeAsk") : T("plan.modePlan"));
const hint = (m: Mode) => (m === "ask" ? T("plan.modeAskHint") : T("plan.modePlanHint"));

const TOOL_NAME = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;

/** Inline code that looks like a tool name gets the tool's help on hover. */
const planComponents: Components = {
  code({ node: _node, className, children, ...rest }) {
    const text = String(children);
    if (!className && TOOL_NAME.test(text)) return <ToolRef name={text} />;
    return (
      <code className={className} {...rest}>
        {children}
      </code>
    );
  },
};

/** The mode dropdown next to the model picker. */
export function ModeSelect({ mode, onChange, disabled }: { mode: Mode; onChange(m: Mode): void; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => root.current && !root.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const Current = MODES.find((m) => m.id === mode)!.icon;
  return (
    <div className="mode" ref={root}>
      <button
        type="button"
        className={`mode-trigger ${mode}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`${T("plan.modeLabel")}: ${label(mode)}`}
        title={hint(mode)}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        <Current size={13} />
        {label(mode)}
        <ChevronDown size={13} className={open ? "rot" : ""} />
      </button>
      {open && (
        <ul className="mode-pop" role="listbox" aria-label={T("plan.modeLabel")}>
          {MODES.map(({ id, icon: Icon }) => (
            <li key={id} role="option" aria-selected={id === mode}>
              <button
                type="button"
                className={`mode-option ${id === mode ? "on" : ""}`}
                onClick={() => {
                  onChange(id);
                  setOpen(false);
                }}
              >
                <Icon size={15} />
                <span className="mode-option-text">
                  <strong>{label(id)}</strong>
                  <span>{hint(id)}</span>
                </span>
                {id === mode && <Check size={14} className="mode-check" />}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** A plan being drafted or waiting for the user's decision. ``text`` is the Markdown the user may edit. */
export interface PlanDraft {
  request: string;
  text: string;
  status: "drafting" | "ready" | "error";
  error?: string;
}

/** The plan of the conversation shown: drafted on demand, edited by the user, gone once approved or rejected. */
export function usePlan(conversation: string | null) {
  const [plan, setPlan] = useState<PlanDraft | null>(null);
  const abort = useRef<AbortController | null>(null);

  // A plan belongs to the conversation it was asked in.
  useEffect(() => {
    abort.current?.abort();
    setPlan(null);
  }, [conversation]);

  const draft = useCallback(async (request: string) => {
    const text = request.trim();
    if (!text) return false;
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setPlan({ request: text, text: "", status: "drafting" });
    try {
      const md = await api.draftPlan(text, ctl.signal);
      if (!ctl.signal.aborted) setPlan({ request: text, text: md, status: "ready" });
    } catch (err) {
      if (!ctl.signal.aborted) setPlan({ request: text, text: "", status: "error", error: (err as Error).message });
    }
    return true;
  }, []);

  const edit = useCallback((text: string) => setPlan((p) => (p ? { ...p, text } : p)), []);
  /** Drop the plan (rejected, approved or cancelled while drafting). */
  const discard = useCallback(() => {
    abort.current?.abort();
    setPlan(null);
  }, []);

  return { plan, drafting: plan?.status === "drafting", draft, edit, discard };
}

/** The dedicated plan space of the workspace: the plan to read or edit, and the decision. */
export function PlanPanel(props: {
  plan: PlanDraft;
  busy?: boolean;
  fullscreen: boolean;
  onToggleFullscreen(): void;
  onEdit(text: string): void;
  onApprove(): void;
  onReject(): void;
  onRetry(): void;
}) {
  const { plan } = props;
  const [editing, setEditing] = useState(false);
  // A new plan starts in the preview.
  useEffect(() => setEditing(false), [plan.request, plan.status === "ready"]);

  return (
    <section className={`plan-panel ${props.fullscreen ? "full" : ""}`} aria-label={T("plan.ariaLabel")}>
      <div className="plan-bar">
        <ClipboardList size={15} />
        <strong>{T("plan.title")}</strong>
        <span className="spacer" />
        {plan.status === "ready" && (
          <div className="plan-seg" role="group">
            <button type="button" className={`ghost ${editing ? "" : "on"}`} aria-pressed={!editing} onClick={() => setEditing(false)}>
              <Eye size={13} /> {T("plan.preview")}
            </button>
            <button type="button" className={`ghost ${editing ? "on" : ""}`} aria-pressed={editing} onClick={() => setEditing(true)}>
              <Pencil size={13} /> {T("plan.edit")}
            </button>
          </div>
        )}
        <button
          type="button"
          className="ghost"
          title={props.fullscreen ? T("plan.exitFullscreen") : T("plan.fullscreen")}
          aria-label={props.fullscreen ? T("plan.exitFullscreen") : T("plan.fullscreen")}
          onClick={props.onToggleFullscreen}
        >
          {props.fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
        </button>
      </div>

      <div className="plan-scroll">
        <p className="plan-request" title={plan.request}>
          <span className="ws-label">{T("plan.request")}</span>
          {plan.request}
        </p>

        {plan.status === "drafting" && (
          <div className="plan-state" role="status">
            <LoaderCircle size={20} className="spin" />
            <p>{T("plan.drafting")}</p>
            <span className="muted small">{T("plan.draftingHint")}</span>
          </div>
        )}

        {plan.status === "error" && (
          <div className="plan-state error" role="alert">
            <CircleAlert size={20} />
            <p>{T("plan.failed")}</p>
            <span className="muted small">{plan.error}</span>
          </div>
        )}

        {plan.status === "ready" &&
          (editing ? (
            <textarea
              className="plan-editor"
              aria-label={T("plan.editLabel")}
              value={plan.text}
              spellCheck={false}
              autoFocus
              onChange={(e) => props.onEdit(e.target.value)}
            />
          ) : (
            <>
              <p className="plan-hint muted small">{T("plan.reviewHint")}</p>
              <article className="plan-doc">
                <Markdown text={plan.text} components={planComponents} />
              </article>
            </>
          ))}
      </div>

      <footer className="plan-foot">
        {plan.status === "error" && (
          <button type="button" onClick={props.onRetry}>
            <RotateCcw size={14} /> {T("plan.retry")}
          </button>
        )}
        <span className="spacer" />
        <button type="button" className="ghost" title={T("plan.rejectTitle")} onClick={props.onReject}>
          <X size={14} /> {T("plan.reject")}
        </button>
        <button
          type="button"
          className="primary"
          title={T("plan.approveTitle")}
          disabled={plan.status !== "ready" || !plan.text.trim() || props.busy}
          onClick={props.onApprove}
        >
          <Check size={14} /> {T("plan.approve")}
        </button>
      </footer>
    </section>
  );
}
