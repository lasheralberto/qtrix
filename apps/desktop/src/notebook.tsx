// What the user writes in the panel (E7·08d): the ribbon under the title, notes and Python cells.
import { Check, CircleAlert, Code2, LoaderCircle, NotebookPen, Play, Plus } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { api, type Block, type KernelOutput, type PanelCellRun } from "./api";
import { Markdown, PlotlyFigure } from "./blocks";
import { CodeEditor } from "./cells";
import { T } from "./i18n";

const ANSI = /\u001b\[[0-9;]*m/g;

/** HTML a cell displayed (a DataFrame, a styled table): inert and isolated, scripts never run. */
function HtmlOutput({ html }: { html: string }) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(80);
  const doc = `<!doctype html><html><head><meta charset="utf-8"><style>
    body{margin:0;font:12.5px/1.45 Inter,"Segoe UI",system-ui,sans-serif;color:#18202f}
    table{border-collapse:collapse}th,td{padding:3px 10px;border-bottom:1px solid #e3e7ef;text-align:right;white-space:nowrap}
    thead th{border-bottom:1px solid #d2d8e4;background:#f9fafc}tbody th{text-align:left}
  </style></head><body>${html}</body></html>`;
  return (
    <iframe
      ref={frame}
      className="cell-html"
      sandbox="allow-same-origin"
      srcDoc={doc}
      style={{ height }}
      onLoad={() => {
        const body = frame.current?.contentDocument?.body;
        if (body) setHeight(Math.min(body.scrollHeight + 4, 520));
      }}
    />
  );
}

function Output({ out }: { out: KernelOutput }) {
  if (out.type === "stream") return <pre className={`cell-raw ${out.name === "stderr" ? "stderr" : ""}`}>{out.text}</pre>;
  const d = out.data;
  if (d["application/vnd.plotly.v1+json"]) return <PlotlyFigure data={d["application/vnd.plotly.v1+json"]} />;
  if (d["image/png"]) return <img className="cell-img" alt={T("notebook.figure")} src={`data:image/png;base64,${d["image/png"]}`} />;
  if (d["image/svg+xml"])
    return <img className="cell-img" alt={T("notebook.figure")} src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(d["image/svg+xml"])}`} />;
  if (d["text/html"]) return <HtmlOutput html={d["text/html"]} />;
  if (d["text/markdown"]) return <Markdown text={d["text/markdown"]} />;
  if (d["application/json"]) return <pre className="cell-raw">{JSON.stringify(d["application/json"], null, 2)}</pre>;
  return <pre className="cell-raw">{d["text/plain"] ?? ""}</pre>;
}

/** The outcome of a panel cell: kernel outputs, or the qx calls it ran, and what it changed. */
export function CellOutcome({ run }: { run: PanelCellRun }) {
  if (run.kind === "invalid") return <div className="cell-note fail">{run.error}</div>;
  const changed = run.versions.length > 0 && (
    <div className="cell-note ok">
      <Check size={13} />
      <span>
        {run.versions.map((v) => `${v.name} → v${v.version}`).join(", ")}
        {run.stale.length > 0 && <> · {T("notebook.staleList", { names: run.stale.join(", ") })}</>}
      </span>
    </div>
  );
  if (run.kind === "api")
    return (
      <>
        {run.calls.map((c, i) => (
          <div key={i} className={`cell-note ${c.ok || c.skipped ? "" : "fail"}`}>
            {c.ok ? <Check size={13} /> : <CircleAlert size={13} />}
            <span>
              <code>{c.tool}</code> {c.ok ? T("notebook.executed") : c.skipped ? T("notebook.notExecuted") : c.error}
            </span>
          </div>
        ))}
        {changed}
      </>
    );
  const err = run.error && typeof run.error === "object" ? run.error : null;
  return (
    <>
      {(run.outputs?.length ?? 0) > 0 && (
        <div className="cell-outputs">
          {run.outputs!.map((o, i) => (
            <Output key={i} out={o} />
          ))}
        </div>
      )}
      {run.error && (
        <div className="cell-note fail cell-error">
          <CircleAlert size={13} />
          {err ? (
            <pre>{(err.traceback?.length ? err.traceback.join("\n") : `${err.ename}: ${err.evalue}`).replace(ANSI, "")}</pre>
          ) : (
            <span>{String(run.error)}</span>
          )}
        </div>
      )}
      {run.ok && !run.outputs?.length && !run.versions.length && <div className="cell-note">{T("notebook.executedNoOutput")}</div>}
      {changed}
    </>
  );
}

/** Ctrl+Enter saves, Esc cancels: the keys every editor of the panel shares. */
function editorKeys(onSave: () => void, onCancel: () => void) {
  return (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      onSave();
    }
  };
}

export function NoteEditor(props: { title?: string; markdown?: string; onSave(title: string, markdown: string): Promise<unknown>; onCancel(): void }) {
  const [title, setTitle] = useState(props.title ?? "");
  const [text, setText] = useState(props.markdown ?? "");
  const [preview, setPreview] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    area.current?.focus();
  }, []);

  const save = async () => {
    if (!text.trim() || saving) return;
    setSaving(true);
    try {
      await props.onSave(title.trim() || T("notebook.note"), text);
    } catch (e) {
      setError((e as Error).message);
      setSaving(false);
    }
  };

  return (
    <div className="note-editor" onKeyDown={editorKeys(save, props.onCancel)}>
      <input className="note-title" placeholder={T("notebook.noteTitle")} value={title} onChange={(e) => setTitle(e.target.value)} />
      {preview ? (
        <div className="note-preview">{text.trim() ? <Markdown text={text} /> : <span className="muted">{T("notebook.nothingToPreview")}</span>}</div>
      ) : (
        <textarea
          ref={area}
          rows={6}
          placeholder={T("notebook.markdownPlaceholder")}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      )}
      {error && <div className="error">{error}</div>}
      <div className="editor-bar">
        <div className="tabs-mini">
          <button type="button" className={!preview ? "on" : ""} onClick={() => setPreview(false)}>
            {T("notebook.write")}
          </button>
          <button type="button" className={preview ? "on" : ""} onClick={() => setPreview(true)}>
            {T("notebook.preview")}
          </button>
        </div>
        <span className="spacer" />
        <span className="muted small">{T("notebook.saveHint")}</span>
        <button type="button" className="ghost" onClick={props.onCancel}>
          {T("notebook.cancel")}
        </button>
        <button type="button" className="primary" onClick={save} disabled={!text.trim() || saving}>
          {saving ? <LoaderCircle size={14} className="spin" /> : <Check size={14} />} {T("notebook.save")}
        </button>
      </div>
    </div>
  );
}

/** A Python editor that runs its code: the cell being written from the ribbon, or a saved cell being re-run. */
function CellEditor(props: { initial: string; running: boolean; onRun(code: string): void; onCancel?(): void; autoFocus?: boolean }) {
  const [code, setCode] = useState(props.initial);
  useEffect(() => setCode(props.initial), [props.initial]);
  const run = () => {
    if (code.trim() && !props.running) props.onRun(code);
  };
  return (
    <div className="py-cell" onKeyDown={props.onCancel ? editorKeys(run, props.onCancel) : undefined}>
      <CodeEditor value={code} editable={!props.running} onChange={setCode} onRun={run} diagnostics={[]} autoFocus={props.autoFocus} />
      <div className="editor-bar">
        <span className="muted small">{T(props.onCancel ? "notebook.pythonHintCancel" : "notebook.pythonHint")}</span>
        <span className="spacer" />
        {props.onCancel && (
          <button type="button" className="ghost" onClick={props.onCancel} disabled={props.running}>
            {T("notebook.cancel")}
          </button>
        )}
        <button type="button" className="primary" onClick={run} disabled={!code.trim() || props.running}>
          {props.running ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />} {T("notebook.run")}
        </button>
      </div>
    </div>
  );
}

/** The body of a saved Python cell: its code, always editable, and the outputs of its last run. */
export function PythonCellBody({ block }: { block: Block }) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = block.data.run as PanelCellRun | undefined;
  const busy = running || block.data.running === true || run === undefined;
  return (
    <>
      <CellEditor
        initial={block.code ?? ""}
        running={busy}
        autoFocus={false}
        onRun={(code) => {
          setRunning(true);
          setError(null);
          api
            .rerunCell(block.id, code)
            .catch((e) => setError(e.message))
            .finally(() => setRunning(false));
        }}
      />
      {error && <div className="cell-note fail">{error}</div>}
      {run && !busy && <CellOutcome run={run} />}
    </>
  );
}

export type Draft = "note" | "python";

/** The notebook's toolbar under its title, as in Jupyter: add a Python cell or a note at the end, or ask the agent. */
export function NotebookRibbon(props: { draft: Draft | null; onAdd(kind: Draft): void; children?: ReactNode }) {
  return (
    <div className="nb-ribbon" role="toolbar" aria-label={T("notebook.notebook")}>
      <button className={`ghost ${props.draft === "python" ? "on" : ""}`} title={T("notebook.addPythonCell")} onClick={() => props.onAdd("python")}>
        <Plus size={15} />
        <Code2 size={15} /> {T("notebook.code")}
      </button>
      <button className={`ghost ${props.draft === "note" ? "on" : ""}`} title={T("notebook.addMarkdownNote")} onClick={() => props.onAdd("note")}>
        <Plus size={15} />
        <NotebookPen size={15} /> {T("notebook.note")}
      </button>
      {props.children && <span className="ribbon-sep" />}
      {props.children}
    </div>
  );
}

/** The cell being written at the end of the panel, opened from the ribbon. Not a block itself. */
export function DraftCell({ kind, onClose }: { kind: Draft; onClose(): void }) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const card = useRef<HTMLElement>(null);
  useEffect(() => {
    card.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }); // returns a Promise in recent Chromium: never hand it to React as a cleanup
  }, [kind]);

  const runCell = async (code: string) => {
    setRunning(true);
    setError(null);
    try {
      await api.addCell(code); // the cell itself arrives in the panel through its live stream
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  if (kind === "note")
    return (
      <article ref={card} className="block draft-cell">
        <header>
          <span className="block-kind">{T("notebook.note")}</span>
          <h3>{T("notebook.newNote")}</h3>
        </header>
        <div className="block-body">
          <NoteEditor onSave={(t, m) => api.addNote(t, m).then(onClose)} onCancel={onClose} />
        </div>
      </article>
    );

  return (
    <article ref={card} className="block draft-cell wide">
      <header>
        <span className="block-kind">Python</span>
        <h3>{T("notebook.newCell")}</h3>
      </header>
      <div className="block-body flush">
        <CellEditor initial="" running={running} onRun={runCell} onCancel={onClose} />
        {error && <div className="cell-note fail">{error}</div>}
      </div>
    </article>
  );
}
