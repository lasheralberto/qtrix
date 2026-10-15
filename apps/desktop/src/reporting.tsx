// Reporting mode: a switch in the panel header turns on a ReportingAgent that rewrites the conversation's report after
// every turn; the arrow next to the switch opens it as a paper-style document that embeds the objects it cites.
import { ArrowRight, FileDown, FileText, RefreshCw, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  api,
  rebuildReporting,
  threads as threadApi,
  type Block,
  type ChatEvent,
  type ConversationReport,
  type ReportObjectRef,
} from "./api";
import { BlockBody, Markdown } from "./blocks";
import { T } from "./i18n";

export type ReportingEvent = Extract<ChatEvent, { type: "reporting" }>;

export type Reporting = ReturnType<typeof useReporting>;

/** The reporting state of the open conversation, fed by the ``reporting`` events of the chat stream. */
export function useReporting(conversation: string | null, threadId: string | null, onChange: () => void) {
  const [enabled, setEnabled] = useState(false);
  const [report, setReport] = useState<ConversationReport | null>(null);
  const [version, setVersion] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const live = useRef(conversation);
  live.current = conversation;

  const load = useCallback(() => {
    const at = conversation;
    return api
      .reporting()
      .then((r) => {
        if (live.current !== at) return;
        setEnabled(r.enabled);
        setReport(r.report);
        setVersion(r.version);
      })
      .catch(() => {});
  }, [conversation]);

  useEffect(() => {
    setEnabled(false);
    setReport(null);
    setVersion(0);
    setRunning(false);
    setError(null);
    if (conversation) load();
  }, [conversation, load]);

  const handleEvent = useCallback(
    (ev: ReportingEvent) => {
      if (ev.status === "running") {
        setRunning(true);
        setError(null);
      } else {
        setRunning(false);
        if (ev.status === "error") setError(ev.message ?? T("reporting.updateFailed"));
        else load();
      }
    },
    [load],
  );

  const rebuild = useCallback(async () => {
    try {
      for await (const ev of rebuildReporting()) if (ev.type === "reporting") handleEvent(ev);
    } catch (e) {
      handleEvent({ type: "reporting", status: "error", message: String(e) });
    }
  }, [handleEvent]);

  const toggle = useCallback(async () => {
    if (!threadId) return;
    const next = !enabled;
    setEnabled(next);
    try {
      await threadApi.setReporting(threadId, next);
    } catch {
      setEnabled(!next);
      return;
    }
    onChange();
    if (next && !report) await rebuild(); // switched on mid-conversation: write it from every earlier turn
  }, [threadId, enabled, report, rebuild, onChange]);

  return { enabled, report, version, running, error, toggle, rebuild, handleEvent };
}

/** The switch, and, when it is on, the arrow that opens the document. */
export function ReportingToggle({
  reporting,
  disabled,
  onOpen,
}: {
  reporting: Reporting;
  disabled: boolean;
  onOpen: () => void;
}) {
  const { enabled, running } = reporting;
  return (
    <div className={`report-toggle ${enabled ? "on" : ""} ${running ? "live" : ""}`}>
      <button
        className="ghost report-switch"
        role="switch"
        aria-checked={enabled}
        disabled={disabled || running}
        title={enabled ? T("reporting.turnOff") : T("reporting.turnOn")}
        onClick={reporting.toggle}
      >
        <FileText size={15} />
        <span className="switch-track" aria-hidden>
          <span className="switch-thumb" />
        </span>
        {running && <span className="ws-live-dot" />}
      </button>
      {enabled && (
        <button
          className="ghost report-open"
          title={running ? T("reporting.updating") : T("reporting.open")}
          aria-label={T("reporting.open")}
          onClick={onOpen}
        >
          <ArrowRight size={15} />
        </button>
      )}
    </div>
  );
}

function ObjectFigure({ obj, label }: { obj: ReportObjectRef; label: string }) {
  const [block, setBlock] = useState<Block | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api
      .objectView(obj.name, obj.view)
      .then((v) => !cancelled && setBlock({ id: obj.name, kind: v.kind, title: v.title, data: v.data } as Block))
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [obj.name, obj.view]);
  return (
    <figure className="report-figure">
      {block ? (
        <BlockBody block={block} />
      ) : (
        <p className="muted small">{failed ? T("reporting.showFailed", { name: obj.name }) : T("reporting.loading")}</p>
      )}
      <figcaption>
        <b>{label}.</b> {obj.caption || block?.title || obj.name} <span className="muted">({obj.name})</span>
      </figcaption>
    </figure>
  );
}

/** The report as a document page. Figures and tables are numbered in reading order. */
export function ReportPage({ report, version, title }: { report: ConversationReport; version: number; title: string }) {
  let figures = 0;
  let tables = 0;
  const label = (name: string) => (/^gph/.test(name) ? T("reporting.figure", { n: ++figures }) : T("reporting.table", { n: ++tables }));
  return (
    <article className="report-page">
      <header>
        <p className="report-kicker">{T("reporting.kicker", { version })}</p>
        <h1>{report.title || title}</h1>
        <p className="muted small">
          {title} · {new Date().toLocaleDateString("en-US", { day: "numeric", month: "long", year: "numeric" })}
        </p>
      </header>
      {report.abstract && (
        <section className="report-abstract">
          <h2>{T("reporting.abstract")}</h2>
          <Markdown text={report.abstract} />
        </section>
      )}
      {report.sections.map((s, i) => (
        <section key={i}>
          <h2>
            {i + 1}. {s.heading}
          </h2>
          <Markdown text={s.body_md} />
          {s.objects.map((o) => (
            <ObjectFigure key={`${o.name}/${o.view ?? ""}`} obj={o} label={label(o.name)} />
          ))}
        </section>
      ))}
      {report.conclusions_md && (
        <section>
          <h2>{T("reporting.conclusions")}</h2>
          <Markdown text={report.conclusions_md} />
        </section>
      )}
      {report.limitations_md && (
        <section>
          <h2>{T("reporting.limitations")}</h2>
          <Markdown text={report.limitations_md} />
        </section>
      )}
      {report.next_steps.length > 0 && (
        <section>
          <h2>{T("reporting.nextSteps")}</h2>
          <ul>
            {report.next_steps.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}

/** The report in a PDF-like viewer: grey desk, white A4 sheet, export to PDF. */
export function ReportDocument({
  reporting,
  title,
  notebook,
  onClose,
}: {
  reporting: Reporting;
  title: string;
  notebook: string;
  onClose: () => void;
}) {
  const { report, version, running, error } = reporting;
  const [saved, setSaved] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const exportPdf = async () => {
    document.body.classList.add("report-printing");
    await new Promise((r) => setTimeout(r, 400)); // let the figures settle at the page width
    try {
      const bridge = window.qtrix;
      if (bridge?.exportPdf) {
        const out = await bridge.exportPdf(`${notebook}-${title}`);
        if (out.ok && out.path) setSaved(out.path);
      } else window.print();
    } finally {
      document.body.classList.remove("report-printing");
    }
  };

  return createPortal(
    <div className="report-overlay" role="dialog" aria-label={T("reporting.report")}>
      <div className="report-bar">
        <FileText size={16} />
        <b>{T("reporting.report")}</b>
        <span className="muted small">
          {report ? `v${version}` : T("reporting.noContent")}
          {running && T("reporting.updatingSuffix")}
        </span>
        <span className="spacer" />
        {saved && (
          <button className="ghost small" title={saved} onClick={() => window.qtrix?.openPath(saved)}>
            {T("reporting.openSavedPdf")}
          </button>
        )}
        <button
          className="ghost"
          disabled={running}
          title={T("reporting.rebuildHint")}
          onClick={reporting.rebuild}
        >
          <RefreshCw size={15} /> {T("reporting.rebuild")}
        </button>
        <button className="ghost" disabled={!report} onClick={exportPdf}>
          <FileDown size={15} /> {T("reporting.exportPdf")}
        </button>
        <button className="ghost" title={T("reporting.close")} aria-label={T("reporting.close")} onClick={onClose}>
          <X size={16} />
        </button>
      </div>
      <div className="report-desk">
        {error && <p className="report-error">{error}</p>}
        {report ? (
          <ReportPage report={report} version={version} title={title} />
        ) : (
          <article className="report-page empty">
            <p className="muted">
              {running
                ? T("reporting.writing")
                : T("reporting.placeholder")}
            </p>
          </article>
        )}
      </div>
    </div>,
    document.body,
  );
}
