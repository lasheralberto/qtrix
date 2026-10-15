// The readable side of the office: what the controlled agent is looking at. At a desk, that agent's workspace (the
// same components as the workspace panel); at the whiteboard, the report, or the lead's answer when reporting is off.
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { ConversationReport } from "../api";
import { Markdown } from "../blocks";
import type { AgentWork, Turn } from "../chat";
import { T } from "../i18n";
import { MemberConclusions, TurnWork } from "../workspace";
import { wrapLines } from "./textures";

export type Panel = { kind: "desk"; id: string } | { kind: "board" };

/** The lines the 3D whiteboard shows: the report's title and abstract, else the lead's answer, else a note. */
export function boardLines(report: ConversationReport | null, leadText: string): string[] {
  if (report) {
    const lines = [report.title, ...wrapLines(report.abstract, 38, 6)].filter(Boolean);
    if (lines.length) return lines;
  }
  const lines = wrapLines(leadText, 38, 7);
  return lines.length ? lines : [T("office.nothingYet")];
}

export function OfficePanel({
  panel,
  turn,
  member,
  lead,
  memberName,
  report,
  at,
  total,
  onGo,
  onClose,
  onOpenObject,
}: {
  panel: Panel;
  turn: Turn | undefined;
  member: AgentWork | undefined;
  /** The desk is the lead's: it shows the turn's own work. */
  lead: boolean;
  memberName: string;
  /** The report to show on the board, already null when reporting is off. */
  report: ConversationReport | null;
  at: number;
  total: number;
  onGo(i: number): void;
  onClose(): void;
  onOpenObject(name: string, blockId?: string | null): void;
}) {
  const title = panel.kind === "board" ? T("office.results") : memberName;
  return (
    <aside className="office-panel" aria-label={title}>
      <header className="ws-head">
        <div className="ws-head-row">
          <span className="ws-title">{title}</span>
          <span className="spacer" />
          <button className="ghost" title={T("office.prevTurn")} disabled={at <= 0} onClick={() => onGo(at - 1)}>
            <ChevronLeft size={15} />
          </button>
          <span className="ws-pos muted small">{total ? `${at + 1}/${total}` : "—"}</span>
          <button className="ghost" title={T("office.nextTurn")} disabled={at >= total - 1} onClick={() => onGo(at + 1)}>
            <ChevronRight size={15} />
          </button>
          <button className="ghost" title={T("office.closePanel")} onClick={onClose}>
            <X size={15} />
          </button>
        </div>
        {panel.kind === "desk" && turn && (lead || member) && (
          <p className="ws-question" title={member ? member.task : turn.question}>
            {member && <span className="ws-label">{T("team.task")}</span>}
            {member ? member.task : turn.question}
          </p>
        )}
      </header>
      <div className="ws-body">
        {panel.kind === "board" ? (
          <Board report={report} leadText={turn?.text ?? ""} />
        ) : member ? (
          <>
            <MemberConclusions agent={member} />
            <TurnWork turn={member} onOpenObject={onOpenObject} />
          </>
        ) : lead && turn ? (
          <TurnWork turn={turn} onOpenObject={onOpenObject} />
        ) : (
          <p className="muted small">{T("office.noReport")}</p>
        )}
      </div>
    </aside>
  );
}

function Board({ report, leadText }: { report: ConversationReport | null; leadText: string }) {
  if (report)
    return (
      <div className="ws-card conclusion office-report">
        <h3>{report.title}</h3>
        <Markdown text={report.abstract} />
        {report.sections.map((s) => (
          <section key={s.heading}>
            <h4>{s.heading}</h4>
            <Markdown text={s.body_md} />
          </section>
        ))}
        <h4>{T("workspace.provisionalConclusion")}</h4>
        <Markdown text={report.conclusions_md} />
      </div>
    );
  if (!leadText) return <p className="muted small">{T("office.noReport")}</p>;
  return (
    <div className="ws-card conclusion">
      <span className="ws-label">{T("office.leadAnswer")}</span>
      <Markdown text={leadText} />
    </div>
  );
}
