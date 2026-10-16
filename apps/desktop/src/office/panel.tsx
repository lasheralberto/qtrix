// The office side bar, sliding in from the right: the team roster and, for the member picked (or the whiteboard),
// what there is to read or set. A member has two tabs: its work in the shown turn (the same components as the
// workspace panel) and its settings (the team configuration lives here). The whiteboard shows the report, or the
// lead's answer when reporting is off.
import { ChevronLeft, ChevronRight, ClipboardList, LoaderCircle, PanelRightClose, PanelRightOpen, Plus, X } from "lucide-react";
import { type ConversationReport, LEAD_ID, MAX_TEAM, type SkillInfo, type TeamAgent, type TeamConfig } from "../api";
import { Markdown } from "../blocks";
import type { AgentWork, Turn } from "../chat";
import { T } from "../i18n";
import { MemberConclusions, TurnWork } from "../workspace";
import type { AgentLive } from "./avatar";
import { MemberForm, newMember } from "./member";
import { wrapLines } from "./textures";

export type Panel = { kind: "desk"; id: string } | { kind: "board" };
export type Tab = "work" | "config";

/** The lines the 3D whiteboard shows: the report's title and abstract, else the lead's answer, else a note. */
export function boardLines(report: ConversationReport | null, leadText: string): string[] {
  if (report) {
    const lines = [report.title, ...wrapLines(report.abstract, 38, 6)].filter(Boolean);
    if (lines.length) return lines;
  }
  const lines = wrapLines(leadText, 38, 7);
  return lines.length ? lines : [T("office.nothingYet")];
}

export interface SideProps {
  open: boolean;
  onToggle(): void;
  /** The team being edited; it reaches the office (and the backend) on save. */
  draft: TeamConfig;
  onDraft(t: TeamConfig): void;
  dirty: boolean;
  saving: boolean;
  error: string | null;
  onSave(): void;
  onDiscard(): void;
  skills: SkillInfo[];
  profiles: string[];
  live: Record<string, AgentLive>;
  /** The agent selected in the room. */
  selected: string | null;
  panel: Panel | null;
  onPanel(p: Panel | null): void;
  tab: Tab;
  onTab(t: Tab): void;
  /** A member picked in the roster: it opens and is selected in the room as well. */
  onPick(id: string): void;
  turn: Turn | undefined;
  /** A member's work folded up to the shown turn (members other than the lead). */
  workOf(id: string): AgentWork | undefined;
  report: ConversationReport | null;
  at: number;
  total: number;
  onGo(i: number): void;
  onOpenObject(name: string, blockId?: string | null): void;
}

export function OfficeSide(p: SideProps) {
  const { draft } = p;
  const patch = (id: string, change: Partial<TeamAgent>) =>
    p.onDraft({ ...draft, agents: draft.agents.map((a) => (a.id === id ? { ...a, ...change } : a)) });
  const add = () => {
    if (draft.agents.length >= MAX_TEAM) return;
    const a = newMember(draft.agents.length);
    p.onDraft({ ...draft, agents: [...draft.agents, a] });
    p.onPanel({ kind: "desk", id: a.id });
    p.onTab("config");
  };
  const remove = (id: string) => {
    if (id === LEAD_ID) return;
    p.onDraft({ ...draft, agents: draft.agents.filter((a) => a.id !== id) });
    p.onPanel(null);
  };
  const deskId = p.panel?.kind === "desk" ? p.panel.id : null;
  const agent = draft.agents.find((a) => a.id === deskId);

  return (
    <aside className={`office-side ${p.open ? "" : "closed"}`} aria-label={T("team.title")}>
      <button className="office-side-handle" title={p.open ? T("office.hideSide") : T("office.showSide")} onClick={p.onToggle}>
        {p.open ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
      </button>
      <div className="office-side-inner" inert={!p.open}>
        <header className="office-side-head">
          <span className="ws-title">{T("team.title")}</span>
          <span className="spacer" />
          <label className="row team-switch">
            <input type="checkbox" checked={draft.enabled} onChange={(e) => p.onDraft({ ...draft, enabled: e.target.checked })} />
            {T("team.enabled")}
          </label>
        </header>
        <nav className="office-roster" aria-label={T("team.members")}>
          {draft.agents.map((a) => (
            <button
              key={a.id}
              className={`ghost ${p.panel?.kind === "desk" && p.panel.id === a.id ? "on" : ""} ${a.enabled ? "" : "off"}`}
              onClick={() => p.onPick(a.id)}
            >
              {p.live[a.id]?.running ? <LoaderCircle size={14} className="spin" /> : <span aria-hidden>{a.emoji || "🧑‍💼"}</span>}
              <span className="office-roster-name">{a.name || "—"}</span>
              <span className="office-roster-tail">
                {a.id === LEAD_ID && <span className="muted small">{T("team.defaultBadge")}</span>}
                {p.selected === a.id && <span className="office-here" title={T("office.inRoom")} />}
              </span>
            </button>
          ))}
          <div className="row office-roster-foot">
            <button className="ghost team-add" disabled={draft.agents.length >= MAX_TEAM} onClick={add}>
              <Plus size={14} /> {T("team.add")}
            </button>
            <span className="muted small">{T("team.count", { count: draft.agents.length, max: MAX_TEAM })}</span>
          </div>
          <button className={`ghost ${p.panel?.kind === "board" ? "on" : ""}`} onClick={() => p.onPanel({ kind: "board" })}>
            <ClipboardList size={14} /> <span className="office-roster-name">{T("office.whiteboard")}</span>
          </button>
          {!draft.enabled && <p className="muted small office-off">{T("team.offHint")}</p>}
        </nav>

        {p.panel?.kind === "board" || agent ? (
          <Detail {...p} agent={agent} onChange={(c) => agent && patch(agent.id, c)} onRemove={() => agent && remove(agent.id)} />
        ) : (
          <p className="muted small office-side-empty">{T("office.pickHint")}</p>
        )}

        {(p.dirty || p.error) && (
          <footer className="office-side-foot">
            {p.error && <span className="team-error small">{p.error}</span>}
            <span className="spacer" />
            <button className="ghost" disabled={p.saving || !p.dirty} onClick={p.onDiscard}>
              {T("office.discard")}
            </button>
            <button className="primary" disabled={p.saving || !p.dirty} onClick={p.onSave}>
              {T("team.save")}
            </button>
          </footer>
        )}
      </div>
    </aside>
  );
}

function Detail(p: SideProps & { agent: TeamAgent | undefined; onChange(c: Partial<TeamAgent>): void; onRemove(): void }) {
  const { panel, agent, turn } = p;
  const board = panel?.kind === "board";
  const lead = agent?.id === LEAD_ID;
  const member = agent && !lead ? p.workOf(agent.id) : undefined;
  const title = board ? T("office.results") : `${agent?.emoji ?? ""} ${agent?.name ?? ""}`.trim() || "—";
  const tab = board ? "work" : p.tab;
  return (
    <section className="office-detail" aria-label={title}>
      <header className="ws-head">
        <div className="ws-head-row">
          <span className="ws-title">{title}</span>
          <span className="spacer" />
          {tab === "work" && (
            <>
              <button className="ghost" title={T("office.prevTurn")} disabled={p.at <= 0} onClick={() => p.onGo(p.at - 1)}>
                <ChevronLeft size={15} />
              </button>
              <span className="ws-pos muted small">{p.total ? `${p.at + 1}/${p.total}` : "—"}</span>
              <button className="ghost" title={T("office.nextTurn")} disabled={p.at >= p.total - 1} onClick={() => p.onGo(p.at + 1)}>
                <ChevronRight size={15} />
              </button>
            </>
          )}
          <button className="ghost" title={T("office.closePanel")} onClick={() => p.onPanel(null)}>
            <X size={15} />
          </button>
        </div>
        {!board && (
          <div className="ws-tabs" role="tablist">
            {(["work", "config"] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => p.onTab(t)}>
                {T(t === "work" ? "office.tabWork" : "office.tabConfig")}
              </button>
            ))}
          </div>
        )}
        {tab === "work" && !board && turn && (lead || member) && (
          <p className="ws-question" title={member ? member.task : turn.question}>
            {member && <span className="ws-label">{T("team.task")}</span>}
            {member ? member.task : turn.question}
          </p>
        )}
      </header>
      <div className="ws-body">
        {board ? (
          <Board report={p.report} leadText={turn?.text ?? ""} />
        ) : tab === "config" && agent ? (
          <MemberForm key={agent.id} agent={agent} skills={p.skills} profiles={p.profiles} onChange={p.onChange} onRemove={p.onRemove} />
        ) : member ? (
          <>
            <MemberConclusions agent={member} />
            <TurnWork turn={member} onOpenObject={p.onOpenObject} />
          </>
        ) : lead && turn ? (
          <TurnWork turn={turn} onOpenObject={p.onOpenObject} />
        ) : (
          <p className="muted small">{T("office.noReport")}</p>
        )}
      </div>
    </section>
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
