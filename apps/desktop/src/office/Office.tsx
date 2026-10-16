// The office: a full-screen 3D view of the team at its desks, and where the team is set up. The room and the avatars
// live in the canvas; the side bar on the right (roster, each member's work and settings, the whiteboard) is plain
// HTML next to it. Movement rules are in state.ts.
import { OrthographicCamera } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, LEAD_ID, type SkillInfo, type TeamAgent, type TeamConfig, type UserSettings } from "../api";
import type { Turn } from "../chat";
import { T } from "../i18n";
import type { Reporting } from "../reporting";
import { mergeAgents } from "../workspace";
import { Avatar, type AgentLive } from "./avatar";
import { BOARD_SPOT, COFFEE_SPOT, layoutDesks, READING_ZONE, same, type Dir } from "./grid";
import { boardLines, OfficeSide, type Panel, type Tab } from "./panel";
import { Cat, CoffeeMachine, DeskMesh, Floor, Poster, Walls, Whiteboard } from "./room";
import { office, useOffice } from "./state";

export interface OfficeProps {
  settings: UserSettings;
  onSaved(s: UserSettings): void;
  turns: Turn[];
  /** The turn shown in the panels (null = the latest), shared with the workspace. */
  turnKey: string | null;
  onSelectTurn(key: string | null): void;
  reporting: Reporting;
  onOpenObject(name: string, blockId?: string | null): void;
  onClose(): void;
}

type Member = Pick<TeamAgent, "id" | "name" | "emoji" | "enabled">;
/** The lead when settings have no team yet. */
const LEAD: Member = { id: LEAD_ID, name: "", emoji: "", enabled: true };
const NO_TEAM: TeamConfig = { enabled: false, agents: [] };
const ARROWS: Record<string, Dir> = { ArrowUp: "n", ArrowDown: "s", ArrowLeft: "w", ArrowRight: "e" };

function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Fits the room to the viewport: an isometric orthographic camera whose zoom follows the window and the wheel. */
function CameraRig({ zoom }: { zoom: number }) {
  const size = useThree((s) => s.size);
  const base = Math.min(size.width / 17, size.height / 12);
  return (
    <OrthographicCamera
      makeDefault
      position={[12, 12, 12]}
      zoom={base * zoom}
      near={0.1}
      far={100}
      onUpdate={(c) => {
        c.lookAt(0, 0.4, 0);
        c.updateProjectionMatrix();
      }}
    />
  );
}

export function Office(props: OfficeProps) {
  const { settings, onSaved, turns } = props;
  const webgl = useMemo(hasWebGL, []);
  const [zoom, setZoom] = useState(1);
  const team = settings.team ?? NO_TEAM;
  // The lead always; the other members only with the team on and the member available. The room shows the saved
  // team: edits in the side bar take a seat once saved.
  const members = useMemo<Member[]>(() => {
    const agents: Member[] = team.agents.length ? team.agents : [LEAD];
    return team.enabled ? agents.filter((a) => a.enabled || a.id === LEAD_ID) : agents.slice(0, 1);
  }, [team]);

  // The team being edited in the side bar; it starts over from the saved one whenever that changes.
  const [draft, setDraft] = useState<TeamConfig>(team);
  useEffect(() => setDraft(team), [team]);
  const dirty = JSON.stringify(draft) !== JSON.stringify(team);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  useEffect(() => {
    api.skills().then(setSkills).catch(() => {});
  }, []);
  const save = async () => {
    if (draft.agents.some((a) => !a.name.trim())) return setError(T("team.nameRequired"));
    setSaving(true);
    setError(null);
    try {
      onSaved(await api.saveSettings({ ...settings, team: draft }));
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };
  const discard = () => {
    setDraft(team);
    setError(null);
  };
  // Leaving with unsaved changes to the team asks first.
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;
  const onClose = useCallback(() => {
    if (!dirtyRef.current || window.confirm(T("office.unsaved"))) props.onClose();
  }, [props.onClose]); // eslint-disable-line react-hooks/exhaustive-deps
  const desks = useMemo(() => layoutDesks(members.map((m) => m.id)), [members]);
  useEffect(() => office.setDesks(desks), [desks]);
  const state = useOffice();

  // Live state of each agent, from the latest turn: who is working, on what, who failed.
  const last = turns[turns.length - 1];
  const live = useMemo(() => {
    const out: Record<string, AgentLive> = {};
    for (const m of members) out[m.id] = { running: false, failed: false };
    if (last && out[members[0]?.id]) {
      out[members[0].id] = { running: last.running, status: last.status, failed: !!last.error };
      for (const a of last.agents ?? [])
        if (out[a.id]) out[a.id] = { running: a.running, status: a.status, failed: a.ok === false || !!a.error };
    }
    return out;
  }, [members, last]);
  const liveRef = useRef(live);
  liveRef.current = live;

  // Monitors light up once their agent has worked in any turn.
  const worked = useMemo(() => {
    const s = new Set<string>();
    if (turns.length && members[0]) s.add(members[0].id);
    for (const t of turns) for (const a of t.agents ?? []) s.add(a.id);
    return s;
  }, [turns, members]);

  // A new turn sends the whole team back to their desks, and a member starting on its task goes to its own; the
  // agent the user steers is left alone (it can still be walked home by hand).
  const runningIds = Object.entries(live)
    .filter(([, l]) => l.running)
    .map(([id]) => id)
    .join(",");
  const newTurn = last?.running ? last.key : null;
  useEffect(() => {
    const s = office.get();
    const ids = newTurn ? members.map((m) => m.id) : runningIds.split(",").filter(Boolean);
    for (const id of ids) {
      const p = s.positions[id];
      if (p && !p.seated && !p.home && s.steered !== id) office.walkHome(id);
    }
    // members changes only with the settings; the turn and who is running are the triggers.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runningIds, newTurn]);

  // Every so often an idle agent nobody is steering goes for a coffee and comes back; the cat strolls.
  useEffect(() => {
    const trip = () => {
      const s = office.get();
      const idle = members.filter((m) => s.positions[m.id]?.seated && s.steered !== m.id && !liveRef.current[m.id]?.running);
      if (!idle.length) return;
      const id = idle[Math.floor(Math.random() * idle.length)].id;
      if (office.walkTo(id, COFFEE_SPOT)) setTimeout(() => office.walkHome(id), 6000);
    };
    const coffee = setInterval(trip, 20000);
    const cat = setInterval(office.catStep, 2500);
    return () => {
      clearInterval(coffee);
      clearInterval(cat);
    };
  }, [members]);

  // The turn the panels show (the latest unless one was picked), with the members' work folded up to it.
  const index = props.turnKey ? turns.findIndex((t) => t.key === props.turnKey) : turns.length - 1;
  const at = index < 0 ? turns.length - 1 : index;
  const turn = turns[at];
  const merged = useMemo(() => mergeAgents(turns.slice(0, at + 1)), [turns, at]);
  const go = (i: number) => props.onSelectTurn(i >= turns.length - 1 ? null : turns[i].key);

  // The side bar and what it shows: a member (its work or its settings) or the whiteboard. Walking the controlled
  // agent to a chair or to the whiteboard opens it there; walking away (or Escape) closes it.
  const [sideOpen, setSideOpen] = useState(true);
  const [panel, setPanel] = useState<Panel | null>(null);
  const [tab, setTab] = useState<Tab>("work");
  const show = (p: Panel | null) => {
    setPanel(p);
    if (p) setSideOpen(true);
  };
  const pick = (id: string) => {
    show({ kind: "desk", id });
    if (state.positions[id]) office.select(id);
    // A member not saved yet has no work to show.
    if (!members.some((m) => m.id === id)) setTab("config");
  };
  const sel = state.selected ? state.positions[state.selected] : undefined;
  const selTile = sel ? `${state.selected}@${sel.tile.x},${sel.tile.z}` : "";
  const prevSel = useRef(selTile);
  useEffect(() => {
    // Only a step of the agent counts: opening the office or picking another agent leaves the panel as it is.
    const stepped = prevSel.current.split("@")[0] === state.selected && prevSel.current !== selTile && state.steered === state.selected;
    prevSel.current = selTile;
    if (!sel || !stepped) return;
    const desk = desks.find((d) => same(d.chair, sel.tile));
    if (desk) {
      show({ kind: "desk", id: desk.id });
      setTab("work");
    } else if (READING_ZONE.some((t) => same(t, sel.tile))) show({ kind: "board" });
    else setPanel(null);
    // The tile is the trigger; `sel` itself is a new object on every store update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selTile, desks]);

  // Arrow keys walk the selected agent, Enter sits it down; Escape closes the panel, then the side bar, then the office. The office
  // covers the app, so it takes these keys first (capture) and keeps them from the listeners underneath — the
  // workspace would otherwise close on the same Escape.
  useEffect(() => {
    const take = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopImmediatePropagation();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        take(e);
        if (panel) setPanel(null);
        else if (sideOpen) setSideOpen(false);
        else onClose();
        return;
      }
      if ((e.target as HTMLElement | null)?.closest("input, textarea, select, [contenteditable]")) return;
      const id = office.get().selected;
      if (!id) return;
      if (ARROWS[e.key]) {
        take(e);
        office.move(id, ARROWS[e.key]);
      } else if (e.key === "Enter") {
        take(e);
        office.sit(id);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose, panel, sideOpen]);

  // The whiteboard: the report (reporting on) or the lead's answer of the shown turn, once it is settled — a turn
  // still streaming keeps the board as it was. When it changes while the office follows the latest turn, the lead
  // walks over and writes it; opening the office or browsing old turns shows it at once.
  const report = props.reporting.enabled ? props.reporting.report : null;
  const settledText = useRef("");
  if (turn && !turn.running) settledText.current = turn.text;
  if (!turn) settledText.current = "";
  const lines = useMemo(() => boardLines(report, settledText.current), [report, settledText.current]); // eslint-disable-line react-hooks/exhaustive-deps
  const text = lines.join("\n");
  const [chars, setChars] = useState(Infinity);
  const firstBoard = useRef(true);
  useEffect(() => {
    const nothing = lines.length === 1 && lines[0] === T("office.nothingYet");
    if (firstBoard.current || props.turnKey !== null || nothing) {
      firstBoard.current = false;
      setChars(Infinity);
      return;
    }
    // The lead goes to the board unless the user is steering it.
    const lead = members[0]?.id;
    const walks = !!lead && office.get().steered !== lead && office.walkTo(lead, BOARD_SPOT);
    setChars(0);
    const timer = setInterval(() => setChars((c) => (c + 1 >= text.length ? Infinity : c + 1)), 40);
    const back = setTimeout(() => walks && office.walkHome(lead), text.length * 40 + 1500);
    return () => {
      clearInterval(timer);
      clearTimeout(back);
    };
    // Only a change of the text itself starts a writing session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return (
    <div className="office" role="dialog" aria-label={T("office.title")}>
      <header className="office-top">
        <span className="office-title">{T("office.title")}</span>
        <span className="muted small office-hint">{T("office.hint")}</span>
        <span className="spacer" />
        <button className="ghost" title={T("office.close")} onClick={onClose}>
          <X size={16} />
        </button>
      </header>
      <div className="office-body">
        <div className="office-stage" onWheel={(e) => setZoom((z) => Math.min(1.6, Math.max(0.7, z - e.deltaY * 0.001)))}>
          {webgl ? (
            <Canvas shadows="percentage">
              <color attach="background" args={["#dbeafe"]} />
              <CameraRig zoom={zoom} />
              <hemisphereLight args={["#ffffff", "#c9b8a8", 0.9]} />
              <directionalLight
                position={[6, 12, 4]}
                intensity={1.6}
                castShadow
                shadow-mapSize={[2048, 2048]}
                shadow-camera-left={-10}
                shadow-camera-right={10}
                shadow-camera-top={10}
                shadow-camera-bottom={-10}
              />
              <Floor />
              <Walls />
              <Whiteboard lines={lines} chars={chars} />
              <CoffeeMachine />
              <Poster />
              {desks.map((d, i) => (
                <DeskMesh key={d.id} desk={d} lead={i === 0} running={live[d.id]?.running ?? false} lit={worked.has(d.id)} failed={live[d.id]?.failed ?? false} />
              ))}
              {members.map((m) =>
                state.positions[m.id] ? (
                  <Avatar
                    key={m.id}
                    id={m.id}
                    name={m.name}
                    emoji={m.emoji}
                    pos={state.positions[m.id]}
                    selected={state.selected === m.id}
                    live={live[m.id]}
                  />
                ) : null,
              )}
              <Cat tile={state.cat} />
            </Canvas>
          ) : (
            <div className="office-nowebgl">
              <p>{T("office.noWebgl")}</p>
              <button className="primary" onClick={onClose}>
                {T("office.close")}
              </button>
            </div>
          )}
        </div>
        <OfficeSide
          open={sideOpen}
          onToggle={() => setSideOpen((o) => !o)}
          draft={draft}
          onDraft={setDraft}
          dirty={dirty}
          saving={saving}
          error={error}
          onSave={save}
          onDiscard={discard}
          skills={skills}
          profiles={settings.profiles.map((p) => p.name)}
          live={live}
          selected={state.selected}
          panel={panel}
          onPanel={show}
          tab={tab}
          onTab={setTab}
          onPick={pick}
          turn={turn}
          workOf={(id) => merged.find((a) => a.id === id)}
          report={report}
          at={at}
          total={turns.length}
          onGo={go}
          onOpenObject={props.onOpenObject}
        />
      </div>
    </div>
  );
}
