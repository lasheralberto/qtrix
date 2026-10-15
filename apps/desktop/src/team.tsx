// The agent team: up to MAX_TEAM specialist econometricians the lead may hand parts of a request to. Each has its
// own behaviour, skills and model; the lead decides per request which of them (if any) to call on.
import { Plus, Trash2, Users, X } from "lucide-react";
import { useEffect, useState } from "react";
import {
  api,
  LEAD_ID,
  MAX_TEAM,
  type SkillInfo,
  type TeamAgent,
  type TeamConfig,
  type UserSettings,
} from "./api";
import { T } from "./i18n";

const newId = () => `ag-${Math.random().toString(16).slice(2, 8)}`;

export function TeamModal({
  settings,
  onClose,
  onSaved,
}: {
  settings: UserSettings;
  onClose(): void;
  onSaved(s: UserSettings): void;
}) {
  const [team, setTeam] = useState<TeamConfig>(
    settings.team ?? { enabled: false, agents: [] },
  );
  const [selected, setSelected] = useState<string | null>(
    team.agents[0]?.id ?? null,
  );
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .skills()
      .then(setSkills)
      .catch(() => {});
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const agent = team.agents.find((a) => a.id === selected) ?? null;
  const patch = (id: string, p: Partial<TeamAgent>) =>
    setTeam((t) => ({
      ...t,
      agents: t.agents.map((a) => (a.id === id ? { ...a, ...p } : a)),
    }));
  const add = () => {
    if (team.agents.length >= MAX_TEAM) return;
    const a: TeamAgent = {
      id: newId(),
      name: T("team.newMember", { n: team.agents.length + 1 }),
      emoji: "",
      behavior: "",
      skills: [],
      profile: null,
      enabled: true,
    };
    setTeam((t) => ({ ...t, agents: [...t.agents, a] }));
    setSelected(a.id);
  };
  const remove = (id: string) => {
    if (id === LEAD_ID) return;
    setTeam((t) => ({ ...t, agents: t.agents.filter((a) => a.id !== id) }));
    setSelected(LEAD_ID);
  };

  const save = async () => {
    if (team.agents.some((a) => !a.name.trim()))
      return setError(T("team.nameRequired"));
    setSaving(true);
    setError(null);
    try {
      onSaved(await api.saveSettings({ ...settings, team }));
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal wide team-modal"
        role="dialog"
        aria-label={T("team.title")}
        onClick={(e) => e.stopPropagation()}
      >
        <header>
          <Users size={17} />
          <h2>{T("team.title")}</h2>
          <label className="row team-switch">
            <input
              type="checkbox"
              checked={team.enabled}
              onChange={(e) => setTeam({ ...team, enabled: e.target.checked })}
            />
            {T("team.enabled")}
          </label>
          <button className="ghost" title={T("team.close")} onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="methods-body">
          <nav className="methods-nav" aria-label={T("team.members")}>
            {team.agents.map((a) => (
              <button
                key={a.id}
                className={`ghost ${a.id === selected ? "on" : ""} ${a.enabled ? "" : "off"}`}
                onClick={() => setSelected(a.id)}
              >
                <span aria-hidden>{a.emoji || "🧑‍💼"}</span> {a.name || "—"}
                {a.id === LEAD_ID && (
                  <span className="muted small">{T("team.defaultBadge")}</span>
                )}
              </button>
            ))}
            <button
              className="ghost team-add"
              disabled={team.agents.length >= MAX_TEAM}
              onClick={add}
            >
              <Plus size={14} /> {T("team.add")}
            </button>
            <p className="muted small team-count">
              {T("team.count", { count: team.agents.length, max: MAX_TEAM })}
            </p>
          </nav>
          <div className="methods-list team-detail">
            {!team.enabled && (
              <p className="muted small">{T("team.offHint")}</p>
            )}
            {agent && (
              <MemberForm
                key={agent.id}
                agent={agent}
                skills={skills}
                profiles={settings.profiles.map((p) => p.name)}
                onChange={(p) => patch(agent.id, p)}
                onRemove={() => remove(agent.id)}
              />
            )}
          </div>
        </div>
        <footer>
          {error && <span className="team-error small">{error}</span>}
          <button className="ghost" onClick={onClose}>
            {T("team.cancel")}
          </button>
          <button className="primary" disabled={saving} onClick={save}>
            {T("team.save")}
          </button>
        </footer>
      </div>
    </div>
  );
}

function MemberForm({
  agent,
  skills,
  profiles,
  onChange,
  onRemove,
}: {
  agent: TeamAgent;
  skills: SkillInfo[];
  profiles: string[];
  onChange(p: Partial<TeamAgent>): void;
  onRemove(): void;
}) {
  const toggleSkill = (name: string, on: boolean) =>
    onChange({
      skills: on
        ? [...agent.skills, name]
        : agent.skills.filter((s) => s !== name),
    });
  // The default member is the lead: always there, always working, on the model chosen in the chat.
  const isLead = agent.id === LEAD_ID;
  return (
    <div className="team-form">
      {isLead && <p className="muted small">{T("team.leadHint")}</p>}
      <div className="row">
        <label className="col team-emoji">
          {T("team.emoji")}
          <input
            value={agent.emoji}
            maxLength={8}
            onChange={(e) => onChange({ emoji: e.target.value })}
            placeholder="📊"
          />
        </label>
        <label className="col team-name">
          {T("team.name")}
          <input
            value={agent.name}
            maxLength={60}
            onChange={(e) => onChange({ name: e.target.value })}
          />
        </label>
        {!isLead && (
          <label className="row team-enabled">
            <input
              type="checkbox"
              checked={agent.enabled}
              onChange={(e) => onChange({ enabled: e.target.checked })}
            />
            {T("team.memberEnabled")}
          </label>
        )}
      </div>
      <label className="col">
        {T("team.behavior")}
        <textarea
          rows={6}
          value={agent.behavior}
          onChange={(e) => onChange({ behavior: e.target.value })}
          placeholder={T("team.behaviorPlaceholder")}
        />
        <span className="muted small">{T("team.behaviorHint")}</span>
      </label>
      {isLead ? (
        <p className="muted small">
          {T("team.model")}: {T("team.leadModel")}
        </p>
      ) : (
        <label className="col">
          {T("team.model")}
          <select
            value={agent.profile ?? ""}
            onChange={(e) => onChange({ profile: e.target.value || null })}
          >
            <option value="">{T("team.inheritModel")}</option>
            {profiles.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
      )}
      <fieldset className="team-skills">
        <legend>{T("team.skills")}</legend>
        {skills.length === 0 && (
          <p className="muted small">{T("team.noSkills")}</p>
        )}
        {skills.map((s) => (
          <label key={s.name} className="row team-skill" title={s.description}>
            <input
              type="checkbox"
              checked={agent.skills.includes(s.name)}
              onChange={(e) => toggleSkill(s.name, e.target.checked)}
            />
            <span>
              <code>{s.name}</code>
              <span className="muted small"> {s.description}</span>
            </span>
          </label>
        ))}
      </fieldset>
      {!isLead && (
        <button className="ghost danger team-remove" onClick={onRemove}>
          <Trash2 size={14} /> {T("team.remove")}
        </button>
      )}
    </div>
  );
}
