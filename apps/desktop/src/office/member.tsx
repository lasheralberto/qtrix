// A team member's settings, edited in the office side bar: up to MAX_TEAM specialist econometricians the lead may
// hand parts of a request to. Each has its own behaviour, skills and model; the lead decides per request which of
// them (if any) to call on.
import { Trash2 } from "lucide-react";
import { LEAD_ID, type SkillInfo, type TeamAgent } from "../api";
import { T } from "../i18n";

const newId = () => `ag-${Math.random().toString(16).slice(2, 8)}`;

/** A fresh specialist, numbered after the members already there. */
export function newMember(count: number): TeamAgent {
  return { id: newId(), name: T("team.newMember", { n: count + 1 }), emoji: "", behavior: "", skills: [], profile: null, enabled: true };
}

export function MemberForm({
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
