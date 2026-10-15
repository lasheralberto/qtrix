// API keys, privacy mode, the qtrix folder and super-agent configuration. The active model is chosen
// only in the chat header (modelpicker.tsx).
import { Activity, Brain, Cloud, FolderOpen, FolderSync, HardDrive, KeyRound, ShieldCheck, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, formatBytes, localai, type LlmUsage, type SystemInfo, type UserSettings } from "./api";
import { PROVIDERS } from "./modelpicker";
import { T } from "./i18n";

/** Choices for the cap on the conversation's context (tokens). */
const CONTEXT_PRESETS = [16_000, 32_000, 64_000, 128_000, 200_000, 500_000, 1_000_000];

/** Tokens and cost per model, as recorded by the LiteLLM gateway. */
function UsageSection() {
  const [usage, setUsage] = useState<LlmUsage | null>(null);
  useEffect(() => {
    api.llmUsage().then(setUsage).catch(() => {});
  }, []);
  const rows = Object.entries(usage?.models ?? {});
  return (
    <section>
      <h3>
        <Activity size={15} /> {T("settings.usage")}
      </h3>
      {rows.length === 0 ? (
        <p className="muted small">{T("settings.noCalls")}</p>
      ) : (
        <table className="folder-table">
          <tbody>
            {rows.map(([model, m]) => (
              <tr key={model}>
                <td>
                  <code>{model}</code>
                </td>
                <td className="num small">{T("settings.calls", { count: m.calls })}</td>
                <td className="num small">{T("settings.tokens", { count: (m.prompt_tokens + m.completion_tokens).toLocaleString() })}</td>
                <td className="num small" title={m.unpriced_calls ? T("settings.unpriced", { count: m.unpriced_calls }) : undefined}>
                  {m.cost_usd > 0 ? `$${m.cost_usd.toFixed(4)}` : "—"}
                  {m.unpriced_calls > 0 && m.cost_usd > 0 ? "+" : ""}
                </td>
                <td className="muted small">{m.fallbacks ? T("settings.asFallback", { count: m.fallbacks }) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted small">{T("settings.gatewayNote")}</p>
    </section>
  );
}

/** The qtrix folder: where it is, what it holds, open it or move it. */
function FolderSection() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    localai.systemInfo().then(setInfo).catch(() => {});
  }, []);
  const bridge = window.qtrix;
  const total = info?.folders.reduce((acc, f) => acc + f.size, 0) ?? 0;

  async function apply(move: boolean) {
    if (!bridge || !target) return;
    setBusy(true);
    setError("");
    const res = await bridge.setHome(target, move); // the engine restarts and the window reloads
    setBusy(false);
    if (!res.ok) setError(res.error ?? T("settings.folderChangeFailed"));
    setTarget(null);
  }

  return (
    <section>
      <h3>
        <HardDrive size={15} /> {T("settings.folder")}
      </h3>
      <div className="row">
        <code className="home-path">{info?.home ?? "…"}</code>
        <span className="muted small">{formatBytes(total)}</span>
        <span className="spacer" />
        {bridge && info && (
          <button className="ghost" onClick={() => bridge.openPath(info.home)}>
            <FolderOpen size={14} /> {T("settings.open")}
          </button>
        )}
        {bridge && (
          <button className="ghost" disabled={busy} onClick={async () => setTarget(await bridge.chooseFolder())}>
            <FolderSync size={14} /> {T("settings.change")}
          </button>
        )}
      </div>
      {target && (
        <div className="move-box">
          <p className="small">
            {T("settings.newFolder")} <code>{target}</code>
          </p>
          <div className="row">
            <button className="primary" disabled={busy} onClick={() => apply(true)}>
              {T("settings.moveData")}
            </button>
            <button disabled={busy} onClick={() => apply(false)} title={T("settings.useAsIsTitle")}>
              {T("settings.useAsIs")}
            </button>
            <button className="ghost" disabled={busy} onClick={() => setTarget(null)}>
              {T("settings.cancel")}
            </button>
          </div>
          <p className="muted small">{T("settings.restartNote")}</p>
        </div>
      )}
      {error && <p className="error small">{error}</p>}
      {info && (
        <table className="folder-table">
          <tbody>
            {info.folders.map((f) => (
              <tr key={f.name}>
                <td>
                  <code>{f.name}/</code>
                </td>
                <td className="muted small">{f.description}</td>
                <td className="num small">{formatBytes(f.size)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {!bridge && <p className="muted small">{T("settings.desktopOnly")}</p>}
    </section>
  );
}

/** Where prompts go and whether the semantic memory is up: what the rail used to show as badges. */
function StatusSection({ s }: { s: UserSettings }) {
  const [memory, setMemory] = useState("…");
  useEffect(() => {
    const read = () => api.health().then((h) => setMemory(h.memory)).catch(() => setMemory("offline"));
    read();
    const poll = setInterval(read, 5000);
    return () => clearInterval(poll);
  }, []);
  const provider = (s.profiles.find((p) => p.name === s.active_profile) ?? s.profiles[0])?.provider;
  return (
    <section>
      <h3>
        <Activity size={15} /> {T("settings.status")}
      </h3>
      <div className="row">
        {s.active_is_local ? <HardDrive size={14} /> : <Cloud size={14} />}
        {s.active_is_local ? T("settings.promptsLocal") : T("settings.promptsRemote", { provider: provider ?? "" })}
      </div>
      <div className="row">
        <Brain size={14} /> {T("settings.memory", { status: memory })}
      </div>
    </section>
  );
}

export function SettingsModal({ onClose, onSaved }: { onClose(): void; onSaved(s: UserSettings): void }) {
  const [s, setS] = useState<UserSettings | null>(null);
  const [keys, setKeys] = useState<Record<string, string>>({});

  useEffect(() => {
    api.settings().then(setS);
  }, []);
  if (!s) return null;

  const update = (patch: Partial<UserSettings>) => setS({ ...s, ...patch });
  const cloudProviders = PROVIDERS.filter((p) => !p.local);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>{T("settings.title")}</h2>
          <button className="ghost" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="settings">
          <FolderSection />
          <section>
            <h3>
              <ShieldCheck size={15} /> {T("settings.privacy")}
            </h3>
            <label className="row">
              <input
                type="checkbox"
                checked={s.local_only}
                disabled={s.forced_local_only}
                onChange={(e) => update({ local_only: e.target.checked })}
              />
              {T("settings.localOnly")}
            </label>
            <p className="muted small">
              {T("settings.dataNote")}
            </p>
            <label className="row">
              <input
                type="checkbox"
                checked={s.agent.allow_code ?? false}
                onChange={(e) => update({ agent: { ...s.agent, allow_code: e.target.checked } })}
              />
              {T("settings.autoExec")}
            </label>
            <p className="muted small">
              {T("settings.autoExecNote")}
            </p>
          </section>

          <StatusSection s={s} />

          <section>
            <h3>
              <KeyRound size={15} /> {T("settings.apiKeys")}
            </h3>
            <p className="muted small">{T("settings.apiKeysNote")}</p>
            {cloudProviders.map((p) => (
              <div key={p.id} className="row">
                <span className="key-label">{p.label}</span>
                <input
                  type="password"
                  placeholder={s.api_keys?.[p.id] ? T("settings.keySaved") : T("settings.keyUnset")}
                  value={keys[p.id] ?? ""}
                  onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
                />
                <button
                  className="ghost"
                  disabled={!keys[p.id]}
                  onClick={() => api.setApiKey(p.id, keys[p.id]).then(() => api.settings().then((n) => (setS({ ...s, api_keys: n.api_keys }), setKeys({ ...keys, [p.id]: "" }))))}
                >
                  {T("settings.save")}
                </button>
              </div>
            ))}
          </section>

          <UsageSection />

          <section>
            <h3>{T("settings.contextTitle")}</h3>
            <p className="muted small">
              {T("settings.contextNote")}
            </p>
            <label className="row">
              {T("settings.maxContext")}
              <select
                value={s.agent.max_context ?? 1_000_000}
                onChange={(e) => update({ agent: { ...s.agent, max_context: Number(e.target.value) } })}
              >
                {[...new Set([...CONTEXT_PRESETS, s.agent.max_context ?? 1_000_000])]
                  .sort((a, b) => a - b)
                  .map((n) => (
                    <option key={n} value={n}>
                      {T("settings.tokensSize", { size: n >= 1_000_000 ? `${n / 1_000_000}M` : `${Math.round(n / 1000)}k` })}
                    </option>
                  ))}
              </select>
            </label>
            <label className="row">
              <input
                type="checkbox"
                checked={s.agent.auto_compact ?? true}
                onChange={(e) => update({ agent: { ...s.agent, auto_compact: e.target.checked } })}
              />
              {T("settings.compactAt")}
              <input
                type="number"
                min={10}
                max={100}
                step={5}
                value={Math.round((s.agent.compact_at ?? 0.8) * 100)}
                disabled={s.agent.auto_compact === false}
                onChange={(e) =>
                  update({ agent: { ...s.agent, compact_at: Math.min(Math.max(Number(e.target.value), 10), 100) / 100 } })
                }
                style={{ width: 64 }}
              />
              {T("settings.ofContext")}
            </label>
          </section>
        </div>
        <footer>
          <button className="ghost" onClick={onClose}>
            {T("settings.cancel")}
          </button>
          <button className="primary" onClick={() => api.saveSettings(s).then((n) => (onSaved(n), onClose()))}>
            {T("settings.save")}
          </button>
        </footer>
      </div>
    </div>
  );
}
