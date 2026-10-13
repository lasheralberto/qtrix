// API keys, privacy mode, the qtrix folder and super-agent configuration. The active model is chosen
// only in the chat header (modelpicker.tsx).
import { Activity, FolderOpen, FolderSync, HardDrive, KeyRound, ShieldCheck, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, formatBytes, localai, type LlmUsage, type SystemInfo, type UserSettings } from "./api";
import { PROVIDERS } from "./modelpicker";

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
        <Activity size={15} /> Uso de modelos
      </h3>
      {rows.length === 0 ? (
        <p className="muted small">Aún no hay llamadas registradas.</p>
      ) : (
        <table className="folder-table">
          <tbody>
            {rows.map(([model, m]) => (
              <tr key={model}>
                <td>
                  <code>{model}</code>
                </td>
                <td className="num small">{m.calls} llamadas</td>
                <td className="num small">{(m.prompt_tokens + m.completion_tokens).toLocaleString()} tokens</td>
                <td className="num small" title={m.unpriced_calls ? `${m.unpriced_calls} llamadas sin precio conocido` : undefined}>
                  {m.cost_usd > 0 ? `$${m.cost_usd.toFixed(4)}` : "—"}
                  {m.unpriced_calls > 0 && m.cost_usd > 0 ? "+" : ""}
                </td>
                <td className="muted small">{m.fallbacks ? `${m.fallbacks} como respaldo` : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="muted small">Todas las llamadas pasan por el gateway LiteLLM del motor (también en /llm/v1, compatible con OpenAI).</p>
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
    if (!res.ok) setError(res.error ?? "No se pudo cambiar la carpeta");
    setTarget(null);
  }

  return (
    <section>
      <h3>
        <HardDrive size={15} /> Carpeta de qtrix
      </h3>
      <div className="row">
        <code className="home-path">{info?.home ?? "…"}</code>
        <span className="muted small">{formatBytes(total)}</span>
        <span className="spacer" />
        {bridge && info && (
          <button className="ghost" onClick={() => bridge.openPath(info.home)}>
            <FolderOpen size={14} /> Abrir
          </button>
        )}
        {bridge && (
          <button className="ghost" disabled={busy} onClick={async () => setTarget(await bridge.chooseFolder())}>
            <FolderSync size={14} /> Cambiar…
          </button>
        )}
      </div>
      {target && (
        <div className="move-box">
          <p className="small">
            Nueva carpeta: <code>{target}</code>
          </p>
          <div className="row">
            <button className="primary" disabled={busy} onClick={() => apply(true)}>
              Mover mis datos aquí
            </button>
            <button disabled={busy} onClick={() => apply(false)} title="Abre esa carpeta tal cual (nueva o de otra instalación de qtrix)">
              Usar sin mover
            </button>
            <button className="ghost" disabled={busy} onClick={() => setTarget(null)}>
              Cancelar
            </button>
          </div>
          <p className="muted small">qtrix se reiniciará. Mover incluye datos, memoria, modelos descargados y registros.</p>
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
      {!bridge && <p className="muted small">Cambiar la carpeta solo está disponible en la app de escritorio.</p>}
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
          <h2>Ajustes</h2>
          <button className="ghost" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="settings">
          <FolderSection />
          <section>
            <h3>
              <ShieldCheck size={15} /> Privacidad
            </h3>
            <label className="row">
              <input
                type="checkbox"
                checked={s.local_only}
                disabled={s.forced_local_only}
                onChange={(e) => update({ local_only: e.target.checked })}
              />
              Solo modelos locales: ningún prompt sale de esta máquina
            </label>
            <p className="muted small">
              Los datos nunca se envían: los modelos remotos solo reciben tu conversación y resúmenes de resultados.
            </p>
          </section>

          <section>
            <h3>
              <KeyRound size={15} /> API keys (se guardan en el llavero del sistema)
            </h3>
            <p className="muted small">Con una clave guardada, los modelos de ese proveedor aparecen en el selector de modelo del chat.</p>
            {cloudProviders.map((p) => (
              <div key={p.id} className="row">
                <span className="key-label">{p.label}</span>
                <input
                  type="password"
                  placeholder={s.api_keys?.[p.id] ? "•••••••• guardada" : "sin configurar"}
                  value={keys[p.id] ?? ""}
                  onChange={(e) => setKeys({ ...keys, [p.id]: e.target.value })}
                />
                <button
                  className="ghost"
                  disabled={!keys[p.id]}
                  onClick={() => api.setApiKey(p.id, keys[p.id]).then(() => api.settings().then((n) => (setS({ ...s, api_keys: n.api_keys }), setKeys({ ...keys, [p.id]: "" }))))}
                >
                  Guardar
                </button>
              </div>
            ))}
          </section>

          <UsageSection />

          <section>
            <h3>Super-agente</h3>
            <label className="col">
              Instrucciones adicionales (estilo, campo de investigación, convenciones)
              <textarea
                rows={3}
                value={s.agent.instructions_extra}
                onChange={(e) => update({ agent: { ...s.agent, instructions_extra: e.target.value } })}
                placeholder="Ej.: Soy economista laboral; usa siempre errores clusterizados por individuo."
              />
            </label>
            <label className="row">
              Máximo de llamadas a herramientas por turno
              <input
                type="number"
                min={1}
                max={200}
                value={s.agent.max_tool_calls}
                onChange={(e) => update({ agent: { ...s.agent, max_tool_calls: Number(e.target.value) } })}
              />
            </label>
            <label className="row">
              <input
                type="checkbox"
                checked={s.agent.planner ?? true}
                onChange={(e) => update({ agent: { ...s.agent, planner: e.target.checked } })}
              />
              Planificar antes de actuar (planificador + ejecutor; una llamada al modelo más por mensaje)
            </label>
          </section>
        </div>
        <footer>
          <button className="ghost" onClick={onClose}>
            Cancelar
          </button>
          <button className="primary" onClick={() => api.saveSettings(s).then((n) => (onSaved(n), onClose()))}>
            Guardar
          </button>
        </footer>
      </div>
    </div>
  );
}
