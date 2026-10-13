// Local models: download GGUF models from Hugging Face into the qtrix folder and run them with the
// embedded llama.cpp runtime. Choosing which model the chat uses happens only in the chat header.
import { Check, Cpu, Download, FolderOpen, KeyRound, LoaderCircle, Search, Square, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import {
  formatBytes,
  localai,
  type Job,
  type LocalAIStatus,
  type LocalModel,
  type Recommended,
  type RepoFiles,
} from "./api";

const ACTIVE: Job["status"][] = ["queued", "running", "verifying"];

function Progress({ job, onCancel }: { job: Job; onCancel(): void }) {
  const pct = job.progress == null ? null : Math.round(job.progress * 100);
  const running = ACTIVE.includes(job.status);
  return (
    <div className={`job ${job.status}`}>
      <div className="job-head">
        <span className="job-title" title={job.title}>
          {job.title.split("/").pop()}
        </span>
        <span className="muted small">
          {job.status === "error"
            ? "error"
            : job.status === "done"
              ? "completado"
              : job.status === "cancelled"
                ? "cancelado"
                : `${formatBytes(job.done)}${job.total ? ` / ${formatBytes(job.total)}` : ""}${pct != null ? ` · ${pct}%` : ""}`}
        </span>
        {running && (
          <button className="ghost" title="Cancelar" onClick={onCancel}>
            <X size={13} />
          </button>
        )}
      </div>
      {running && (
        <div className="bar">
          <div style={{ width: `${pct ?? 5}%` }} className={pct == null ? "indeterminate" : ""} />
        </div>
      )}
      {job.error ? <p className="error small">{job.error}</p> : running && <p className="muted small">{job.message}</p>}
    </div>
  );
}

function RuntimeCard({ status, onChange }: { status: LocalAIStatus; onChange(): void }) {
  const installed = status.runtime.installed;
  const mac = status.platform.startsWith("macos");
  const [variant, setVariant] = useState(status.variant);
  const job = status.jobs.find((j) => j.kind === "runtime" && ACTIVE.includes(j.status));
  return (
    <section className="runtime-card">
      <div className="runtime-head">
        <Cpu size={16} />
        <div>
          <strong>Motor llama.cpp</strong>
          <div className="muted small">
            {installed ? `Instalado ${installed.tag} · ${installed.variant}` : "No instalado (≈20–40 MB, se descarga una vez)"}
            {status.runtime.running && status.runtime.model && ` · sirviendo ${status.runtime.model.split(/[\\/]/).pop()}`}
          </div>
        </div>
        <span className="spacer" />
        {!mac && (
          <select value={variant} onChange={(e) => setVariant(e.target.value)} title="Aceleración">
            <option value="cpu">CPU</option>
            <option value="vulkan">GPU · Vulkan (AMD/Intel/NVIDIA)</option>
            <option value="cuda">GPU · CUDA (NVIDIA)</option>
          </select>
        )}
        <button disabled={!!job} onClick={() => localai.installRuntime(mac ? "cpu" : variant).then(onChange)}>
          <Download size={14} /> {installed ? "Actualizar" : "Instalar"}
        </button>
        {status.runtime.running && (
          <button className="ghost" title="Detener el modelo cargado y liberar memoria" onClick={() => localai.stopRuntime().then(onChange)}>
            <Square size={13} /> Detener
          </button>
        )}
      </div>
      {job && <Progress job={job} onCancel={() => localai.cancel(job.id).then(onChange)} />}
    </section>
  );
}

function HubBrowser({ onDownload }: { onDownload(repo: string, file: string): void }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<{ repo: string; downloads: number }[] | null>(null);
  const [open, setOpen] = useState<RepoFiles | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run(fn: () => Promise<void>) {
    setLoading(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="hub">
      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          run(async () => setResults(await localai.search(query)));
        }}
      >
        <input className="grow" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Buscar modelos GGUF: qwen, llama, mistral, gemma…" />
        <button type="submit" disabled={loading}>
          {loading ? <LoaderCircle size={14} className="spin" /> : <Search size={14} />} Buscar
        </button>
      </form>
      {error && <p className="error small">{error}</p>}
      {open ? (
        <div className="repo-files">
          <button className="ghost" onClick={() => setOpen(null)}>
            ← resultados
          </button>
          <h4>
            {open.repo} {open.gated && <span className="pill warn">requiere aceptar licencia + token</span>}
            {open.license && <span className="pill">{open.license}</span>}
          </h4>
          <table className="df-table">
            <thead>
              <tr>
                <th>Fichero</th>
                <th>Cuantización</th>
                <th>Tamaño</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {open.files
                .filter((f) => f.kind === "model")
                .map((f) => (
                  <tr key={f.name}>
                    <td>
                      {f.name}
                      {f.files.length > 1 && <span className="muted small"> ({f.files.length} partes)</span>}
                    </td>
                    <td>{f.quant ?? "—"}</td>
                    <td className="num">{formatBytes(f.size)}</td>
                    <td>
                      <button onClick={() => onDownload(open.repo, f.name)}>
                        <Download size={13} /> Descargar
                      </button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="muted small">Q4_K_M es el mejor punto de partida: buena calidad con un cuarto del tamaño original.</p>
        </div>
      ) : (
        results && (
          <ul className="repo-list">
            {results.length === 0 && <li className="muted">Sin resultados</li>}
            {results.map((r) => (
              <li key={r.repo}>
                <button className="repo" onClick={() => run(async () => setOpen(await localai.repo(r.repo)))}>
                  <span>{r.repo}</span>
                  <span className="muted small">{r.downloads?.toLocaleString("es-ES")} descargas</span>
                </button>
              </li>
            ))}
          </ul>
        )
      )}
    </div>
  );
}

export function ModelsModal({ onClose }: { onClose(): void }) {
  const [tab, setTab] = useState<"recommended" | "hub" | "local">("recommended");
  const [status, setStatus] = useState<LocalAIStatus | null>(null);
  const [recommended, setRecommended] = useState<Recommended[]>([]);
  const [models, setModels] = useState<LocalModel[]>([]);
  const [token, setToken] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [s, r, m] = await Promise.all([localai.status(), localai.recommended(), localai.models()]);
      setStatus(s);
      setRecommended(r);
      setModels(m);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const busy = status?.jobs.some((j) => ACTIVE.includes(j.status)) ?? false;
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, [busy, refresh]);

  const download = (repo: string, file: string) =>
    localai
      .download(repo, file)
      .then(refresh)
      .catch((e) => setError(e.message));

  const modelJobs = status?.jobs.filter((j) => j.kind === "model").slice(0, 6) ?? [];
  const installedNames = new Set(models.map((m) => m.id.split("/").pop()));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal wide models" onClick={(e) => e.stopPropagation()}>
        <header>
          <Cpu size={16} />
          <h2>Modelos locales</h2>
          <span className="muted small">Se guardan en {status?.models_dir ?? "…"} y se ejecutan en tu máquina · elígelos en el selector del chat</span>
          <button className="ghost" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="models-body">
          {error && (
            <p className="error small" onClick={() => setError("")}>
              {error}
            </p>
          )}
          {status && <RuntimeCard status={status} onChange={refresh} />}
          {modelJobs.length > 0 && (
            <section className="jobs">
              {modelJobs.map((j) => (
                <Progress key={j.id} job={j} onCancel={() => localai.cancel(j.id).then(refresh)} />
              ))}
            </section>
          )}
          <nav className="tabs">
            <button className={tab === "recommended" ? "on" : ""} onClick={() => setTab("recommended")}>
              Recomendados
            </button>
            <button className={tab === "hub" ? "on" : ""} onClick={() => setTab("hub")}>
              Buscar en Hugging Face
            </button>
            <button className={tab === "local" ? "on" : ""} onClick={() => setTab("local")}>
              Descargados ({models.length})
            </button>
          </nav>

          {tab === "recommended" && (
            <div className="rec-grid">
              {recommended.map((r) => {
                const local = models.find((m) => m.id.endsWith(r.file));
                const downloading = status?.jobs.some((j) => j.title === `${r.repo}/${r.file}` && ACTIVE.includes(j.status));
                return (
                  <article key={r.repo} className={`rec ${local?.active ? "active" : ""}`}>
                    <h4>{r.title}</h4>
                    <p className="muted small">{r.note}</p>
                    <p className="small">
                      {r.size_gb} GB · ≈{r.ram_gb} GB de RAM · <span className="muted">{r.repo}</span>
                    </p>
                    {local ? (
                      <span className={`pill ${local.active ? "ok" : ""}`}>
                        <Check size={12} /> {local.active ? "en uso" : "descargado"}
                      </span>
                    ) : (
                      <button disabled={downloading} onClick={() => download(r.repo, r.file)}>
                        {downloading ? <LoaderCircle size={13} className="spin" /> : <Download size={13} />} Descargar
                      </button>
                    )}
                  </article>
                );
              })}
            </div>
          )}

          {tab === "hub" && <HubBrowser onDownload={download} />}

          {tab === "local" && (
            <div className="table-scroll">
              {models.length === 0 ? (
                <p className="muted">Aún no hay modelos. Descarga uno de los recomendados o busca en Hugging Face.</p>
              ) : (
                <table className="df-table">
                  <thead>
                    <tr>
                      <th>Modelo</th>
                      <th>Cuant.</th>
                      <th>Tamaño</th>
                      <th>Origen</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m) => (
                      <tr key={m.id}>
                        <td>
                          {m.name} {m.active && <span className="pill ok">activo</span>} {m.serving && <span className="pill">cargado</span>}
                        </td>
                        <td>{m.quant ?? "—"}</td>
                        <td className="num">{formatBytes(m.size)}</td>
                        <td className="muted small">{m.repo ?? "copiado a mano"}</td>
                        <td className="row-actions">
                          <button
                            className="ghost danger"
                            title="Eliminar del disco"
                            onClick={() => {
                              if (confirm(`¿Eliminar ${m.name} (${formatBytes(m.size)}) del disco?`)) localai.remove(m.id).then(refresh);
                            }}
                          >
                            <Trash2 size={14} />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {installedNames.size > 0 && <p className="muted small">También puedes copiar ficheros .gguf a la carpeta de modelos: aparecerán aquí.</p>}
            </div>
          )}
        </div>
        <footer className="models-foot">
          <KeyRound size={14} />
          <span className="small">Token de Hugging Face (opcional, para modelos con licencia restringida)</span>
          <input
            type="password"
            value={token}
            placeholder={status?.hf_token ? "•••••••• guardado en el llavero" : "hf_…"}
            onChange={(e) => setToken(e.target.value)}
          />
          <button
            disabled={!token}
            onClick={() =>
              localai
                .setHfToken(token)
                .then(() => setToken(""))
                .then(refresh)
            }
          >
            Guardar
          </button>
          <span className="spacer" />
          {status && window.qtrix?.openPath && (
            <button className="ghost" onClick={() => window.qtrix?.openPath(status.models_dir)}>
              <FolderOpen size={14} /> Abrir carpeta
            </button>
          )}
        </footer>
      </div>
    </div>
  );
}
