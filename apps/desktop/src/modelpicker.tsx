// The single place where the active model is chosen: installed local models, cloud models with a stored
// key, saved custom endpoints, and the fallback chain. Settings and the local-models manager only
// configure (keys, downloads); they never pick.
import { Boxes, Check, ChevronDown, Cloud, HardDrive, KeyRound, LifeBuoy, LoaderCircle, Plus, Search, Settings, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type ModelProfile, type UserSettings } from "./api";
import { T } from "./i18n";

type Provider = ModelProfile["provider"];

export const PROVIDERS: { id: Provider; label: string; local: boolean; example: string }[] = [
  { id: "llamacpp", label: "qtrix · llama.cpp", local: true, example: T("modelpicker.exampleLlamacpp") },
  { id: "ollama", label: "Ollama", local: true, example: "qwen3:8b" },
  { id: "lmstudio", label: "LM Studio", local: true, example: "qwen2.5-7b-instruct" },
  { id: "openai_compatible", label: "OpenAI-compatible (vLLM, llama.cpp…)", local: true, example: "model-name" },
  { id: "google", label: "Google Gemini", local: false, example: "gemini-3.8-flash" },
  { id: "anthropic", label: "Anthropic", local: false, example: "claude-sonnet-5-5" },
  { id: "openai", label: "OpenAI", local: false, example: "gpt-5" },
  { id: "mistral", label: "Mistral", local: false, example: "mistral-large-latest" },
];

const LOCAL_RUNTIMES: Provider[] = ["llamacpp", "ollama", "lmstudio"];
const CLOUD_PROVIDERS: Provider[] = ["google", "anthropic", "openai", "mistral"];
const SHORT: Record<Provider, string> = {
  llamacpp: "qtrix",
  ollama: "Ollama",
  lmstudio: "LM Studio",
  openai_compatible: "Endpoint",
  google: "Gemini",
  anthropic: "Anthropic",
  openai: "OpenAI",
  mistral: "Mistral",
};

/** Downloaded GGUFs are identified by their path inside models/; show just the file name. */
const shortModel = (provider: string, model: string) =>
  provider === "llamacpp" ? (model.split("/").pop() ?? model).replace(/\.gguf$/i, "") : model;

interface Option {
  provider: Provider;
  model: string;
  base_url?: string | null;
  /** Saved profile backing this option, if any. */
  profile?: string;
  /** Still reported by the runtime / provider (false = stale saved profile). */
  detected: boolean;
}

const same = (a: { provider: string; model: string; base_url?: string | null }, b: typeof a) =>
  a.provider === b.provider && a.model === b.model && (a.base_url ?? null) === (b.base_url ?? null);

/** Return settings that contain a profile for `o`, plus that profile's name. */
function ensureProfile(s: UserSettings, o: Omit<Option, "detected" | "profile">): [UserSettings, string] {
  const existing = s.profiles.find((p) => same(p, o));
  if (existing) return [s, existing.name];
  let name = `${SHORT[o.provider]} · ${shortModel(o.provider, o.model)}`;
  for (let i = 2; s.profiles.some((p) => p.name === name); i++) name = `${SHORT[o.provider]} · ${shortModel(o.provider, o.model)} (${i})`;
  const profile: ModelProfile = { name, provider: o.provider, model: o.model, ...(o.base_url ? { base_url: o.base_url } : {}) };
  return [{ ...s, profiles: [...s.profiles, profile] }, name];
}

export function ModelPicker({
  settings,
  onChange,
  onOpenModels,
  onOpenSettings,
}: {
  settings: UserSettings | null;
  onChange(s: UserSettings): void;
  onOpenModels(): void;
  onOpenSettings(): void;
}) {
  const [open, setOpen] = useState(false);
  const [detected, setDetected] = useState<Record<string, string[]>>({});
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState("");
  const [custom, setCustom] = useState<{ provider: Provider; model: string; base_url: string } | null>(null);
  const [test, setTest] = useState("");
  const root = useRef<HTMLDivElement>(null);

  const refresh = () => {
    setLoading(true);
    api
      .localModels()
      .then(setDetected)
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (!open) return;
    refresh();
    setTest("");
    const onDown = (e: MouseEvent) => root.current && !root.current.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!settings) return null;
  const s = settings;
  const active = s.profiles.find((p) => p.name === s.active_profile) ?? s.profiles[0];
  const fallbacks = s.fallbacks ?? [];
  const isLocal = s.active_is_local ?? true;

  // Everything the user can pick: detected models first, then saved profiles the runtime no longer reports.
  const options: Option[] = [];
  for (const provider of [...LOCAL_RUNTIMES, ...CLOUD_PROVIDERS])
    for (const model of detected[provider] ?? []) {
      const p = s.profiles.find((x) => x.provider === provider && x.model === model);
      options.push({ provider, model, profile: p?.name, detected: true });
    }
  for (const p of s.profiles)
    if (!options.some((o) => same(o, p))) options.push({ provider: p.provider, model: p.model, base_url: p.base_url, profile: p.name, detected: false });

  const q = query.trim().toLowerCase();
  const visible = (o: Option) => !q || `${SHORT[o.provider]} ${o.model} ${o.profile ?? ""}`.toLowerCase().includes(q);
  const isLocalProvider = (p: Provider) => PROVIDERS.find((x) => x.id === p)?.local ?? false;
  const localOptions = options.filter((o) => isLocalProvider(o.provider) && visible(o));
  const cloudOptions = options.filter((o) => !isLocalProvider(o.provider) && visible(o));
  const missingKeys = CLOUD_PROVIDERS.filter((p) => !s.api_keys?.[p]);

  const save = (next: UserSettings) => api.saveSettings(next).then(onChange);

  function choose(o: Omit<Option, "detected" | "profile">) {
    const [next, name] = ensureProfile(s, o);
    save({ ...next, active_profile: name, fallbacks: (next.fallbacks ?? []).filter((n) => n !== name) });
    setOpen(false);
  }

  function toggleFallback(o: Option) {
    const [next, name] = ensureProfile(s, o);
    const on = fallbacks.includes(name);
    save({ ...next, fallbacks: on ? fallbacks.filter((n) => n !== name) : [...fallbacks, name] });
  }

  function forget(o: Option) {
    save({ ...s, profiles: s.profiles.filter((p) => p.name !== o.profile), fallbacks: fallbacks.filter((n) => n !== o.profile) });
  }

  const row = (o: Option) => {
    const isActive = !!active && same(o, active);
    const fb = o.profile ? fallbacks.indexOf(o.profile) : -1;
    return (
      <li key={`${o.provider}::${o.model}::${o.base_url ?? ""}`} className={`mp-row ${isActive ? "on" : ""}`}>
        <button className="mp-choose" onClick={() => choose(o)} title={o.base_url ?? o.model}>
          <span className="mp-check">{isActive && <Check size={13} />}</span>
          <span className="mp-runtime">{SHORT[o.provider]}</span>
          <span className="mp-name">{shortModel(o.provider, o.model)}</span>
          {!o.detected && <span className="pill">{T("modelpicker.saved")}</span>}
        </button>
        {!isActive && (
          <button
            className={`ghost mp-fb ${fb >= 0 ? "on" : ""}`}
            title={fb >= 0 ? T("modelpicker.fallbackNth", { n: fb + 1 }) : T("modelpicker.fallbackUse")}
            onClick={() => toggleFallback(o)}
          >
            <LifeBuoy size={13} />
            {fb >= 0 && <span className="small">{fb + 1}</span>}
          </button>
        )}
        {!o.detected && !isActive && (
          <button className="ghost" title={T("modelpicker.forget")} onClick={() => forget(o)}>
            <Trash2 size={13} />
          </button>
        )}
      </li>
    );
  };

  const go = (fn: () => void) => () => {
    setOpen(false);
    fn();
  };

  return (
    <div className="mp" ref={root}>
      <button className={`mp-trigger ${isLocal ? "local" : "cloud"}`} onClick={() => setOpen((v) => !v)} title={T("modelpicker.choose")}>
        {isLocal ? <HardDrive size={13} /> : <Cloud size={13} />}
        <span className="mp-trigger-name">
          {active ? `${SHORT[active.provider]} · ${shortModel(active.provider, active.model) || T("modelpicker.noModel")}` : T("modelpicker.choose")}
        </span>
        {fallbacks.length > 0 && (
          <span className="mp-trigger-fb" title={T("modelpicker.fallbackChain", { chain: fallbacks.join(" → ") })}>
            <LifeBuoy size={11} /> {fallbacks.length}
          </span>
        )}
        <ChevronDown size={13} />
      </button>

      {open && (
        <div className="mp-pop">
          <div className="mp-search">
            {loading ? <LoaderCircle size={13} className="spin muted" /> : <Search size={13} className="muted" />}
            <input autoFocus value={query} placeholder={T("modelpicker.search")} onChange={(e) => setQuery(e.target.value)} />
          </div>

          <div className="mp-scroll">
            <h4>
              <HardDrive size={12} /> {T("modelpicker.onDevice")}
            </h4>
            {localOptions.length > 0 ? (
              <ul>{localOptions.map(row)}</ul>
            ) : (
              <p className="mp-empty">
                {q ? T("modelpicker.noMatches") : T("modelpicker.noLocal")}{" "}
                <button className="link" onClick={go(onOpenModels)}>
                  {T("modelpicker.downloadOne")}
                </button>
              </p>
            )}

            <h4>
              <Cloud size={12} /> {T("modelpicker.cloud")}
            </h4>
            {s.local_only ? (
              <p className="mp-empty">
                {T("modelpicker.localOnlyNotice")}{" "}
                {!s.forced_local_only && (
                  <button className="link" onClick={go(onOpenSettings)}>
                    {T("modelpicker.changeInSettings")}
                  </button>
                )}
              </p>
            ) : (
              <>
                {cloudOptions.length > 0 ? <ul>{cloudOptions.map(row)}</ul> : q && <p className="mp-empty">{T("modelpicker.noMatches")}</p>}
                {missingKeys.length > 0 && !q && (
                  <p className="mp-empty">
                    <KeyRound size={12} /> {T("modelpicker.missingKeys", { providers: missingKeys.map((p) => SHORT[p]).join(", ") })}{" "}
                    <button className="link" onClick={go(onOpenSettings)}>
                      {T("modelpicker.addKey")}
                    </button>
                  </p>
                )}
              </>
            )}

            {custom ? (
              <form
                className="mp-custom"
                onSubmit={(e) => {
                  e.preventDefault();
                  const model = custom.model.trim();
                  if (!model) return;
                  choose({ provider: custom.provider, model, base_url: custom.provider === "openai_compatible" ? custom.base_url.trim() || null : null });
                  setCustom(null);
                }}
              >
                <select value={custom.provider} onChange={(e) => setCustom({ ...custom, provider: e.target.value as Provider })}>
                  {PROVIDERS.filter((p) => p.local || !s.local_only).map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </select>
                <input
                  autoFocus
                  value={custom.model}
                  placeholder={PROVIDERS.find((p) => p.id === custom.provider)?.example}
                  onChange={(e) => setCustom({ ...custom, model: e.target.value })}
                />
                {custom.provider === "openai_compatible" && (
                  <input value={custom.base_url} placeholder="http://localhost:8000/v1" onChange={(e) => setCustom({ ...custom, base_url: e.target.value })} />
                )}
                <div className="row">
                  <button type="submit" className="primary" disabled={!custom.model.trim()}>
                    {T("modelpicker.use")}
                  </button>
                  <button type="button" className="ghost" onClick={() => setCustom(null)}>
                    {T("modelpicker.cancel")}
                  </button>
                </div>
              </form>
            ) : (
              <button className="ghost mp-add" onClick={() => setCustom({ provider: "openai_compatible", model: "", base_url: "" })}>
                <Plus size={13} /> {T("modelpicker.other")}
              </button>
            )}
          </div>

          <footer className="mp-foot">
            <button
              className="ghost"
              disabled={!active || test === T("modelpicker.testing")}
              onClick={async () => {
                setTest(T("modelpicker.testing"));
                const r = await api.testModel(active);
                setTest(r.ok ? `✓ ${T("modelpicker.replies", { reply: r.reply ?? "" })}` : `✗ ${r.error}`);
              }}
            >
              {T("modelpicker.test")}
            </button>
            <span className="spacer" />
            <button className="ghost" onClick={go(onOpenModels)}>
              <Boxes size={13} /> {T("modelpicker.localModels")}
            </button>
            <button className="ghost" onClick={go(onOpenSettings)}>
              <Settings size={13} /> {T("modelpicker.apiKeys")}
            </button>
          </footer>
          {test && <p className={`mp-test small ${test.startsWith("✗") ? "bad" : ""}`}>{test}</p>}
        </div>
      )}
    </div>
  );
}
