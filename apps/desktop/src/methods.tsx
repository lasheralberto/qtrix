// The catalog of analysis methods (GET /methods): what the user can ask the agent for, browsed before the conversation.
// A method is never run from here: "Preguntar" writes its example request in the composer, so the agent stays the way
// to use it and the user only fills in the «…» parts.
import {
  Activity,
  ArrowRight,
  ChartColumn,
  Database,
  GitBranch,
  LayoutGrid,
  LibraryBig,
  Search,
  Sigma,
  TrendingUp,
  X,
  type LucideIcon,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { T } from "./i18n";
import { api, type AnalysisMethod, type Dataset, type MethodArea, type PageKind } from "./api";

const AREA_ICON: Record<string, LucideIcon> = {
  data: Database,
  cross_section: Sigma,
  panel: LayoutGrid,
  causal: GitBranch,
  time_series: TrendingUp,
  volatility: Activity,
  charts: ChartColumn,
};

const KIND_LABEL: Record<PageKind, string> = {
  cross_section: T("methods.kindCrossSection"),
  time_series: T("methods.kindTimeSeries"),
  panel: T("methods.kindPanel"),
  panel_undated: T("methods.kindPanel"),
};

/** The method catalog, fetched once; null while it loads (or if the engine is unreachable). */
export function useMethods(): MethodArea[] | null {
  const [areas, setAreas] = useState<MethodArea[] | null>(null);
  useEffect(() => {
    api.methods().then(setAreas).catch(() => setAreas(null));
  }, []);
  return areas;
}

/** The areas as a grid of cards for the empty conversation: a glance at what qtrix can do. */
export function MethodAreas({ areas, onOpen }: { areas: MethodArea[]; onOpen(areaId: string): void }) {
  return (
    <section className="method-areas" aria-label={T("methods.available")}>
      <h3>{T("methods.whatToAsk")}</h3>
      <div className="method-area-grid">
        {areas.map((a) => {
          const Icon = AREA_ICON[a.id] ?? LibraryBig;
          return (
            <button key={a.id} className="method-area-card" onClick={() => onOpen(a.id)}>
              <span className="method-area-icon">
                <Icon size={16} />
              </span>
              <strong>{a.label}</strong>
              <span className="muted small">{a.methods.map((m) => m.label).join(" · ")}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function requirement(m: AnalysisMethod, kinds: Set<PageKind>): { text: string; unmet: boolean } | null {
  if (!m.structures) return null;
  const labels = [...new Set(m.structures.map((k) => KIND_LABEL[k]))];
  const unmet = kinds.size > 0 && !m.structures.some((k) => kinds.has(k));
  return { text: T("methods.requires", { kinds: labels.join(T("methods.or")) }), unmet };
}

function matches(m: AnalysisMethod, area: MethodArea, q: string): boolean {
  const hay = [m.label, m.summary, area.label, ...m.variants, ...m.tools, ...m.library].join(" ").toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .every((w) => hay.includes(w));
}

function MethodCard({ method, kinds, onUse }: { method: AnalysisMethod; kinds: Set<PageKind>; onUse(): void }) {
  const req = requirement(method, kinds);
  return (
    <article className={`method-card ${method.available ? "" : "off"}`}>
      <div className="method-main">
        <div className="method-title">
          <strong>{method.label}</strong>
          {method.library.length > 0 && (
            <span className="method-tag" title={T("methods.pythonLibrary", { libs: method.library.join(", ") })}>
              Python
            </span>
          )}
        </div>
        <p className="muted small">{method.summary}</p>
        {method.variants.length > 0 && (
          <div className="method-variants">
            {method.variants.map((v) => (
              <span key={v}>{v}</span>
            ))}
          </div>
        )}
        {req && (
          <p className={`method-req small ${req.unmet ? "unmet" : ""}`}>
            {req.text}
            {req.unmet && T("methods.noDatasetMatches")}
          </p>
        )}
        {!method.available && <p className="method-req unmet small">{T("methods.unavailable", { missing: method.missing.join(", ") })}</p>}
        <code className="method-tools" title={T("methods.engineTools")}>
          {method.tools.join(" · ")}
        </code>
      </div>
      <button className="method-use" disabled={!method.available} onClick={onUse} title={method.prompt}>
        {T("methods.ask")} <ArrowRight size={13} />
      </button>
    </article>
  );
}

/** The full catalog: areas on the left, their methods on the right, a search across all of them. */
export function MethodsModal({
  areas,
  datasets,
  initialArea,
  onUse,
  onClose,
}: {
  areas: MethodArea[];
  datasets: Dataset[];
  initialArea?: string | null;
  onUse(prompt: string): void;
  onClose(): void;
}) {
  const [areaId, setAreaId] = useState(initialArea ?? areas[0]?.id);
  const [query, setQuery] = useState("");
  const kinds = useMemo(() => new Set(datasets.map((d) => d.kind)), [datasets]);
  const q = query.trim();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const shown = q
    ? areas.map((a) => ({ ...a, methods: a.methods.filter((m) => matches(m, a, q)) })).filter((a) => a.methods.length)
    : areas.filter((a) => a.id === areaId);
  const total = areas.reduce((n, a) => n + a.methods.length, 0);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal wide methods-modal" role="dialog" aria-label={T("methods.title")} onClick={(e) => e.stopPropagation()}>
        <header>
          <LibraryBig size={17} />
          <h2>{T("methods.title")}</h2>
          <span className="muted small">{T("methods.count", { count: total })}</span>
          <label className="method-search">
            <Search size={14} />
            <input
              autoFocus
              value={query}
              placeholder={T("methods.searchPlaceholder")}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <button className="ghost" title={T("methods.close")} onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <div className="methods-body">
          <nav className="methods-nav" aria-label={T("methods.areas")}>
            {areas.map((a) => {
              const Icon = AREA_ICON[a.id] ?? LibraryBig;
              return (
                <button
                  key={a.id}
                  className={`ghost ${!q && a.id === areaId ? "on" : ""}`}
                  onClick={() => {
                    setQuery("");
                    setAreaId(a.id);
                  }}
                >
                  <Icon size={15} /> {a.label}
                  <span className="muted small">{a.methods.length}</span>
                </button>
              );
            })}
          </nav>
          <div className="methods-list">
            {shown.length === 0 && <p className="muted">{T("methods.noMatch", { query: q })}</p>}
            {shown.map((a) => (
              <section key={a.id}>
                <h3>{a.label}</h3>
                <p className="muted small">{a.summary}</p>
                {a.methods.map((m) => (
                  <MethodCard key={m.id} method={m} kinds={kinds} onUse={() => onUse(m.prompt)} />
                ))}
              </section>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
