// Dataset sidebar (import, list) and the dataset inspector modal.
import { Database, FileUp, Sparkles, Table2, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type Dataset, type Freq, type PageKind, type SampleInfo, type StructureSpec } from "./api";
import { T } from "./i18n";

const KIND_LABEL: Record<PageKind, string> = {
  cross_section: T("data.kindCrossSection"),
  time_series: T("data.kindTimeSeries"),
  panel: T("data.kindPanel"),
  panel_undated: T("data.kindPanelUndated"),
};
const FREQ_LABEL: Record<Exclude<Freq, "U">, string> = {
  A: T("data.freqAnnual"),
  S: T("data.freqSemiannual"),
  Q: T("data.freqQuarterly"),
  M: T("data.freqMonthly"),
  W: T("data.freqWeekly"),
  D5: T("data.freqDaily5"),
  D7: T("data.freqDaily7"),
};

function structureLabel(d: Dataset): string {
  if (d.kind === "cross_section") return T("data.structureCrossSection");
  if (d.kind === "panel_undated") return T("data.structurePanelUndated", { id: d.id_col!, t: d.t_col! });
  const freq = FREQ_LABEL[d.freq as Exclude<Freq, "U">]?.toLowerCase() ?? d.freq;
  const range = `${d.start}–${d.end}`;
  return d.kind === "panel" ? T("data.structurePanel", { id: d.id_col!, freq, range }) : `${freq} ${range}`;
}

const SAMPLE_HINT: Record<PageKind, string> = {
  cross_section: T("data.sampleHintCross"),
  time_series: T("data.sampleHintTs"),
  panel: "@all · 2002 2005 · if educ > 10",
  panel_undated: "@all · 2 3 · if score > 0",
};

/** The active sample: which rows every estimate and descriptive uses (ADR 0002 §2). */
function SampleEditor({ dataset, onSaved }: { dataset: Dataset; onSaved(): void }) {
  const [text, setText] = useState(dataset.sample ?? "");
  const [info, setInfo] = useState<SampleInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.sample(dataset.id).then(setInfo).catch((err) => setError(String(err.message ?? err)));
  }, [dataset.id, dataset.sample]);

  async function save(next: string | null) {
    setSaving(true);
    setError(null);
    try {
      setInfo(await api.setSample(dataset.id, next));
      if (next === null) setText("");
      onSaved();
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="sample">
      <div className="panel-structure">
        <span>{T("data.sample")}</span>
        <input
          className="sample-input"
          value={text}
          placeholder={SAMPLE_HINT[dataset.kind]}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && save(text.trim() || null)}
        />
        <button disabled={saving} onClick={() => save(text.trim() || null)}>
          {T("data.apply")}
        </button>
        {dataset.sample && (
          <button className="ghost" disabled={saving} onClick={() => save(null)}>
            {T("data.all")}
          </button>
        )}
      </div>
      {error ? <p className="error structure-error">{error}</p> : info && <p className="muted small sample-info">{info.description}</p>}
    </div>
  );
}

/** Declare what the data are. Time is optional: cross-sections never get dates (ADR 0002 §1). */
function StructureEditor({ dataset, onSaved }: { dataset: Dataset; onSaved(): void }) {
  const cols = dataset.columns.map((c) => c.name);
  const [spec, setSpec] = useState<StructureSpec>({
    kind: dataset.kind,
    freq: dataset.freq === "U" ? null : dataset.freq,
    date_col: dataset.date_col === "__obs" ? null : dataset.date_col,
    id_col: dataset.id_col,
    t_col: dataset.t_col,
    start: null,
  });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const set = (patch: Partial<StructureSpec>) => setSpec((s) => ({ ...s, ...patch }));
  const dated = spec.kind === "time_series" || spec.kind === "panel";
  const needsId = spec.kind === "panel" || spec.kind === "panel_undated";

  async function save(next: StructureSpec) {
    setSaving(true);
    setError(null);
    try {
      await api.setStructure(dataset.id, next);
      onSaved();
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setSaving(false);
    }
  }

  const colSelect = (value: string | null | undefined, onChange: (v: string | null) => void, placeholder: string) => (
    <select value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">{placeholder}</option>
      {cols.map((c) => (
        <option key={c}>{c}</option>
      ))}
    </select>
  );

  const s = dataset.suggestion;
  return (
    <div className="structure">
      {s && (
        <div className="structure-suggest">
          <Sparkles size={14} />
          <span>
            {T("data.looksLike")} <b>{KIND_LABEL[s.kind].toLowerCase()}</b> {FREQ_LABEL[s.freq as Exclude<Freq, "U">]?.toLowerCase()}{" "}
            {s.start}–{s.end}: {s.reason}.
          </span>
          <button
            disabled={saving}
            onClick={() => save({ kind: s.kind, freq: s.freq, date_col: s.date_col, id_col: s.id_col })}
          >
            {T("data.apply")}
          </button>
        </div>
      )}
      <div className="panel-structure">
        <span>{T("data.structure")}</span>
        <select value={spec.kind} onChange={(e) => set({ kind: e.target.value as PageKind })}>
          {(Object.keys(KIND_LABEL) as PageKind[]).map((k) => (
            <option key={k} value={k}>
              {KIND_LABEL[k]}
            </option>
          ))}
        </select>
        {needsId && colSelect(spec.id_col, (v) => set({ id_col: v }), T("data.entityPlaceholder"))}
        {dated && colSelect(spec.date_col, (v) => set({ date_col: v }), T("data.datePlaceholder"))}
        {spec.kind === "panel_undated" && colSelect(spec.t_col, (v) => set({ t_col: v }), T("data.timeIndexPlaceholder"))}
        {dated && (
          <select value={spec.freq ?? ""} onChange={(e) => set({ freq: (e.target.value || null) as Freq | null })}>
            <option value="">{T("data.frequencyAuto")}</option>
            {Object.entries(FREQ_LABEL).map(([f, label]) => (
              <option key={f} value={f}>
                {label}
              </option>
            ))}
          </select>
        )}
        {spec.kind === "time_series" && !spec.date_col && (
          <input
            className="short"
            placeholder={T("data.startPlaceholder")}
            value={spec.start ?? ""}
            onChange={(e) => set({ start: e.target.value || null })}
          />
        )}
        <button disabled={saving} onClick={() => save(spec)}>
          {T("data.save")}
        </button>
        <span className="muted small">{dataset.gaps > 0 && T("data.gapsAdded", { count: dataset.gaps })}</span>
      </div>
      {error && <p className="error structure-error">{error}</p>}
    </div>
  );
}

export function DatasetList(props: {
  datasets: Dataset[];
  onChange(): void;
  onOpen(d: Dataset): void;
  onError(msg: string): void;
}) {
  const [dragging, setDragging] = useState(false);
  const [importing, setImporting] = useState(false);

  async function importPaths(paths: string[]) {
    if (!paths.length) return;
    setImporting(true);
    try {
      for (const p of paths) await api.importPath(p);
      props.onChange();
    } catch (err) {
      props.onError(String((err as Error).message ?? err));
    } finally {
      setImporting(false);
    }
  }

  return (
    <div
      className={`datasets ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const paths = Array.from(e.dataTransfer.files).map((f) => window.qtrix?.pathForFile(f) ?? "");
        importPaths(paths.filter(Boolean));
      }}
    >
      <div className="side-title">
        <span>
          {T("data.title")} <span className="count">{props.datasets.length}</span>
        </span>
        <button
          className="ghost"
          title={T("data.importTitle")}
          disabled={importing}
          onClick={async () => importPaths((await window.qtrix?.openDataFiles()) ?? [])}
        >
          <FileUp size={15} />
        </button>
      </div>
      {props.datasets.length === 0 && (
        <p className="muted small drop-hint">{T("data.dropHint")}</p>
      )}
      <ul>
        {props.datasets.map((d) => (
          <li key={d.id}>
            <button className="dataset" title={d.structure} onClick={() => props.onOpen(d)}>
              <Database size={14} />
              <span className="ds-name">{d.name}</span>
              {d.suggestion && (
                <span className="ds-suggest" title={T("data.suggestTitle", { kind: KIND_LABEL[d.suggestion.kind].toLowerCase() })}>
                  {T("data.suggestMark")}
                </span>
              )}
              <span className="ds-meta" title={structureLabel(d)}>
                {d.n_rows.toLocaleString("es-ES")} × {d.columns.length}
              </span>
            </button>
          </li>
        ))}
      </ul>
      {importing && <p className="muted small">{T("data.importing")}</p>}
    </div>
  );
}

export function DatasetModal({ dataset, onClose, onChange }: { dataset: Dataset; onClose(): void; onChange(): void }) {
  const [tab, setTab] = useState<"preview" | "profile">("preview");
  const [preview, setPreview] = useState<{ columns: string[]; rows: unknown[][] } | null>(null);
  const [profile, setProfile] = useState<Record<string, any>[] | null>(null);

  useEffect(() => {
    api.preview(dataset.id).then(setPreview);
    api.profile(dataset.id).then(setProfile);
  }, [dataset.id]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal wide" onClick={(e) => e.stopPropagation()}>
        <header>
          <Table2 size={16} />
          <h2>{dataset.name}</h2>
          <span className="muted small">
            {T("data.datasetSummary", { rows: dataset.n_rows.toLocaleString("es-ES"), cols: dataset.columns.length, source: dataset.source })}
          </span>
          <button className="ghost" onClick={onClose}>
            <X size={16} />
          </button>
        </header>
        <StructureEditor dataset={dataset} onSaved={onChange} />
        <SampleEditor key={`${dataset.id}:${dataset.structure}`} dataset={dataset} onSaved={onChange} />
        <div className="panel-structure">
          <span className="muted small">{dataset.structure}</span>
          <span className="spacer" />
          <button
            className="danger ghost"
            onClick={() => {
              if (confirm(T("data.confirmDelete", { name: dataset.name }))) api.deleteDataset(dataset.id).then(() => (onChange(), onClose()));
            }}
          >
            <Trash2 size={14} /> {T("data.delete")}
          </button>
        </div>
        <nav className="tabs">
          <button className={tab === "preview" ? "on" : ""} onClick={() => setTab("preview")}>
            {T("data.preview")}
          </button>
          <button className={tab === "profile" ? "on" : ""} onClick={() => setTab("profile")}>
            {T("data.profile")}
          </button>
        </nav>
        <div className="table-scroll modal-table">
          {tab === "preview" && preview && (
            <table className="df-table">
              <thead>
                <tr>
                  {preview.columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((v, j) => (
                      <td key={j} className={typeof v === "number" ? "num" : ""}>
                        {v === null ? "—" : String(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {tab === "profile" && profile && (
            <table className="df-table">
              <thead>
                <tr>
                  {[T("data.colVariable"), T("data.colType"), T("data.colMissing"), T("data.colUnique"), T("data.colMean"), T("data.colSd"), T("data.colMin"), T("data.colMax")].map((h) => (
                    <th key={h}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profile.map((p) => (
                  <tr key={p.name}>
                    <td>{p.name}</td>
                    <td className="muted">{p.dtype}</td>
                    <td className="num">{p.missing}</td>
                    <td className="num">{p.unique}</td>
                    {["mean", "sd", "min", "max"].map((k) => (
                      <td key={k} className="num">
                        {p[k] == null ? "" : Number(p[k]).toPrecision(4)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
