// Autocomplete for the composer: as the user types a word, it suggests the analysis methods of the catalog (GET /methods),
// their qx tools and the variables of the loaded datasets that match it. The entry under the cursor gets a help popup
// beside the list (the method's description, a tool's signature, a variable's profile). Suggestions only fill the
// message in: the agent is still the one that runs anything.
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import {
  api,
  type AnalysisMethod,
  type ColumnProfile,
  type Dataset,
  type MethodArea,
  type PageKind,
  type ToolHelp,
} from "./api";
import { T } from "./i18n";

export interface Suggestion {
  text: string;
  kind: "variable" | "method" | "tool";
  /** Where it comes from: the dataset (and type) of a variable, the area or method of a method. */
  detail: string;
  /** The method it is, or belongs to (a variant or a tool of it). */
  method?: AnalysisMethod;
  area?: MethodArea;
  /** The datasets that hold this variable. */
  datasets?: Dataset[];
  type?: string;
}

const KIND_LABEL: Record<Suggestion["kind"], string> = {
  variable: T("suggest.kindVariable"),
  method: T("suggest.kindMethod"),
  tool: T("suggest.kindTool"),
};

const STRUCTURE_LABEL: Record<PageKind, string> = {
  cross_section: T("suggest.structCrossSection"),
  time_series: T("suggest.structTimeSeries"),
  panel: T("suggest.structPanel"),
  panel_undated: T("suggest.structPanel"),
};

const MAX_SUGGESTIONS = 8;
const WORD = /[\p{L}\p{N}_]+$/u;

const fold = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();

/** Everything that can be suggested: the variables of the datasets and the methods of the catalog (with their qx tools). */
export function useSuggestionSource(areas: MethodArea[] | null, datasets: Dataset[]) {
  return useMemo(() => {
    const out: Suggestion[] = [];
    const seen = new Set<string>();
    const add = (s: Suggestion) => {
      const key = `${s.kind}:${s.text}`;
      if (seen.has(key)) return;
      seen.add(key);
      out.push(s);
    };
    // A variable present in several datasets is listed once, with all of them.
    const vars = new Map<string, { type: string; from: Dataset[] }>();
    for (const d of datasets)
      for (const c of d.columns) {
        const v = vars.get(c.name) ?? { type: c.type, from: [] };
        v.from.push(d);
        vars.set(c.name, v);
      }
    for (const [name, v] of vars)
      add({
        text: name,
        kind: "variable",
        detail: `${v.from.map((d) => d.name).join(", ")} · ${v.type}`,
        datasets: v.from,
        type: v.type,
      });
    for (const area of areas ?? [])
      for (const method of area.methods) {
        add({ text: method.label, kind: "method", detail: area.label, method, area });
        for (const variant of method.variants)
          add({ text: variant, kind: "method", detail: method.label, method, area });
        for (const t of method.tools) add({ text: t, kind: "tool", detail: method.label, method, area });
      }
    return out.map((s) => ({ s, folded: fold(s.text), words: fold(s.text).split(/[\s_\-()/,.:]+/).filter(Boolean) }));
  }, [areas, datasets]);
}

export type SuggestionSource = ReturnType<typeof useSuggestionSource>;

/** The word being typed just before the caret, if any (a bare number is not one). */
export function wordBefore(text: string, caret: number) {
  const m = WORD.exec(text.slice(0, caret));
  if (!m || /^\d+$/.test(m[0])) return null;
  return { word: m[0], start: caret - m[0].length };
}

/** The entries matching ``word``: those that start with it first, then those with a word that does, then (from three
 * letters) those that contain it. Variables come before methods at equal rank. */
export function suggest(source: SuggestionSource, word: string): Suggestion[] {
  const q = fold(word);
  const ranked: { s: Suggestion; rank: number }[] = [];
  for (const { s, folded, words } of source) {
    if (folded === q) continue;
    const rank = folded.startsWith(q)
      ? 0
      : words.some((w) => w.startsWith(q))
        ? 1
        : q.length >= 3 && folded.includes(q)
          ? 2
          : -1;
    if (rank >= 0) ranked.push({ s, rank });
  }
  const kindOrder = { variable: 0, method: 1, tool: 2 };
  ranked.sort(
    (a, b) =>
      a.rank - b.rank ||
      kindOrder[a.s.kind] - kindOrder[b.s.kind] ||
      a.s.text.length - b.s.text.length ||
      a.s.text.localeCompare(b.s.text),
  );
  return ranked.slice(0, MAX_SUGGESTIONS).map((r) => r.s);
}

function Highlighted({ text, word }: { text: string; word: string }) {
  const at = fold(text).indexOf(fold(word));
  if (at < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, at)}
      <b>{text.slice(at, at + word.length)}</b>
      {text.slice(at + word.length)}
    </>
  );
}

// ---- help popup ---------------------------------------------------------------------------

type Loaded<T> = { state: "loading" } | { state: "error" } | { state: "ready"; value: T };

const cache = new Map<string, Promise<unknown>>();

/** A value fetched once per key (the popup is reopened on every keystroke) and shared by whoever asks for it again. */
function useCached<T>(key: string, load: () => Promise<T>): Loaded<T> {
  const [res, setRes] = useState<Loaded<T>>({ state: "loading" });
  useEffect(() => {
    let cancelled = false;
    setRes({ state: "loading" });
    let p = cache.get(key) as Promise<T> | undefined;
    if (!p) {
      p = load();
      cache.set(key, p);
      p.catch(() => cache.delete(key)); // a failed fetch is retried the next time
    }
    p.then(
      (value) => !cancelled && setRes({ state: "ready", value }),
      () => !cancelled && setRes({ state: "error" }),
    );
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return res;
}

const num = (n: number | null | undefined) =>
  n === null || n === undefined ? "–" : Number.isInteger(n) ? n.toLocaleString() : `${+n.toPrecision(4)}`;

function MethodHelp({ s, datasets }: { s: Suggestion; datasets: Dataset[] }) {
  const m = s.method!;
  const kinds = new Set(datasets.map((d) => d.kind));
  const structures = m.structures ? [...new Set(m.structures.map((k) => STRUCTURE_LABEL[k]))] : null;
  const unmet = !!m.structures && kinds.size > 0 && !m.structures.some((k) => kinds.has(k));
  const variant = s.text !== m.label ? s.text : null;
  return (
    <>
      <header>
        <strong>{variant ?? m.label}</strong>
        <span className="suggest-chip">{variant ? T("suggest.variantOf", { method: m.label }) : s.area?.label}</span>
      </header>
      <p>{m.summary}</p>
      {m.variants.length > 0 && (
        <section>
          <h4>{T("suggest.variants")}</h4>
          <div className="suggest-tags">
            {m.variants.map((v) => (
              <span key={v} className={v === variant ? "on" : ""}>
                {v}
              </span>
            ))}
          </div>
        </section>
      )}
      <section>
        <h4>{T("suggest.requestExample")}</h4>
        <blockquote>{m.prompt}</blockquote>
      </section>
      <section>
        <h4>{T("suggest.tools")}</h4>
        <code>{m.tools.join(" · ")}</code>
        {m.library.length > 0 && <p className="small">{T("suggest.python", { libraries: m.library.join(", ") })}</p>}
      </section>
      {structures && (
        <p className={`small ${unmet ? "unmet" : ""}`}>
          {T("suggest.requiresData", { structures: structures.join(T("suggest.or")) })}
          {unmet && T("suggest.noDatasetMatches")}
        </p>
      )}
      {!m.available && <p className="small unmet">{T("suggest.unavailable", { missing: m.missing.join(", ") })}</p>}
    </>
  );
}

function ToolHelpView({ s }: { s: Suggestion }) {
  const res = useCached(`tool:${s.text}`, () => api.toolHelp(s.text));
  if (res.state !== "ready")
    return (
      <>
        <header>
          <strong>{s.text}</strong>
          <span className="suggest-chip">{T("suggest.qxTool")}</span>
        </header>
        <p>{res.state === "loading" ? T("suggest.loadingSignature") : T("suggest.signatureFailed")}</p>
        {s.method && <p className="small">{T("suggest.method", { method: s.method.label })}</p>}
      </>
    );
  const t: ToolHelp = res.value;
  const call = `${t.name}(${t.params.map((p) => (p.required ? p.name : `${p.name}=${p.default ?? "None"}`)).join(", ")})`;
  return (
    <>
      <header>
        <strong>{t.name}</strong>
        <span className="suggest-chip">{T("suggest.qxTool")}</span>
      </header>
      {t.summary && <p>{t.summary}</p>}
      <section>
        <h4>{T("suggest.signature")}</h4>
        <pre>{call}</pre>
        {t.qx && <p className="small">{T("suggest.inPython", { qx: t.qx })}</p>}
      </section>
      {t.params.length > 0 && (
        <section>
          <h4>{T("suggest.parameters")}</h4>
          <dl>
            {t.params.map((p) => (
              <div key={p.name}>
                <dt>
                  {p.name}
                  {p.type && <span className="suggest-type">{p.type}</span>}
                  {p.required && <span className="suggest-req">{T("suggest.required")}</span>}
                </dt>
                {p.doc && <dd>{p.doc}</dd>}
              </div>
            ))}
          </dl>
        </section>
      )}
      {s.method && <p className="small">{T("suggest.method", { method: s.method.label })}</p>}
      {t.kinds && (
        <p className="small">
          {T("suggest.data", {
            structures: [...new Set(t.kinds.map((k) => STRUCTURE_LABEL[k] ?? k))].join(T("suggest.or")),
          })}
        </p>
      )}
      {!t.available && <p className="small unmet">{T("suggest.unavailable", { missing: t.requires.join(", ") })}</p>}
    </>
  );
}

/** Whether each name is a registry tool (its help, or null when the engine does not know it), asked once per name. */
const toolLookup = new Map<string, Promise<ToolHelp | null>>();

/** A tool named in a text (the plan): hovering or focusing it shows the tool's help, the same one the autocomplete
 * shows. A name the engine does not know as a tool (a variable, say) stays plain code. */
export function ToolRef({ name }: { name: string }) {
  const [known, setKnown] = useState(false);
  const [at, setAt] = useState<DOMRect | null>(null);
  useEffect(() => {
    let cancelled = false;
    let p = toolLookup.get(name);
    if (!p) {
      p = api.toolHelp(name).catch(() => null);
      toolLookup.set(name, p);
    }
    p.then((h) => !cancelled && setKnown(h !== null));
    return () => {
      cancelled = true;
    };
  }, [name]);
  if (!known) return <code>{name}</code>;

  const show = (e: { currentTarget: Element }) => setAt(e.currentTarget.getBoundingClientRect());
  const room = at ? window.innerHeight - at.bottom : 0;
  // The popup goes in the body: the workspace clips and transforms whatever it contains.
  const popup = at && (
    <aside
      className="suggest-help tool tool-ref-pop"
      role="tooltip"
      style={{
        left: Math.max(12, Math.min(at.left, window.innerWidth - 372)),
        ...(room < 320 ? { bottom: window.innerHeight - at.top + 6 } : { top: at.bottom + 6 }),
      }}
    >
      <ToolHelpView s={{ text: name, kind: "tool" } as Suggestion} />
    </aside>
  );
  return (
    <span
      className="tool-ref"
      tabIndex={0}
      onMouseEnter={show}
      onMouseLeave={() => setAt(null)}
      onFocus={show}
      onBlur={() => setAt(null)}
    >
      <code>{name}</code>
      {popup && createPortal(popup, document.body)}
    </span>
  );
}

function VariableHelp({ s }: { s: Suggestion }) {
  const sets = (s.datasets ?? []).slice(0, 3);
  return (
    <>
      <header>
        <strong>{s.text}</strong>
        <span className="suggest-chip">{T("suggest.variableChip", { type: s.type ?? "" })}</span>
      </header>
      {sets.map((d) => (
        <VariableProfile key={d.id} name={s.text} dataset={d} />
      ))}
      {(s.datasets?.length ?? 0) > sets.length && (
        <p className="small">
          {T(s.datasets!.length - sets.length === 1 ? "suggest.moreDatasetsOne" : "suggest.moreDatasetsOther", {
            count: s.datasets!.length - sets.length,
          })}
        </p>
      )}
    </>
  );
}

function VariableProfile({ name, dataset }: { name: string; dataset: Dataset }) {
  const res = useCached(`profile:${dataset.id}:${dataset.n_rows}`, () =>
    api.profile(dataset.id).then((rows) => rows as ColumnProfile[]),
  );
  const col = res.state === "ready" ? res.value.find((c) => c.name === name) : undefined;
  const role =
    name === dataset.entity_col
      ? T("suggest.identifiesUnit")
      : name === dataset.time_col || name === dataset.date_col || name === dataset.t_col
        ? T("suggest.timeIndex")
        : null;
  return (
    <section>
      <h4>
        {dataset.name} <span className="muted">· {dataset.structure}</span>
      </h4>
      {res.state === "loading" && <p className="small">{T("suggest.computing")}</p>}
      {res.state === "error" && <p className="small">{T("suggest.profileFailed")}</p>}
      {col && (
        <dl className="suggest-stats">
          <div>
            <dt>{T("suggest.type")}</dt>
            <dd>{col.dtype}</dd>
          </div>
          <div>
            <dt>{T("suggest.observations")}</dt>
            <dd>
              {T("suggest.observationsOf", {
                count: (dataset.n_rows - col.missing).toLocaleString(),
                total: dataset.n_rows.toLocaleString(),
              })}
            </dd>
          </div>
          {col.missing > 0 && (
            <div>
              <dt>{T("suggest.missing")}</dt>
              <dd>{col.missing.toLocaleString()}</dd>
            </div>
          )}
          <div>
            <dt>{T("suggest.uniqueValues")}</dt>
            <dd>{col.unique.toLocaleString()}</dd>
          </div>
          {col.mean !== undefined && (
            <>
              <div>
                <dt>{T("suggest.mean")}</dt>
                <dd>{num(col.mean)}</dd>
              </div>
              <div>
                <dt>{T("suggest.sd")}</dt>
                <dd>{num(col.sd)}</dd>
              </div>
              <div>
                <dt>{T("suggest.minMax")}</dt>
                <dd>
                  {num(col.min)} – {num(col.max)}
                </dd>
              </div>
            </>
          )}
        </dl>
      )}
      {role && <p className="small">{role}</p>}
    </section>
  );
}

/** The help popup of the entry under the cursor: description for a method, signature for a tool, profile for a variable. */
function HelpPopup({ s, datasets }: { s: Suggestion; datasets: Dataset[] }) {
  return (
    <aside className={`suggest-help ${s.kind}`} aria-label={T("suggest.help", { text: s.text })}>
      {s.kind === "method" && s.method && <MethodHelp s={s} datasets={datasets} />}
      {s.kind === "tool" && <ToolHelpView s={s} />}
      {s.kind === "variable" && <VariableHelp s={s} />}
    </aside>
  );
}

/** The list that opens above the composer, with the help of its active entry beside it. Mouse down accepts (so the
 * textarea keeps its focus). */
export function SuggestionList({
  id,
  items,
  word,
  active,
  datasets,
  onHover,
  onPick,
}: {
  id: string;
  items: Suggestion[];
  word: string;
  active: number;
  datasets: Dataset[];
  onHover(i: number): void;
  onPick(s: Suggestion): void;
}) {
  useEffect(() => {
    document.getElementById(`${id}-${active}`)?.scrollIntoView?.({ block: "nearest" });
  }, [id, active]);
  const current = items[active];
  return (
    <div className="suggest-wrap">
      <ul className="suggest" id={id} role="listbox" aria-label={T("suggest.suggestions")}>
        {items.map((s, i) => (
          <li
            key={`${s.kind}:${s.text}`}
            id={`${id}-${i}`}
            role="option"
            aria-selected={i === active}
            className={`${i === active ? "on" : ""} ${s.kind}`}
            onMouseEnter={() => onHover(i)}
            onMouseDown={(e) => {
              e.preventDefault();
              onPick(s);
            }}
          >
            <span className="suggest-text">
              <Highlighted text={s.text} word={word} />
            </span>
            <span className="suggest-detail">{s.detail}</span>
            <span className="suggest-kind">{KIND_LABEL[s.kind]}</span>
          </li>
        ))}
      </ul>
      {current && <HelpPopup key={`${current.kind}:${current.text}`} s={current} datasets={datasets} />}
    </div>
  );
}

/** Which entry is highlighted: reset to the first whenever the list changes. */
export function useActiveIndex(items: Suggestion[]) {
  const [active, setActive] = useState(0);
  const key = items.map((s) => s.text).join("\u0000");
  useEffect(() => setActive(0), [key]);
  return [active, setActive] as const;
}
