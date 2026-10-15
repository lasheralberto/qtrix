// Renderers for each block kind of the dynamic analysis panel.
import katex from "katex";
import { Check, Code2, Copy, MessageSquarePlus, Pencil, Pin, PinOff, Snowflake, X } from "lucide-react";
import Plotly from "plotly.js-dist-min";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { api, type Block } from "./api";
import { T } from "./i18n";
import { NoteEditor, PythonCellBody } from "./notebook";

const KIND_LABEL: Record<Block["kind"], string> = {
  regression_table: T("blocks.kindRegression"),
  plotly_figure: T("blocks.kindPlot"),
  dataframe: T("blocks.kindTable"),
  diagnostics: T("blocks.kindDiagnostics"),
  markdown: T("blocks.kindNote"),
  equation: T("blocks.kindEquation"),
  python_cell: T("blocks.kindPython"),
};

const fmt = (v: unknown, digits = 3) => {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e9) return v.toLocaleString("es-ES");
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return v.toFixed(digits);
};

const stars = (p: number | null | undefined) =>
  p == null ? "" : p < 0.01 ? "***" : p < 0.05 ? "**" : p < 0.1 ? "*" : "";

function RegressionTable({ data }: { data: Block["data"] }) {
  const models: any[] = data.models ?? [];
  const vars = useMemo(() => {
    const seen: string[] = [];
    for (const m of models) for (const c of m.coefficients ?? []) if (!seen.includes(c.name)) seen.push(c.name);
    return seen;
  }, [models]);
  const stat = (key: string, label: string, digits = 3) =>
    models.some((m) => m[key] != null) ? (
      <tr className="stat-row">
        <th>{label}</th>
        {models.map((m, i) => (
          <td key={i}>{fmt(m[key], digits)}</td>
        ))}
      </tr>
    ) : null;
  return (
    <div className="table-scroll">
      <table className="reg-table">
        <thead>
          <tr>
            <th />
            {models.map((m, i) => (
              <th key={i}>
                <div className="model-head">({i + 1}) {m.model}</div>
                <div className="model-dep">{m.dep_var}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {vars.map((v) => (
            <tr key={v}>
              <th>{v}</th>
              {models.map((m, i) => {
                const c = (m.coefficients ?? []).find((x: any) => x.name === v);
                return (
                  <td key={i}>
                    {c ? (
                      <>
                        <div className={c.p < 0.05 ? "coef sig" : "coef"}>
                          {fmt(c.coef, 4)}
                          <sup>{stars(c.p)}</sup>
                        </div>
                        <div className="se">({fmt(c.se, 4)})</div>
                      </>
                    ) : null}
                  </td>
                );
              })}
            </tr>
          ))}
          {stat("n_obs", T("blocks.observations"), 0)}
          {stat("n_entities", T("blocks.entities"), 0)}
          {stat("r2", "R²")}
          {stat("adj_r2", T("blocks.adjR2"))}
          {stat("r2_within", "R² within")}
          {stat("pseudo_r2", "Pseudo R²")}
          {stat("f_stat", T("blocks.fStat"), 2)}
          {stat("aic", "AIC", 1)}
          <tr className="stat-row">
            <th>{T("blocks.stdErrors")}</th>
            {models.map((m, i) => (
              <td key={i}>{m.cov_type}</td>
            ))}
          </tr>
          {models.some((m) => m.sample) && (
            <tr className="stat-row sample-row">
              <th>{T("blocks.sample")}</th>
              {models.map((m, i) => (
                <td key={i}>{m.sample}</td>
              ))}
            </tr>
          )}
        </tbody>
      </table>
      <p className="table-note">{T("blocks.stdErrorsNote")} * p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01</p>
      {models.some((m) => m.iv_diagnostics?.first_stage) && <IvDiagnostics models={models} />}
    </div>
  );
}

function IvDiagnostics({ models }: { models: any[] }) {
  return (
    <div className="iv-diag">
      {models.map((m, i) =>
        m.iv_diagnostics?.first_stage ? (
          <div key={i}>
            <strong>{T("blocks.firstStage", { n: i + 1 })}</strong>
            {Object.entries(m.iv_diagnostics.first_stage).map(([k, v]: [string, any]) => (
              <span key={k} className={v.f_stat < 10 ? "pill warn" : "pill ok"}>
                {k}: F={fmt(v.f_stat, 1)} · {T("blocks.partialR2")}={fmt(v.partial_r2)}
              </span>
            ))}
          </div>
        ) : null,
      )}
    </div>
  );
}

export function PlotlyFigure({ data }: { data: Block["data"] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const layout = {
      ...(data.layout ?? {}),
      autosize: true,
      height: data.layout?.height ?? 360,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, Segoe UI, system-ui, sans-serif", size: 12, color: "#2a3345" },
      colorway: ["#2f62d6", "#d0663a", "#1f9e89", "#b08900", "#7a4cb8", "#5b6b8c"],
      xaxis: { gridcolor: "#e6eaf1", zerolinecolor: "#c9d1de", ...(data.layout?.xaxis ?? {}) },
      yaxis: { gridcolor: "#e6eaf1", zerolinecolor: "#c9d1de", ...(data.layout?.yaxis ?? {}) },
      title: undefined,
      template: undefined,
    };
    Plotly.react(el, data.data ?? [], layout, { responsive: true, displaylogo: false });
    const ro = new ResizeObserver(() => Plotly.Plots.resize(el));
    ro.observe(el);
    return () => {
      ro.disconnect();
      Plotly.purge(el);
    };
  }, [data]);
  return <div ref={ref} className="plot" />;
}

function DataFrame({ data }: { data: Block["data"] }) {
  return (
    <div className="table-scroll tall">
      <table className="df-table">
        <thead>
          <tr>
            {(data.columns as string[]).map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(data.rows as unknown[][]).map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className={typeof v === "number" ? "num" : ""}>
                  {fmt(v, 4)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.total_rows > (data.rows?.length ?? 0) && (
        <p className="table-note">
          {T("blocks.showingRows", { shown: data.rows.length, total: data.total_rows.toLocaleString("es-ES") })}
        </p>
      )}
    </div>
  );
}

function Diagnostics({ data }: { data: Block["data"] }) {
  return (
    <div className="diag">
      <table className="df-table">
        <thead>
          <tr>
            <th>{T("blocks.test")}</th>
            <th>{T("blocks.statistic")}</th>
            <th>{T("blocks.pValue")}</th>
            <th>H₀</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {(data.tests ?? []).map((t: any) => (
            <tr key={t.test}>
              <td>{t.test}</td>
              <td className="num">{fmt(t.statistic)}</td>
              <td className="num">{fmt(t.p_value)}</td>
              <td className="muted">{t.null_hypothesis}</td>
              <td>
                {t.reject_5pct == null ? null : t.reject_5pct ? (
                  <span className="pill warn">{T("blocks.rejectsH0")}</span>
                ) : (
                  <span className="pill ok">{T("blocks.doesNotReject")}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {data.vif?.length ? (
        <div className="vif">
          <strong>VIF</strong>
          {data.vif.map((v: any) => (
            <span key={v.variable} className={v.vif > 10 ? "pill warn" : "pill"}>
              {v.variable} {fmt(v.vif, 2)}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Equation({ data }: { data: Block["data"] }) {
  const html = useMemo(
    () => katex.renderToString(data.latex ?? "", { displayMode: true, throwOnError: false }),
    [data.latex],
  );
  return (
    <div className="equation">
      <div dangerouslySetInnerHTML={{ __html: html }} />
      {data.description && <p className="muted">{data.description}</p>}
    </div>
  );
}

/** ``components`` overrides how elements render (the plan turns tool names into hover help). */
export function Markdown({ text, components }: { text: string; components?: Components }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}

export function BlockBody({ block }: { block: Block }) {
  switch (block.kind) {
    case "regression_table":
      return <RegressionTable data={block.data} />;
    case "plotly_figure":
      return <PlotlyFigure data={block.data} />;
    case "dataframe":
      return <DataFrame data={block.data} />;
    case "diagnostics":
      return <Diagnostics data={block.data} />;
    case "equation":
      return <Equation data={block.data} />;
    case "markdown":
      return <Markdown text={block.data.markdown ?? ""} />;
    case "python_cell":
      return <PythonCellBody block={block} />;
  }
}

export function BlockCard(props: {
  block: Block;
  index: number;
  fresh: boolean;
  onPin(pinned: boolean): void;
  onRemove(): void;
  onAsk(text: string): void;
  /** Focus the composer with this block as the subject of the next question. */
  onAskAbout(): void;
  onFreeze(): void;
}) {
  const { block, index, fresh } = props;
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);
  const [editing, setEditing] = useState(false);
  const mine = block.source === "user";
  const cell = block.kind === "python_cell";
  const wide = block.kind === "regression_table" || block.kind === "dataframe" || cell;
  return (
    <article
      id={`block-${block.id}`}
      className={`block kind-${block.kind} ${wide ? "wide" : ""} ${fresh ? "fresh" : ""} ${block.pinned ? "pinned" : ""} ${mine ? "mine" : ""}`}
    >
      <header>
        <span className="block-index">{index}</span>
        <span className="block-kind">{KIND_LABEL[block.kind]}</span>
        <h3 title={block.title}>{block.title}</h3>
        {block.ref ? (
          <span className="pill live" title={T("blocks.liveTitle", { object: block.ref.object })}>
            {T("blocks.live")}
          </span>
        ) : block.result_id ? (
          <span className="pill" title={T("blocks.frozenTitle")}>
            {T("blocks.frozen")}
          </span>
        ) : null}
        {mine && (
          <span className="pill mine" title={T("blocks.mineTitle")}>
            {T("blocks.mine")}
          </span>
        )}
        {block.stale && (
          <span className="pill warn" title={T("blocks.staleTitle")}>
            {T("blocks.stale")}
          </span>
        )}
        <div className="block-actions">
          <button title={T("blocks.askAbout")} onClick={props.onAskAbout}>
            <MessageSquarePlus size={15} />
          </button>
          {block.ref && (
            <button title={T("blocks.freeze")} onClick={props.onFreeze}>
              <Snowflake size={15} />
            </button>
          )}
          {mine && block.kind === "markdown" && !editing && (
            <button title={T("blocks.editNote")} onClick={() => setEditing(true)}>
              <Pencil size={15} />
            </button>
          )}
          {block.code && !cell && (
            <button title={T("blocks.showCode")} onClick={() => setShowCode((s) => !s)} className={showCode ? "on" : ""}>
              <Code2 size={15} />
            </button>
          )}
          <button title={block.pinned ? T("blocks.unpin") : T("blocks.pin")} onClick={() => props.onPin(!block.pinned)}>
            {block.pinned ? <PinOff size={15} /> : <Pin size={15} />}
          </button>
          <button title={T("blocks.remove")} onClick={props.onRemove}>
            <X size={15} />
          </button>
        </div>
      </header>
      {showCode && block.code && (
        <div className="code">
          <button
            className="copy"
            onClick={() => {
              navigator.clipboard.writeText(block.code ?? "");
              setCopied(true);
              setTimeout(() => setCopied(false), 1200);
            }}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
          <pre>{block.code}</pre>
        </div>
      )}
      <div className={`block-body ${cell ? "flush" : ""}`}>
        {editing ? (
          <NoteEditor
            title={block.title}
            markdown={block.data.markdown ?? ""}
            onSave={(t, m) => api.editNote(block.id, t, m).then(() => setEditing(false))}
            onCancel={() => setEditing(false)}
          />
        ) : (
          <BlockBody block={block} />
        )}
      </div>
      {block.result_id && block.kind === "regression_table" && (
        <footer>
          <button onClick={() => props.onAsk(T("blocks.askDiagnostics", { id: block.result_id! }))}>{T("blocks.diagnostics")}</button>
          <button onClick={() => props.onAsk(T("blocks.askCoefplot", { id: block.result_id! }))}>{T("blocks.coefplot")}</button>
          <button onClick={() => props.onAsk(T("blocks.askRobustness", { id: block.result_id! }))}>{T("blocks.robustness")}</button>
        </footer>
      )}
    </article>
  );
}
