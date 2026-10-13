// Renderers for each block kind of the dynamic analysis panel.
import katex from "katex";
import { Check, Code2, Copy, MessageSquarePlus, Pencil, Pin, PinOff, Snowflake, X } from "lucide-react";
import Plotly from "plotly.js-dist-min";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { api, type Block } from "./api";
import { NoteEditor, PythonCellBody } from "./notebook";

const KIND_LABEL: Record<Block["kind"], string> = {
  regression_table: "Regresión",
  plotly_figure: "Gráfico",
  dataframe: "Tabla",
  diagnostics: "Diagnóstico",
  markdown: "Nota",
  equation: "Ecuación",
  python_cell: "Python",
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
          {stat("n_obs", "Observaciones", 0)}
          {stat("n_entities", "Entidades", 0)}
          {stat("r2", "R²")}
          {stat("adj_r2", "R² ajustado")}
          {stat("r2_within", "R² within")}
          {stat("pseudo_r2", "Pseudo R²")}
          {stat("f_stat", "Estadístico F", 2)}
          {stat("aic", "AIC", 1)}
          <tr className="stat-row">
            <th>Errores estándar</th>
            {models.map((m, i) => (
              <td key={i}>{m.cov_type}</td>
            ))}
          </tr>
          {models.some((m) => m.sample) && (
            <tr className="stat-row sample-row">
              <th>Muestra</th>
              {models.map((m, i) => (
                <td key={i}>{m.sample}</td>
              ))}
            </tr>
          )}
        </tbody>
      </table>
      <p className="table-note">Errores estándar entre paréntesis. * p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01</p>
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
            <strong>Primera etapa ({i + 1})</strong>
            {Object.entries(m.iv_diagnostics.first_stage).map(([k, v]: [string, any]) => (
              <span key={k} className={v.f_stat < 10 ? "pill warn" : "pill ok"}>
                {k}: F={fmt(v.f_stat, 1)} · R² parcial={fmt(v.partial_r2)}
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
          Mostrando {data.rows.length} de {data.total_rows.toLocaleString("es-ES")} filas
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
            <th>Contraste</th>
            <th>Estadístico</th>
            <th>p-valor</th>
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
                  <span className="pill warn">rechaza H₀</span>
                ) : (
                  <span className="pill ok">no rechaza</span>
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

export function Markdown({ text }: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
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
          <span className="pill live" title={`Vista viva de ${block.ref.object}: se actualiza al re-estimar`}>
            ● vivo
          </span>
        ) : block.result_id ? (
          <span className="pill" title="Congelado: no sigue los cambios del objeto">
            congelado
          </span>
        ) : null}
        {mine && (
          <span className="pill mine" title="Lo escribiste tú: el agente lo lee pero no lo modifica">
            tuyo
          </span>
        )}
        {block.stale && (
          <span className="pill warn" title="El objeto cambió después de calcular este bloque">
            desactualizado
          </span>
        )}
        <div className="block-actions">
          <button title="Preguntar al agente sobre este bloque" onClick={props.onAskAbout}>
            <MessageSquarePlus size={15} />
          </button>
          {block.ref && (
            <button title="Congelar (deja de seguir al objeto)" onClick={props.onFreeze}>
              <Snowflake size={15} />
            </button>
          )}
          {mine && block.kind === "markdown" && !editing && (
            <button title="Editar nota" onClick={() => setEditing(true)}>
              <Pencil size={15} />
            </button>
          )}
          {block.code && !cell && (
            <button title="Ver código reproducible" onClick={() => setShowCode((s) => !s)} className={showCode ? "on" : ""}>
              <Code2 size={15} />
            </button>
          )}
          <button title={block.pinned ? "Desfijar" : "Fijar (el agente no podrá modificarlo)"} onClick={() => props.onPin(!block.pinned)}>
            {block.pinned ? <PinOff size={15} /> : <Pin size={15} />}
          </button>
          <button title="Eliminar" onClick={props.onRemove}>
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
          <button onClick={() => props.onAsk(`Haz los diagnósticos del modelo ${block.result_id} e interprétalos`)}>Diagnósticos</button>
          <button onClick={() => props.onAsk(`Muestra el gráfico de coeficientes del modelo ${block.result_id}`)}>Coefplot</button>
          <button onClick={() => props.onAsk(`Comprueba la robustez del modelo ${block.result_id} con especificaciones alternativas`)}>Robustez</button>
        </footer>
      )}
    </article>
  );
}
