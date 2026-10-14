// Agent actions as editable Python cells (ADR 0002 §7.3, E7·08b): canonical qx code, result, edit · re-run.
import {
  autocompletion,
  type Completion,
  type CompletionContext,
} from "@codemirror/autocomplete";
import { python } from "@codemirror/lang-python";
import { syntaxTree } from "@codemirror/language";
import { setDiagnostics, type Diagnostic } from "@codemirror/lint";
import { Compartment, EditorState, StateField } from "@codemirror/state";
import {
  Decoration,
  EditorView,
  keymap,
  MatchDecorator,
  showTooltip,
  ViewPlugin,
  type DecorationSet,
  type Tooltip,
  type ViewUpdate,
} from "@codemirror/view";
import type { SyntaxNode } from "@lezer/common";
import { basicSetup } from "codemirror";
import {
  Check,
  ChevronRight,
  CircleAlert,
  Code2,
  Copy,
  ExternalLink,
  Lightbulb,
  LoaderCircle,
  Pencil,
  Play,
  RotateCw,
  TriangleAlert,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type Block,
  type CellInfo,
  type CodeCheck,
  type CodeRun,
  type Completions,
  type QxFunction,
  type QxParam,
  type QxSignature,
} from "./api";
import { BlockBody, Markdown } from "./blocks";
import { T } from "./i18n";

// ---- completions -------------------------------------------------------------------------
let completionsCache: Promise<Completions> | null = null;
let completionsAt = 0;
/** The last names loaded, for what must answer synchronously (the signature tooltip). */
let loaded: Completions | null = null;

/** Names for autocompletion, refreshed at most every 10 s (objects and series change as the agent works). */
function loadCompletions(): Promise<Completions> {
  if (!completionsCache || Date.now() - completionsAt > 10_000) {
    completionsAt = Date.now();
    completionsCache = api
      .completions()
      .catch((): Completions => ({
        functions: { qx: [], page: [], obj: [], report: [] },
        objects: [],
        pages: [],
      }))
      .then((c) => (loaded = c));
  }
  return completionsCache;
}

// ---- signatures: which qx function a call is, which argument the cursor is on --------------
function paramText(p: QxParam) {
  return p.required || p.default === null ? p.name : `${p.name}=${p.default}`;
}

/** `(name, page, method='ls', formula, …)`: the short form shown next to a completion. */
function shortSignature(sig: QxSignature) {
  const shown = sig.params.slice(0, 4).map(paramText);
  return `(${shown.join(", ")}${sig.params.length > 4 ? ", …" : ""})`;
}

function el(tag: string, cls: string, text?: string) {
  const e = document.createElement(tag);
  e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** Every argument with its type, default and description: the info panel of a function completion. */
function functionInfo(fn: QxFunction) {
  const box = el("div", "qx-sig-info");
  const sig = fn.signatures[0];
  box.append(el("div", "qx-sig-doc", sig.doc));
  if (fn.signatures.length > 1)
    box.append(
      el(
        "div",
        "qx-sig-methods",
        "method = " + fn.signatures.map((s) => `'${s.method}'`).join(" | "),
      ),
    );
  const list = el("dl", "qx-sig-params");
  for (const p of sig.params) {
    const dt = el("dt", p.required ? "required" : "");
    dt.append(el("code", "", paramText(p)));
    if (p.type) dt.append(el("span", "qx-sig-type", p.type));
    list.append(dt, el("dd", "", p.doc));
  }
  box.append(list);
  return box;
}

/** The function a call names: qx.X, qx.report.X, qx.page(…).X, qx.obj(…).X or <variable>.X. */
function resolveCall(
  state: EditorState,
  call: SyntaxNode,
  names: Completions,
): QxFunction | null {
  const sigs = names.signatures;
  const callee = call.firstChild;
  if (!sigs || !callee || callee.name !== "MemberExpression") return null;
  const prop = callee.lastChild;
  const target = callee.firstChild;
  if (!prop || !target || prop.name !== "PropertyName") return null;
  const fn = state.sliceDoc(prop.from, prop.to);
  const text = state.sliceDoc(target.from, target.to);
  let on: string[];
  if (text === "qx") on = ["qx"];
  else if (text === "qx.report") on = ["report"];
  else if (text.startsWith("qx.page(")) on = ["page"];
  else if (text.startsWith("qx.obj(")) on = ["obj"];
  else if (target.name === "VariableName") {
    // eq01.estimate(…) · w = qx.page("wages"); w.generate(…)
    const isPage = new RegExp(`\\b${text}\\s*=\\s*qx\\.page\\(`).test(
      state.doc.toString(),
    );
    on = isPage
      ? ["page"]
      : names.objects.some((o) => o.name === text)
        ? ["obj"]
        : ["obj", "page"];
  } else return null;
  for (const o of on) if (sigs[`${o}.${fn}`]) return sigs[`${o}.${fn}`];
  return null;
}

interface ArgPosition {
  call: SyntaxNode;
  args: SyntaxNode;
  /** Keyword arguments already given, and how many positional ones come before the cursor. */
  used: Set<string>;
  positional: number;
  /** The keyword whose value the cursor is on (`formula=|`), else null. */
  keyword: string | null;
  /** Positional index of the argument the cursor is on. */
  index: number;
  /** The call's literal method="…", if any (picks a family's signature). */
  method: string | null;
}

/** Where the cursor is inside the argument list of the innermost call around it. */
function argPosition(state: EditorState, pos: number): ArgPosition | null {
  let node: SyntaxNode | null = syntaxTree(state).resolveInner(pos, -1);
  for (; node; node = node.parent) {
    const closed = node.lastChild?.name === ")";
    if (
      node.name === "ArgList" &&
      pos > node.from &&
      (pos < node.to || !closed)
    )
      break;
  }
  if (!node || node.parent?.name !== "CallExpression") return null;
  const used = new Set<string>();
  let index = 0;
  let positional = 0;
  let named = false; // the current argument is a keyword one
  let keyword: string | null = null;
  let method: string | null = null;
  for (let c = node.firstChild; c; c = c.nextSibling) {
    const assign = c.nextSibling;
    if (c.name === "VariableName" && assign?.name === "AssignOp") {
      const name = state.sliceDoc(c.from, c.to);
      const value = assign.nextSibling;
      if (name === "method" && value?.name === "String")
        method = state
          .sliceDoc(value.from, value.to)
          .replace(/^['"]|['"]$/g, "");
      if (c.to < pos) used.add(name);
      if (assign.to <= pos) keyword = name;
      named = true;
    }
    if (c.from < pos && c.name === ",") {
      if (!named) positional++;
      index++;
      named = false;
      keyword = null;
    }
  }
  return {
    call: node.parent,
    args: node,
    used,
    positional,
    keyword,
    index,
    method,
  };
}

function pickSignature(fn: QxFunction, method: string | null) {
  return fn.signatures.find((s) => s.method === method) ?? fn.signatures[0];
}

function activeParam(sig: QxSignature, at: ArgPosition): QxParam | undefined {
  if (at.keyword)
    return (
      sig.params.find((p) => p.name === at.keyword) ??
      sig.params.find((p) => p.name.startsWith("**"))
    );
  const name = sig.positional[at.index];
  return name ? sig.params.find((p) => p.name === name) : undefined;
}

function signatureTooltip(state: EditorState): Tooltip | null {
  const sel = state.selection.main;
  if (!loaded || !sel.empty) return null;
  const at = argPosition(state, sel.head);
  const fn = at && resolveCall(state, at.call, loaded);
  if (!at || !fn) return null;
  const sig = pickSignature(fn, at.method);
  const active = activeParam(sig, at);
  return {
    pos: at.args.from,
    above: true,
    create: () => {
      const dom = el("div", "qx-sig");
      const line = el("div", "qx-sig-line");
      line.append(el("span", "qx-sig-name", fn.label), "(");
      sig.params.forEach((p, i) => {
        if (i) line.append(", ");
        line.append(
          el(
            "span",
            `qx-sig-param${p === active ? " active" : ""}${p.required ? " required" : ""}`,
            paramText(p),
          ),
        );
      });
      line.append(")");
      dom.append(line);
      if (fn.signatures.length > 1)
        dom.append(
          el(
            "div",
            "qx-sig-methods",
            T("cells.methodOthers", {
              method: String(sig.method),
              others: fn.signatures
                .filter((s) => s !== sig)
                .map((s) => s.method)
                .join(", "),
            }),
          ),
        );
      if (active) {
        const help = el("div", "qx-sig-active");
        help.append(el("code", "", active.name));
        if (active.type) help.append(el("span", "qx-sig-type", active.type));
        if (active.doc) help.append(" — " + active.doc);
        dom.append(help);
      } else if (sig.doc)
        dom.append(
          el(
            "div",
            "qx-sig-doc",
            sig.doc.length > 240 ? sig.doc.slice(0, 240) + "…" : sig.doc,
          ),
        );
      return { dom };
    },
  };
}

/** Signature help: shown while the cursor is inside the arguments of a qx call. */
const signatureHelp = StateField.define<Tooltip | null>({
  create: signatureTooltip,
  update: (value, tr) =>
    tr.docChanged || tr.selection ? signatureTooltip(tr.state) : value,
  provide: (f) => showTooltip.from(f),
});

/** `param=` completions for the call the cursor is in: the ones not given yet, required first. */
function keywordOptions(
  ctx: CompletionContext,
  names: Completions,
): Completion[] {
  const at = argPosition(ctx.state, ctx.pos);
  const fn = at && !at.keyword ? resolveCall(ctx.state, at.call, names) : null;
  if (!at || !fn) return [];
  const sig = pickSignature(fn, at.method);
  const given = new Set([
    ...at.used,
    ...sig.positional.slice(0, at.positional),
  ]);
  return sig.params
    .filter((p) => !p.name.startsWith("**") && !given.has(p.name))
    .map((p, i) => ({
      label: `${p.name}=`,
      type: "property",
      detail: [
        p.type,
        p.required
          ? T("cells.required")
          : p.default !== null
            ? `= ${p.default}`
            : null,
      ]
        .filter(Boolean)
        .join("  "),
      info: p.doc || undefined,
      boost: (p.required ? 99 : 60) - i,
    }));
}

const QX_HANDLES = [
  "page",
  "obj",
  "report",
  "tool",
  "frame",
  "register",
  "series_from",
];

async function qxComplete(ctx: CompletionContext) {
  const names = await loadCompletions();
  const node = syntaxTree(ctx.state).resolveInner(ctx.pos, -1);
  if (node.name === "String" || node.name === "FormatString") {
    // Inside a string: series of the pages (formulas), objects and pages.
    const word = ctx.matchBefore(/\w*/);
    if (!word || (word.from === word.to && !ctx.explicit)) return null;
    const seen = new Set<string>();
    const options: Completion[] = [];
    for (const p of names.pages)
      for (const c of p.columns)
        if (!seen.has(c))
          (seen.add(c),
            options.push({ label: c, type: "variable", detail: p.name }));
    for (const o of names.objects)
      options.push({ label: o.name, type: "class", detail: o.type });
    for (const p of names.pages)
      options.push({ label: p.name, type: "namespace", detail: T("cells.detailPage") });
    return { from: word.from, options, validFor: /^\w*$/ };
  }
  const member = ctx.matchBefore(/(qx\.report|qx|\)|\w+)\.\w*/);
  if (member) {
    const dot = member.text.lastIndexOf(".");
    const target = member.text.slice(0, dot);
    const from = member.from + dot + 1;
    const fn = (labels: string[], on: string, detail: string): Completion[] =>
      labels.map((l) => {
        const f = names.signatures?.[`${on}.${l}`];
        return f
          ? {
              label: l,
              type: "function",
              detail: shortSignature(f.signatures[0]),
              info: () => functionInfo(f),
            }
          : { label: l, type: "function", detail };
      });
    if (target === "qx.report")
      return {
        from,
        options: fn(names.functions.report, "report", "qx.report"),
      };
    if (target === "qx")
      return {
        from,
        options: [
          ...fn(names.functions.qx, "qx", "qx"),
          ...QX_HANDLES.map((h) => ({ label: h, type: "namespace" })),
        ],
      };
    if (target === "np" || target === "pd") return null;
    // qx.page(…). / qx.obj(…). / eq01.  → methods of a page or an object
    return {
      from,
      options: [
        ...fn(names.functions.obj, "obj", T("cells.detailObject")),
        ...fn(names.functions.page, "page", T("cells.detailPage")),
      ],
    };
  }
  // Inside the arguments of a qx call its keywords come first, offered as soon as "(" or "," is typed.
  const word = ctx.matchBefore(/\w+/);
  const keywords = keywordOptions(ctx, names);
  const start = word ? word.from : ctx.pos;
  const afterSeparator = /[(,]\s*$/.test(
    ctx.state.sliceDoc(Math.max(0, start - 40), start),
  );
  if (!word && !ctx.explicit && !(keywords.length && afterSeparator))
    return null;
  return {
    from: start,
    options: [
      ...keywords,
      { label: "qx", type: "namespace" },
      ...names.objects.map((o) => ({
        label: o.name,
        type: "variable",
        detail: o.type,
      })),
    ],
    validFor: /^\w*$/,
  };
}

// ---- the formula argument stands out ----------------------------------------------------
const formulaMarks = new MatchDecorator({
  regexp: /\bformula\s*=\s*("[^"\n]*"?|'[^'\n]*'?)/g,
  decoration: Decoration.mark({ class: "cm-formula" }),
});
const highlightFormula = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor(view: EditorView) {
      this.decorations = formulaMarks.createDeco(view);
    }
    update(u: ViewUpdate) {
      this.decorations = formulaMarks.updateDeco(u, this.decorations);
    }
  },
  { decorations: (v) => v.decorations },
);

const theme = EditorView.theme({
  "&": { fontSize: "12px", background: "#f9fafc" },
  ".cm-content": { fontFamily: "var(--mono)", padding: "6px 0" },
  ".cm-gutters": { background: "#f1f4f9", border: "none", color: "#a3acbd" },
  "&.cm-focused": { outline: "none" },
  ".cm-formula": {
    background: "rgba(47, 98, 214, 0.10)",
    borderRadius: "3px",
    boxShadow: "0 0 0 1px rgba(47,98,214,0.25)",
  },
});

/** Where an issue points: the string of its argument (formula="…") or else the line of the call. */
function locate(code: string, line: number, param: string): [number, number] {
  const lines = code.split("\n");
  const re = new RegExp(`\\b${param}\\s*=\\s*("[^"\\n]*"|'[^'\\n]*')`);
  const m = re.exec(code);
  if (m) return [m.index + m[0].length - m[1].length, m.index + m[0].length];
  const start = lines
    .slice(0, Math.max(0, line - 1))
    .reduce((n, l) => n + l.length + 1, 0);
  return [start, start + (lines[line - 1]?.length ?? 0)];
}

export function CodeEditor(props: {
  value: string;
  editable: boolean;
  onChange(code: string): void;
  onRun(): void;
  diagnostics: Diagnostic[];
  /** Take the focus when the editor becomes editable (default); off for cells that are always editable. */
  autoFocus?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const editable = useRef(new Compartment());
  const handlers = useRef(props);
  handlers.current = props;

  useEffect(() => {
    void loadCompletions(); // the signature tooltip reads them synchronously
    const v = new EditorView({
      parent: host.current!,
      state: EditorState.create({
        doc: props.value,
        extensions: [
          basicSetup,
          python(),
          autocompletion({ override: [qxComplete] }),
          signatureHelp,
          highlightFormula,
          theme,
          EditorView.lineWrapping,
          keymap.of([
            { key: "Mod-Enter", run: () => (handlers.current.onRun(), true) },
          ]),
          editable.current.of([
            EditorView.editable.of(props.editable),
            EditorState.readOnly.of(!props.editable),
          ]),
          EditorView.updateListener.of(
            (u) =>
              u.docChanged && handlers.current.onChange(u.state.doc.toString()),
          ),
        ],
      }),
    });
    view.current = v;
    return () => v.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const effects = [
      editable.current.reconfigure([
        EditorView.editable.of(props.editable),
        EditorState.readOnly.of(!props.editable),
      ]),
    ];
    const changes =
      v.state.doc.toString() !== props.value
        ? { from: 0, to: v.state.doc.length, insert: props.value }
        : undefined;
    v.dispatch({ effects, changes });
    if (props.editable && props.autoFocus !== false) v.focus();
  }, [props.editable, props.value]);

  useEffect(() => {
    const v = view.current;
    if (v)
      v.dispatch(
        setDiagnostics(
          v.state,
          props.diagnostics.filter((d) => d.to <= v.state.doc.length),
        ),
      );
  }, [props.diagnostics]);

  return <div className="cell-editor" ref={host} />;
}

// ---- the cell -----------------------------------------------------------------------------
function fallbackCode(cell: CellInfo) {
  return `qx.tool("${cell.name}", **${JSON.stringify(cell.args, null, 1).replace(/\n\s*/g, " ")})`;
}

function useDebounced<T>(value: T, ms: number) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function CodeCell(props: {
  cell: CellInfo;
  block: Block | undefined;
  onOpenObject(name: string, blockId?: string | null): void;
}) {
  const { cell, block } = props;
  const original = cell.code ?? fallbackCode(cell);
  const [code, setCode] = useState(original);
  const [editing, setEditing] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const [check, setCheck] = useState<CodeCheck | null>(null);
  const [running, setRunning] = useState(false);
  const [run, setRun] = useState<CodeRun | { error: string } | null>(null);
  const [lastSeq, setLastSeq] = useState<number | null | undefined>(cell.seq);
  const [version, setVersion] = useState<number | null | undefined>(
    cell.version,
  );
  const [copied, setCopied] = useState(false);
  const debounced = useDebounced(editing ? code : null, 350);

  useEffect(() => setCode(cell.code ?? fallbackCode(cell)), [cell.code]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setLastSeq(cell.seq), [cell.seq]);
  useEffect(() => setVersion(cell.version), [cell.version]);

  // Live checks while editing: does it bind, unknown variables, operators the page cannot take.
  useEffect(() => {
    if (debounced === null) {
      setCheck(null);
      return;
    }
    let alive = true;
    api
      .checkCode(debounced)
      .then((c) => alive && setCheck(c))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [debounced]);

  const diagnostics = useMemo<Diagnostic[]>(() => {
    if (!check || !editing) return [];
    if (check.kind === "invalid") {
      const m = /line (\d+)/.exec(check.error);
      const [from, to] = locate(code, m ? Number(m[1]) : 1, "__none__");
      return [{ from, to, severity: "error", message: check.error }];
    }
    if (check.kind !== "api") return [];
    return check.calls.flatMap((c) =>
      c.issues.map((i) => {
        const [from, to] = locate(code, c.line, i.param);
        return {
          from,
          to,
          severity: i.level,
          message: i.message,
        } as Diagnostic;
      }),
    );
  }, [check, code, editing]);

  const errors = diagnostics.filter((d) => d.severity === "error").length;
  const status =
    running || cell.ok === undefined ? "running" : cell.ok ? "ok" : "fail";
  const changed = code !== original;
  const object = cell.object ?? block?.ref?.object ?? block?.result_id ?? null;
  const edited =
    run !== null && "kind" in run && run.ok && lastSeq !== cell.seq;

  async function execute() {
    if (running || (editing && errors > 0)) return;
    setRunning(true);
    setRun(null);
    try {
      const out = await api.runCode(
        code,
        lastSeq ?? null,
        cell.cell_id ?? null,
      );
      setRun(out);
      const seq =
        out.kind === "api"
          ? out.calls.find((c) => c.seq != null)?.seq
          : out.seq;
      if (out.ok) {
        setEditing(false);
        if (seq != null) setLastSeq(seq);
        const bumped = out.versions.find((v) => v.name === object);
        if (bumped) setVersion(bumped.version);
      }
    } catch (err) {
      setRun({ error: (err as Error).message });
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className={`cell ${status} ${editing ? "editing" : ""}`}>
      <div className="cell-head">
        {status === "running" ? (
          <LoaderCircle size={13} className="spin" />
        ) : status === "ok" ? (
          <Code2 size={13} />
        ) : (
          <CircleAlert size={13} />
        )}
        <span className="cell-tool">{cell.name}</span>
        {object && (
          <button
            className="chip"
            title={T("cells.openObjectInPanel")}
            onClick={() => props.onOpenObject(object, cell.block_id)}
          >
            {object}
            {version ? <sup>v{version}</sup> : null}
          </button>
        )}
        {edited && <span className="pill ok">{T("cells.edited")}</span>}
        <span className="cell-actions">
          {!editing && (
            <button
              title={T("cells.edit")}
              onClick={() => setEditing(true)}
              disabled={running}
            >
              <Pencil size={13} />
            </button>
          )}
          <button
            title={editing ? T("cells.runShortcut") : T("cells.rerun")}
            onClick={execute}
            disabled={running || (editing && errors > 0)}
          >
            {editing ? <Play size={13} /> : <RotateCw size={13} />}
          </button>
          {editing && (
            <button
              title={T("cells.cancelEdit")}
              onClick={() => {
                setEditing(false);
                setCode(original);
              }}
            >
              <X size={13} />
            </button>
          )}
          {object && (
            <button
              title={T("cells.openObject")}
              onClick={() => props.onOpenObject(object, cell.block_id)}
            >
              <ExternalLink size={13} />
            </button>
          )}
          <button
            title={T("cells.copyCode")}
            onClick={() => {
              navigator.clipboard.writeText(code);
              setCopied(true);
              setTimeout(() => setCopied(false), 1200);
            }}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        </span>
      </div>
      <CodeEditor
        value={code}
        editable={editing}
        onChange={setCode}
        onRun={execute}
        diagnostics={diagnostics}
      />
      {editing && diagnostics.length > 0 && (
        <div className="cell-note fail">
          <CircleAlert size={13} />
          <span>{diagnostics.map((d) => d.message).join(" · ")}</span>
        </div>
      )}
      {editing && check?.kind === "python" && (
        <div className="cell-note warn">
          <TriangleAlert size={13} />
          <span>
            {T("cells.freePython", { reason: check.reason })}
          </span>
        </div>
      )}
      {editing && changed && check?.kind === "api" && errors === 0 && (
        <div className="cell-note">
          {T("cells.runHint")}
        </div>
      )}
      {run && <RunOutcome run={run} />}
      {cell.finding && (
        <div className="cell-finding">
          <Lightbulb size={13} />
          <div>
            <Markdown text={cell.finding} />
          </div>
        </div>
      )}
      {(block || cell.content) && (
        <>
          <button
            className="cell-toggle"
            onClick={() => setShowResult((s) => !s)}
          >
            <ChevronRight size={12} className={showResult ? "rot" : ""} />{" "}
            {T("cells.result")}
            {block?.stale && <span className="pill warn">{T("cells.stale")}</span>}
          </button>
          {showResult &&
            (block ? (
              <div className="cell-result">
                <BlockBody block={block} />
              </div>
            ) : (
              <pre className="cell-raw">{cell.content}</pre>
            ))}
        </>
      )}
    </div>
  );
}

function RunOutcome({ run }: { run: CodeRun | { error: string } }) {
  if (!("kind" in run))
    return <div className="cell-note fail">{run.error}</div>;
  let message: string;
  if (run.ok) {
    message = run.versions.length
      ? run.versions.map((v) => `${v.name} → v${v.version}`).join(", ") +
        T("cells.panelViewsUpdated")
      : T("cells.executed");
  } else if (run.kind === "api") {
    message =
      run.calls.find((c) => !c.ok && !c.skipped)?.error ?? T("cells.cellFailed");
  } else {
    message =
      typeof run.error === "string"
        ? run.error
        : run.error?.ename
          ? `${run.error.ename}: ${run.error.evalue ?? ""}`
          : T("cells.cellFailed");
  }
  return (
    <div className={`cell-note ${run.ok ? "ok" : "fail"}`}>
      {run.ok ? <Check size={13} /> : <CircleAlert size={13} />}
      <span>
        {message}
        {run.stale.length > 0 && (
          <> · {T("cells.staleList", { names: run.stale.join(", ") })}</>
        )}
      </span>
      {run.kind === "python" && run.text ? (
        <pre className="cell-raw">{run.text}</pre>
      ) : null}
    </div>
  );
}

/** All the code of a turn as one copyable block. */
export function TurnCode({ cells }: { cells: CellInfo[] }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const code = cells.map((c) => c.code ?? fallbackCode(c)).join("\n\n");
  return (
    <div className="turn-code">
      <button className="cell-toggle" onClick={() => setOpen((o) => !o)}>
        <ChevronRight size={12} className={open ? "rot" : ""} />{" "}
        {T(cells.length === 1 ? "cells.turnCode.one" : "cells.turnCode.other", {
          count: cells.length,
        })}
      </button>
      {open && (
        <div className="code">
          <button
            className="copy"
            onClick={() => {
              navigator.clipboard.writeText("import qtrix as qx\n\n" + code);
              setCopied(true);
              setTimeout(() => setCopied(false), 1200);
            }}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
          <pre>{code}</pre>
        </div>
      )}
    </div>
  );
}
