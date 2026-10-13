// Typed client for the local qtrix engine.

export type BlockKind = "regression_table" | "plotly_figure" | "dataframe" | "diagnostics" | "markdown" | "equation" | "python_cell";

export interface Block {
  id: string;
  kind: BlockKind;
  title: string;
  data: Record<string, any>;
  /** "user": a note or Python cell the user wrote in the panel (E7·08d); agents never modify it. */
  source: string;
  pinned: boolean;
  result_id: string | null;
  code: string | null;
  /** A live view: the block follows this object view and refreshes when the object changes (E7·08). */
  ref?: { object: string; view: string; params: Record<string, unknown> } | null;
  /** Frozen or stale: the object changed since this block was computed. */
  stale?: boolean;
  created_at: string;
  updated_at: string;
}

export interface PanelSnapshot {
  version: number;
  blocks: Block[];
}

export type PanelEvent =
  | ({ op: "snapshot" } & PanelSnapshot)
  | { op: "add" | "update" | "pin"; version: number; block: Block }
  | { op: "remove"; version: number; block_id: string }
  | { op: "reorder"; version: number; order: string[] };

export type PageKind = "cross_section" | "time_series" | "panel" | "panel_undated";
export type Freq = "U" | "A" | "S" | "Q" | "M" | "W" | "D5" | "D7";

export interface StructureSpec {
  kind: PageKind;
  freq?: Freq | null;
  date_col?: string | null;
  id_col?: string | null;
  t_col?: string | null;
  start?: string | null;
}

export interface StructureSuggestion {
  kind: PageKind;
  freq: Freq;
  date_col: string;
  id_col: string | null;
  start: string;
  end: string;
  reason: string;
}

export interface SampleInfo {
  sample: string;
  /** e.g. "1990q1 2008q4 (1990Q1–2008Q4; 76 of 100 rows)" */
  description: string;
  n_selected: number;
  n_total: number;
  first: string | null;
  last: string | null;
}

export interface Dataset {
  id: string;
  name: string;
  /** "user": a note or Python cell the user wrote in the panel (E7·08d); agents never modify it. */
  source: string;
  n_rows: number;
  columns: { name: string; type: string }[];
  /** What the data are (ADR 0002): cross-section unless declared otherwise. */
  kind: PageKind;
  freq: Freq;
  start: string | null;
  end: string | null;
  id_col: string | null;
  date_col: string | null;
  t_col: string | null;
  gaps: number;
  suggestion: StructureSuggestion | null;
  /** Active sample in text form (null = every row). */
  sample: string | null;
  /** One-line description, e.g. "time series quarterly 1990Q1–2014Q4". */
  structure: string;
  entity_col: string | null;
  time_col: string | null;
}

export interface ModelProfile {
  name: string;
  provider: "llamacpp" | "ollama" | "lmstudio" | "openai_compatible" | "anthropic" | "openai" | "google" | "mistral";
  model: string;
  base_url?: string | null;
  temperature?: number;
}

export interface UserSettings {
  profiles: ModelProfile[];
  active_profile: string;
  /** Profiles tried in order when the active model fails (all calls go through the LiteLLM gateway). */
  fallbacks?: string[];
  local_only: boolean;
  agent: { instructions_extra: string; namespaces: string[] | null; max_tool_calls: number; planner?: boolean };
  llamacpp?: { variant: "cpu" | "vulkan" | "cuda"; ctx_size: number };
  api_keys?: Record<string, boolean>;
  active_is_local?: boolean;
  forced_local_only?: boolean;
}

export interface LlmUsage {
  models: Record<string, { calls: number; prompt_tokens: number; completion_tokens: number; cost_usd: number; unpriced_calls: number; fallbacks: number }>;
  total_cost_usd: number;
}

/** One agent action as a cell (E7·08b): its canonical qx code, journal entry and what it produced. */
export interface CellInfo {
  id: string;
  name: string;
  args: Record<string, unknown>;
  code: string | null;
  ok?: boolean;
  content?: string;
  seq?: number | null;
  cell_id?: string | null;
  object?: string | null;
  version?: number | null;
  block_id?: string | null;
  /** What the agent read in this cell's result, in its own words. */
  finding?: string;
}

/** A step of the plan the planner drafted for a turn; the executor settles it as it works. */
export interface PlanStep {
  text: string;
  status: "pending" | "done" | "skipped";
  note: string;
}

/** A note the agent wrote in its workspace: an alternative it weighed, a provisional conclusion or a deliverable. */
export interface ScratchNote {
  section: "idea" | "conclusion" | "result";
  text: string;
}

/** The agent's workspace in one turn: the goal of the plan and the notes it writes as it works. */
export interface Scratch {
  goal: string;
  notes: ScratchNote[];
}

export type ChatEvent =
  | { type: "turn"; turn_id: string; after_block?: string | null }
  | { type: "plan"; steps: PlanStep[] }
  | ({ type: "scratch" } & Scratch)
  /** Text the agent wrote after some results: it explains those cells (no ids = the turn's intro). The last
   * ``strip`` characters streamed as answer text were that explanation and leave the answer. */
  | { type: "finding"; ids: string[]; text: string; strip: number }
  | ({ type: "tool_call" } & CellInfo)
  | ({ type: "tool_result" } & CellInfo & { ok: boolean; content: string })
  | { type: "text"; delta: string }
  | { type: "status"; message: string }
  | { type: "error"; message: string }
  | { type: "done" };

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  tools?: string[];
  cells?: CellInfo[];
  plan?: PlanStep[];
  scratch?: Scratch;
  /** What the agent said before its first action. */
  intro?: string;
  turn_id?: string;
  /** The panel block that was last when the turn started (E7·08c): where the turn sits in the panel. */
  after_block?: string | null;
}

export interface CodeIssue {
  param: string;
  level: "error" | "warning";
  message: string;
}

export type CodeCheck =
  | { kind: "api"; calls: { line: number; tool: string; args: Record<string, unknown>; code: string | null; issues: CodeIssue[] }[] }
  | { kind: "python"; reason: string }
  | { kind: "invalid"; error: string };

export interface CodeEffects {
  versions: { name: string; version: number }[];
  stale: string[];
}

/** A rich output of a kernel cell: a stream, or a result/display with its MIME bundle. */
export type KernelOutput =
  | { type: "stream"; name: string; text: string }
  | { type: "result" | "display"; data: Record<string, any> };

export type CodeRun =
  | ({ kind: "api"; ok: boolean; calls: { line: number; tool: string; code?: string; seq?: number; ok: boolean; error?: string; skipped?: boolean; result?: any }[] } & CodeEffects)
  | ({
      kind: "python";
      ok: boolean;
      reason: string;
      seq?: number;
      text?: string;
      outputs?: KernelOutput[];
      execution_count?: number | null;
      duration_ms?: number;
      error?: { ename?: string; evalue?: string; traceback?: string[] } | string | null;
    } & CodeEffects);

/** What a Python cell written in the panel holds: its last run (``invalid`` when it could not even start). */
export type PanelCellRun = CodeRun | { kind: "invalid"; ok: false; error: string };

export interface QxParam {
  name: string;
  type: string | null;
  required: boolean;
  default: string | null;
  doc: string;
}

/** One way to call a qx function: a family (qx.equation, qx.var) has one per ``method=``. */
export interface QxSignature {
  method: string | null;
  doc: string;
  positional: string[];
  params: QxParam[];
}

export interface QxFunction {
  on: string;
  name: string;
  label: string;
  signatures: QxSignature[];
}

export interface Completions {
  functions: Record<"qx" | "page" | "obj" | "report", string[]>;
  /** "<target>.<function>" → its signatures, for the editor's signature help. */
  signatures?: Record<string, QxFunction>;
  objects: { name: string; type: string }[];
  pages: { name: string; columns: string[] }[];
}

let enginePromise: Promise<{ baseUrl: string; token: string }> | null = null;

export function engine() {
  if (!enginePromise) {
    enginePromise = window.qtrix
      ? window.qtrix.engineInfo()
      : Promise.resolve({ baseUrl: import.meta.env.VITE_ENGINE_URL ?? "http://127.0.0.1:8765", token: "" });
  }
  return enginePromise;
}

// Notebooks (E8·04): everything a notebook holds (datasets, panel, chat, code, objects, journal) is reached as
// /notebooks/<id>/<route>; what belongs to the app (health, settings, models, notebooks themselves) is not prefixed.
const GLOBAL_ROUTE = /^\/(health|settings|models|llm|localai|system|notebooks)(?=[/?]|$)/;
let activeNotebook: string | null = null;

/** Which notebook the notebook-scoped routes act on. */
export function setActiveNotebook(id: string | null) {
  activeNotebook = id;
}

const route = (path: string) => (activeNotebook && !GLOBAL_ROUTE.test(path) ? `/notebooks/${activeNotebook}${path}` : path);

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const { baseUrl, token } = await engine();
  const res = await fetch(baseUrl + route(path), {
    ...init,
    headers: { "content-type": "application/json", "x-qtrix-token": token, ...(init.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* keep status text */
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<{ status: string; version: string; memory: string }>("/health"),
  datasets: () => request<Dataset[]>("/datasets"),
  importPath: (path: string) => request<Dataset>("/datasets/import", json({ path })),
  preview: (id: string, limit = 200) =>
    request<{ columns: string[]; rows: unknown[][]; total_rows: number }>(`/datasets/${id}/preview?limit=${limit}`),
  profile: (id: string) => request<Record<string, any>[]>(`/datasets/${id}/profile`),
  sample: (id: string) => request<SampleInfo>(`/datasets/${id}/sample`),
  setSample: (id: string, sample: string | null) =>
    request<SampleInfo>(`/datasets/${id}/sample`, { method: "PUT", body: JSON.stringify({ sample }) }),
  setStructure: (id: string, spec: StructureSpec) =>
    request<Dataset>(`/datasets/${id}/structure`, { method: "PUT", body: JSON.stringify(spec) }),
  deleteDataset: (id: string) => request(`/datasets/${id}`, { method: "DELETE" }),
  panel: () => request<PanelSnapshot>("/panel"),
  pin: (id: string, pinned: boolean) => request<Block>(`/panel/blocks/${id}/pin`, json({ pinned })),
  removeBlock: (id: string) => request(`/panel/blocks/${id}`, { method: "DELETE" }),
  reorder: (order: string[]) => request<PanelSnapshot>("/panel/reorder", json({ order })),
  undo: () => request<PanelSnapshot>("/panel/undo", { method: "POST" }),
  redo: () => request<PanelSnapshot>("/panel/redo", { method: "POST" }),
  clear: () => request<PanelSnapshot>("/panel/clear", { method: "POST" }),
  settings: () => request<UserSettings>("/settings"),
  saveSettings: (s: UserSettings) => request<UserSettings>("/settings", { method: "PUT", body: JSON.stringify(s) }),
  setApiKey: (provider: string, key: string | null) => request("/settings/api-key", json({ provider, key })),
  localModels: () => request<Record<string, string[]>>("/models/local"),
  llmUsage: () => request<LlmUsage>("/llm/usage"),
  testModel: (p: ModelProfile) => request<{ ok: boolean; reply?: string; error?: string }>("/models/test", json(p)),
  history: () => request<ChatTurn[]>("/chat/history"),
  resetChat: () => request("/chat/reset", { method: "POST" }),
  tool: <T = any>(name: string, args: Record<string, unknown>) => request<T>(`/tools/${name}`, json(args)),
  freezeBlock: (id: string) => request<Block>(`/panel/blocks/${id}/freeze`, { method: "POST" }),
  // Cells (E7·08b): check as the user types, run an edited cell as a new version.
  checkCode: (code: string) => request<CodeCheck>("/code", json({ code, dry_run: true })),
  runCode: (code: string, editedFrom?: number | null, cellId?: string | null) =>
    request<CodeRun>("/code", json({ code, edited_from: editedFrom ?? null, cell_id: cellId ?? null })),
  completions: () => request<Completions>("/code/completions"),
  // The user's own blocks (E7·08d): notes and Python cells written in the panel.
  addNote: (title: string, markdown: string) => request<Block>("/panel/notes", json({ title, markdown })),
  editNote: (id: string, title: string, markdown: string) =>
    request<Block>(`/panel/blocks/${id}/note`, { method: "PUT", body: JSON.stringify({ title, markdown }) }),
  addCell: (code: string) => request<Block>("/panel/cells", json({ code })),
  rerunCell: (id: string, code: string) =>
    request<Block>(`/panel/blocks/${id}/cell`, { method: "PUT", body: JSON.stringify({ code }) }),
};

export interface Notebook {
  id: string;
  title: string;
  description: string;
  /** Epoch seconds. */
  created_at: number;
  updated_at: number;
  opened_at: number;
  /** The notebook the engine opened last: where the app returns to on start. */
  current: boolean;
  n_datasets: number;
  n_objects: number;
}

export const notebooks = {
  list: () => request<Notebook[]>("/notebooks"),
  create: (title?: string) => request<Notebook>("/notebooks", json({ title: title ?? null })),
  open: (id: string) => request<Notebook>(`/notebooks/${id}/open`, { method: "POST" }),
  rename: (id: string, title: string) => request<Notebook>(`/notebooks/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  duplicate: (id: string) => request<Notebook>(`/notebooks/${id}/duplicate`, json({})),
  remove: (id: string) => request(`/notebooks/${id}`, { method: "DELETE" }),
};

// ---- local AI: Hugging Face models + embedded llama.cpp ---------------------------------
export interface Job {
  id: string;
  kind: "model" | "runtime";
  title: string;
  status: "queued" | "running" | "verifying" | "done" | "error" | "cancelled";
  total: number;
  done: number;
  progress: number | null;
  message: string;
  error: string | null;
}

export interface LocalAIStatus {
  runtime: { installed: { tag: string; variant: string; server: string } | null; running: boolean; model: string | null };
  variant: string;
  variants: string[];
  ctx_size: number;
  platform: string;
  models_dir: string;
  hf_token: boolean;
  jobs: Job[];
}

export interface Recommended {
  repo: string;
  file: string;
  title: string;
  size_gb: number;
  ram_gb: number;
  note: string;
  installed: boolean;
}

export interface LocalModel {
  id: string;
  name: string;
  repo: string | null;
  quant: string | null;
  size: number;
  path: string;
  active: boolean;
  serving: boolean;
}

export interface RepoFiles {
  repo: string;
  gated: boolean;
  license: string | null;
  files: { name: string; files: string[]; size: number; quant: string | null; kind: string }[];
}

export interface SystemInfo {
  home: string;
  origin: string;
  pointer: string;
  folders: { name: string; description: string; path: string; size: number }[];
}

const q = encodeURIComponent;

export const localai = {
  status: () => request<LocalAIStatus>("/localai/status"),
  recommended: () => request<Recommended[]>("/localai/recommended"),
  search: (query: string) => request<{ repo: string; downloads: number; likes: number }[]>(`/localai/search?q=${q(query)}`),
  repo: (repo: string) => request<RepoFiles>(`/localai/repo?repo=${q(repo)}`),
  download: (repo: string, file: string) => request<Job>("/localai/download", json({ repo, file })),
  jobs: () => request<Job[]>("/localai/jobs"),
  cancel: (id: string) => request<Job>(`/localai/jobs/${id}`, { method: "DELETE" }),
  models: () => request<LocalModel[]>("/localai/models"),
  remove: (id: string) => request(`/localai/models?id=${q(id)}`, { method: "DELETE" }),
  use: (id: string) => request<UserSettings>("/localai/use", json({ id })),
  installRuntime: (variant: string) => request<Job>("/localai/runtime/install", json({ variant })),
  stopRuntime: () => request("/localai/runtime/stop", { method: "POST" }),
  setContext: (ctx_size: number) => request<LocalAIStatus>("/localai/runtime/config", json({ ctx_size })),
  setHfToken: (token: string | null) => request("/settings/api-key", json({ provider: "huggingface", key: token })),
  systemInfo: () => request<SystemInfo>("/system/info"),
};

export const formatBytes = (n: number) => {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1);
  return `${(n / 1024 ** i).toFixed(i >= 3 ? 2 : 0)} ${units[i]}`;
};

/** Parse a text/event-stream body into JSON payloads. */
async function* sse<T>(body: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const data = chunk
        .split("\n")
        .filter((l) => l.startsWith("data: "))
        .map((l) => l.slice(6))
        .join("\n");
      if (data) yield JSON.parse(data) as T;
    }
  }
}

export async function* chat(message: string, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const { baseUrl, token } = await engine();
  const res = await fetch(baseUrl + route("/chat"), {
    method: "POST",
    headers: { "content-type": "application/json", "x-qtrix-token": token },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    yield { type: "error", message: detail.detail ?? res.statusText };
    return;
  }
  yield* sse<ChatEvent>(res.body);
}

export async function* panelEvents(signal: AbortSignal): AsyncGenerator<PanelEvent> {
  const { baseUrl, token } = await engine();
  const res = await fetch(baseUrl + route("/panel/stream"), { headers: { "x-qtrix-token": token }, signal });
  if (!res.body) return;
  yield* sse<PanelEvent>(res.body);
}
