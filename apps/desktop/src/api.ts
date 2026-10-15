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
  agent: {
    instructions_extra: string;
    namespaces: string[] | null;
    max_tool_calls: number;
    planner?: boolean;
    /** Let the agent run Python cells without asking (they run with the user's privileges). */
    allow_code?: boolean;
    /** Auto-compaction: fold the conversation at ``compact_at`` of the context, capped at ``max_context`` tokens. */
    auto_compact?: boolean;
    compact_at?: number;
    max_context?: number;
  };
  llamacpp?: { variant: "cpu" | "vulkan" | "cuda"; ctx_size: number };
  /** The agent team the lead may delegate parts of a request to. */
  team?: TeamConfig;
  api_keys?: Record<string, boolean>;
  active_is_local?: boolean;
  forced_local_only?: boolean;
}

/** A member of the agent team: a specialist econometrician with its own behaviour, skills and model. */
export interface TeamAgent {
  /** Stable id (letters, digits, "-" and "_"): the member signs its calls in the journal as ``agent:<id>``. */
  id: string;
  name: string;
  emoji: string;
  behavior: string;
  /** Skills (by name) the member may load. */
  skills: string[];
  namespaces?: string[] | null;
  /** Model profile (by name); null = the active one. */
  profile: string | null;
  enabled: boolean;
}

export interface TeamConfig {
  enabled: boolean;
  agents: TeamAgent[];
}

/** Members of a team at most, the default one included. */
export const MAX_TEAM = 10;
/** The default member: the lead econometrician, always present and first. */
export const LEAD_ID = "lead";

/** A method guide an agent can load. */
export interface SkillInfo {
  name: string;
  description: string;
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
  /** Objects a Python cell registered through its own calls. */
  objects?: { name: string; version?: number | null }[] | null;
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
  section: "idea" | "conclusion" | "issue" | "result";
  text: string;
}

/** The agent's workspace in one turn: the goal of the plan and the notes it writes as it works. */
export interface Scratch {
  goal: string;
  notes: ScratchNote[];
}

/** What is left to do in the conversation, shared by the user and the agent. ``objects`` are the ``[[name]]``
 * references in the text. */
export interface NextStep {
  id: string;
  text: string;
  done: boolean;
  author: "agent" | "user";
  objects: string[];
}

/** A team member's work in a turn: tagged with ``agent``, its events build its own workspace, not the lead's. */
export type ChatEvent = (
  | { type: "turn"; turn_id: string; after_block?: string | null }
  /** The lead hands a task to a member of its team; its events follow, tagged with its id. */
  | { type: "agent_spawn"; agent: string; name: string; emoji: string; task: string }
  /** The member is done; ``content`` is its conclusions. */
  | { type: "agent_done"; agent: string; ok: boolean; content: string }
  | { type: "next_steps"; items: NextStep[] }
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
  /** The context meter after the turn. */
  | ({ type: "context" } & ContextUsage)
  /** The conversation filled the context and was folded into a summary (``before`` = tokens it filled). */
  | ({ type: "compacted"; before: number } & ContextUsage)
  /** Reporting mode: the ReportingAgent is rewriting the conversation's report after the turn. */
  | { type: "reporting"; status: "running" | "done" | "error"; version?: number; message?: string }
  | { type: "done" }
) & { agent?: string };

/** The conversation's report, written by the ReportingAgent (paper-style: sections that cite objects by name). */
export interface ReportObjectRef {
  name: string;
  view?: string | null;
  caption: string;
}
export interface ReportSection {
  heading: string;
  body_md: string;
  objects: ReportObjectRef[];
}
export interface ConversationReport {
  title: string;
  abstract: string;
  sections: ReportSection[];
  conclusions_md: string;
  limitations_md: string;
  next_steps: string[];
}
export interface ReportingState {
  enabled: boolean;
  version: number;
  report: ConversationReport | null;
}

/** How much of the model's context window the conversation fills. */
export interface ContextUsage {
  used: number;
  limit: number;
  /** No usage reported by the provider: counted from the text. */
  estimated: boolean;
  /** Share of the window at which the conversation is compacted (only from ``GET /chat/context``). */
  compact_at?: number;
  auto_compact?: boolean;
}

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
  /** The conversation was compacted when this question arrived. */
  compacted?: boolean;
  /** What each team member did in the turn, in order. */
  agents?: AgentRecord[];
}

/** A team member's work in a saved turn: its task, conclusions (``content``) and workspace. */
export interface AgentRecord {
  id: string;
  name: string;
  emoji?: string;
  task: string;
  ok?: boolean;
  content?: string;
  cells?: CellInfo[];
  plan?: PlanStep[];
  scratch?: Scratch;
  intro?: string;
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
const GLOBAL_ROUTE = /^\/(health|settings|skills|models|llm|localai|system|notebooks)(?=[/?]|$)/;
// A notebook holds several conversations on the same data: the chat and the panel of one are reached as
// /notebooks/<id>/threads/<tid>/<route> (the data routes ignore the conversation).
const THREADS_ROUTE = /^\/threads(?=[/?]|$)/;
let activeNotebook: string | null = null;
let activeThread: string | null = null;

/** Which notebook the notebook-scoped routes act on (its open conversation until one is set). */
export function setActiveNotebook(id: string | null) {
  activeNotebook = id;
  activeThread = null;
}

/** Which conversation of the active notebook the chat and panel routes act on. */
export function setActiveThread(id: string | null) {
  activeThread = id;
}

function route(path: string) {
  if (!activeNotebook || GLOBAL_ROUTE.test(path)) return path;
  const thread = activeThread && !THREADS_ROUTE.test(path) ? `/threads/${activeThread}` : "";
  return `/notebooks/${activeNotebook}${thread}${path}`;
}

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

/** One analysis method of the catalog (GET /methods): what the user can ask the agent for, by area. */
export interface AnalysisMethod {
  id: string;
  label: string;
  summary: string;
  tools: string[];
  variants: string[];
  /** Functions of the causal library the agent runs in a Python cell. */
  library: string[];
  /** An example request; «…» marks what the user fills in. */
  prompt: string;
  /** Data structures the method needs; null = any. */
  structures: PageKind[] | null;
  available: boolean;
  /** Unknown tools or Python modules that are not installed. */
  missing: string[];
}

export interface MethodArea {
  id: string;
  label: string;
  summary: string;
  methods: AnalysisMethod[];
}

/** A registry tool as the help popup shows it (GET /tools/<name>/help): what it does and the parameters it takes. */
export interface ToolHelp {
  name: string;
  summary: string;
  qx: string | null;
  available: boolean;
  requires: string[];
  kinds: PageKind[] | null;
  params: QxParam[];
}

/** One column of a dataset as profiled by GET /datasets/<id>/profile (numeric stats only for numeric columns). */
export interface ColumnProfile {
  name: string;
  dtype: string;
  missing: number;
  unique: number;
  mean?: number | null;
  sd?: number | null;
  min?: number | null;
  max?: number | null;
}

export const api = {
  toolHelp: (name: string) => request<ToolHelp>(`/tools/${encodeURIComponent(name)}/help`),
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
  skills: () => request<SkillInfo[]>("/skills"),
  setApiKey: (provider: string, key: string | null) => request("/settings/api-key", json({ provider, key })),
  localModels: () => request<Record<string, string[]>>("/models/local"),
  llmUsage: () => request<LlmUsage>("/llm/usage"),
  testModel: (p: ModelProfile) => request<{ ok: boolean; reply?: string; error?: string }>("/models/test", json(p)),
  history: () => request<ChatTurn[]>("/chat/history"),
  resetChat: () => request("/chat/reset", { method: "POST" }),
  nextSteps: () => request<NextStep[]>("/chat/next-steps"),
  addNextStep: (text: string) => request<NextStep[]>("/chat/next-steps", json({ text })),
  editNextStep: (id: string, patch: { text?: string; done?: boolean }) =>
    request<NextStep[]>(`/chat/next-steps/${id}`, { method: "PATCH", body: JSON.stringify(patch) }),
  removeNextStep: (id: string) => request<NextStep[]>(`/chat/next-steps/${id}`, { method: "DELETE" }),
  reporting: () => request<ReportingState>("/reporting"),
  objectView: (name: string, view?: string | null) =>
    request<{ kind: BlockKind; title: string; data: Record<string, any> }>(
      view ? `/objects/${name}/views/${view}` : `/objects/${name}/view`,
    ),
  chatContext: () => request<ContextUsage>("/chat/context"),
  compactChat: () => request<ContextUsage>("/chat/compact", { method: "POST" }),
  /** Plan mode: the detailed plan of a request, as Markdown to review before anything runs. */
  draftPlan: (message: string, signal?: AbortSignal) =>
    request<{ plan: string }>("/chat/plan", { ...json({ message }), signal }).then((r) => r.plan),
  tool: <T = any>(name: string, args: Record<string, unknown>) => request<T>(`/tools/${name}`, json(args)),
  methods: () => request<{ areas: MethodArea[] }>("/methods").then((r) => r.areas),
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

/** A conversation of the active notebook: its own chat and panel, on the notebook's datasets and objects. */
export interface Thread {
  id: string;
  title: string;
  /** Epoch seconds. */
  created_at: number;
  updated_at: number;
  n_turns: number;
  n_blocks: number;
  /** The conversation the engine has open in this notebook. */
  current: boolean;
  /** Reporting mode: the ReportingAgent keeps the conversation's report up to date. */
  reporting?: boolean;
  report_version?: number;
}

export const threads = {
  list: () => request<Thread[]>("/threads"),
  create: (title?: string) => request<Thread>("/threads", json({ title: title ?? null })),
  open: (id: string) => request<Thread>(`/threads/${id}/open`, { method: "POST" }),
  rename: (id: string, title: string) => request<Thread>(`/threads/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  remove: (id: string) => request(`/threads/${id}`, { method: "DELETE" }),
  setReporting: (id: string, reporting: boolean) =>
    request<Thread>(`/threads/${id}`, { method: "PATCH", body: JSON.stringify({ reporting }) }),
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

/** One turn of the conversation; ``plan`` is a plan the user approved in plan mode, which the turn follows. */
export async function* chat(message: string, signal?: AbortSignal, plan?: string): AsyncGenerator<ChatEvent> {
  const { baseUrl, token } = await engine();
  const res = await fetch(baseUrl + route("/chat"), {
    method: "POST",
    headers: { "content-type": "application/json", "x-qtrix-token": token },
    body: JSON.stringify({ message, plan }),
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    yield { type: "error", message: detail.detail ?? res.statusText };
    return;
  }
  yield* sse<ChatEvent>(res.body);
}

/** Write the conversation's report again from every turn (streams ``reporting`` events). */
export async function* rebuildReporting(signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const { baseUrl, token } = await engine();
  const res = await fetch(baseUrl + route("/reporting/rebuild"), {
    method: "POST",
    headers: { "content-type": "application/json", "x-qtrix-token": token },
    signal,
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    yield { type: "reporting", status: "error", message: detail.detail ?? res.statusText };
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
