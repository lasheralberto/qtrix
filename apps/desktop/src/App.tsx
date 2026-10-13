// Workbench layout: notebooks and their data in the rail · one analysis panel where the conversation and its results
// live together (E7·08c). Everything in the panel belongs to the active notebook (E8·04).
import { Brain, Cloud, Eraser, HardDrive, PanelRight, Redo2, Settings, Sparkles, Undo2 } from "lucide-react";
import { Fragment, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  api,
  notebooks as notebookApi,
  panelEvents,
  setActiveNotebook,
  type Block,
  type Dataset,
  type Notebook,
  type PanelEvent,
  type UserSettings,
} from "./api";
import { BlockCard } from "./blocks";
import { Composer, SUGGESTIONS, TurnAnswer, TurnHead, useChat, type ComposerHandle, type Turn } from "./chat";
import { DatasetList, DatasetModal } from "./data";
import { ModelPicker } from "./modelpicker";
import { ModelsModal } from "./models";
import { DraftCell, NotebookRibbon, type Draft } from "./notebook";
import { EditableTitle, NotebookList, UNTITLED } from "./notebooks";
import { SettingsModal } from "./settings";
import { Workspace } from "./workspace";

function applyPanelEvent(blocks: Block[], ev: PanelEvent): Block[] {
  switch (ev.op) {
    case "snapshot":
      return ev.blocks;
    case "add":
      return [...blocks.filter((b) => b.id !== ev.block.id), ev.block];
    case "update":
    case "pin":
      return blocks.map((b) => (b.id === ev.block.id ? ev.block : b));
    case "remove":
      return blocks.filter((b) => b.id !== ev.block_id);
    case "reorder":
      return ev.order.map((id) => blocks.find((b) => b.id === id)).filter((b): b is Block => Boolean(b));
  }
}

/** Where each turn starts in the block list: right after its anchor block, never before the previous turn. */
function turnStarts(blocks: Block[], turns: Turn[]): number[] {
  const index = new Map(blocks.map((b, i) => [b.id, i]));
  let prev = 0;
  return turns.map((t) => {
    const at = t.after == null ? 0 : index.has(t.after) ? index.get(t.after)! + 1 : prev;
    prev = Math.max(at, prev);
    return prev;
  });
}

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [blocks, setBlocks] = useState<Block[]>([]);
  const [version, setVersion] = useState(0);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  const [openDataset, setOpenDataset] = useState<Dataset | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showModels, setShowModels] = useState(false);
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [memory, setMemory] = useState("…");
  const [toast, setToast] = useState<string | null>(null);
  // The agent's workspace: open while it works (the user may close it), on the latest turn unless one was picked.
  const [wsOpen, setWsOpen] = useState(false);
  const [wsTurn, setWsTurn] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const closeWorkspace = useCallback(() => setWsOpen(false), []);
  const composer = useRef<ComposerHandle>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true); // follow new content only while the user is at the bottom of the panel
  const blocksRef = useRef<Block[]>([]);
  blocksRef.current = blocks;
  const lastBlock = useCallback(() => blocksRef.current[blocksRef.current.length - 1]?.id ?? null, []);
  const { turns, busy, send, stop, reset } = useChat(lastBlock, activeId);
  const activeRef = useRef<string | null>(null);
  activeRef.current = activeId;
  const active = notebooks.find((n) => n.id === activeId) ?? null;

  const refreshNotebooks = useCallback(() => notebookApi.list().then(setNotebooks).catch(() => {}), []);
  const refreshDatasets = useCallback(() => {
    const id = activeId;
    if (!id) return Promise.resolve();
    return api
      .datasets()
      .then((d) => {
        if (activeRef.current === id) setDatasets(d); // a slow answer of the notebook we just left is dropped
        refreshNotebooks();
      })
      .catch(() => {});
  }, [activeId, refreshNotebooks]);

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      notebookApi
        .list()
        .then((list) => {
          if (cancelled) return;
          const first = list.find((n) => n.current) ?? list[0];
          setNotebooks(list);
          setActiveNotebook(first.id);
          setActiveId(first.id);
        })
        .catch(() => !cancelled && setTimeout(load, 1000)); // the engine may still be starting
    load();
    api.settings().then(setSettings).catch(() => {});
    const poll = setInterval(() => api.health().then((h) => setMemory(h.memory)).catch(() => setMemory("offline")), 5000);
    api.health().then((h) => setMemory(h.memory)).catch(() => {});
    return () => {
      cancelled = true;
      clearInterval(poll);
    };
  }, []);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Leaving a notebook empties the view before the next one loads, so nothing of the old one lingers.
  const switchTo = (id: string) => {
    setActiveNotebook(id);
    setBlocks([]);
    setVersion(0);
    setDatasets([]);
    setOpenDataset(null);
    setFresh(new Set());
    setWsTurn(null);
    setDraft(null);
    stick.current = true;
    setActiveId(id);
  };

  // Live panel: the engine pushes every operation the agents perform in the active notebook.
  useEffect(() => {
    if (!activeId) return;
    const ctrl = new AbortController();
    (async () => {
      while (!ctrl.signal.aborted) {
        try {
          for await (const ev of panelEvents(ctrl.signal)) {
            setVersion(ev.version);
            setBlocks((prev) => applyPanelEvent(prev, ev));
            if (ev.op === "add" || ev.op === "update") {
              const id = ev.block.id;
              setFresh((s) => new Set(s).add(id));
              setTimeout(() => setFresh((s) => (s.delete(id), new Set(s))), 2400);
            }
          }
        } catch {
          /* engine restarting or aborted */
        }
        await new Promise((r) => setTimeout(r, 1000));
      }
    })();
    return () => ctrl.abort();
  }, [activeId]);

  // Datasets may be created by agents (transformations, SQL), so refresh after each turn.
  useEffect(() => {
    if (!busy) {
      refreshDatasets();
      // The engine may have swapped to an installed local model during the turn.
      api.settings().then(setSettings).catch(() => {});
    }
  }, [busy, refreshDatasets]);

  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [blocks, turns]);

  const notify = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  };

  const fail = (e: unknown) => notify(e instanceof Error ? e.message : String(e));

  const renameNotebook = (id: string, title: string) => notebookApi.rename(id, title).then(refreshNotebooks).catch(fail);

  // A notebook nobody named takes its name from the first thing that goes into it: a dataset or the first question.
  const nameFrom = (text: string) => {
    const title = text.replace(/\s+/g, " ").trim().slice(0, 48);
    if (activeId && active?.title === UNTITLED && title) renameNotebook(activeId, title);
  };

  useEffect(() => {
    if (datasets.length > 0 && turns.length === 0) nameFrom(datasets[0].name);
  }, [datasets, active?.title]);

  const selectNotebook = async (id: string) => {
    if (busy || id === activeId) return;
    try {
      await notebookApi.open(id);
      switchTo(id);
      refreshNotebooks();
    } catch (e) {
      fail(e);
    }
  };

  const createNotebook = async () => {
    if (busy) return;
    try {
      const nb = await notebookApi.create();
      switchTo(nb.id);
      refreshNotebooks();
    } catch (e) {
      fail(e);
    }
  };

  const duplicateNotebook = (id: string) =>
    notebookApi
      .duplicate(id)
      .then((nb) => (refreshNotebooks(), notify(`Duplicado como «${nb.title}»`)))
      .catch(fail);

  const deleteNotebook = async (id: string) => {
    if (busy && id === activeId) return notify("Espera a que termine el turno en curso");
    try {
      await notebookApi.remove(id);
      const list = await notebookApi.list();
      setNotebooks(list);
      if (id === activeId) switchTo((list.find((n) => n.current) ?? list[0]).id); // the engine moved on to the next one
    } catch (e) {
      fail(e);
    }
  };

  const openWorkspace = (key: string | null) => {
    setWsTurn(key === turns[turns.length - 1]?.key ? null : key);
    setWsOpen(true);
  };

  const ask = (text: string) => {
    stick.current = true;
    setWsTurn(null);
    setWsOpen(true);
    if (turns.length === 0) nameFrom(text);
    return send(text);
  };

  const flash = (id: string) => {
    document.getElementById(`block-${id}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    setFresh((s) => new Set(s).add(id));
    setTimeout(() => setFresh((s) => (s.delete(id), new Set(s))), 2400);
  };

  // "Abrir objeto" from a chat cell: its block in the panel, else a live view of it is added.
  const openObject = (name: string, blockId?: string | null) => {
    const block =
      blocks.find((b) => b.id === blockId) ??
      blocks.find((b) => b.ref?.object === name) ??
      blocks.find((b) => b.result_id === name);
    if (block) return flash(block.id);
    api
      .tool<{ block_id?: string }>("obj_view", { name })
      .then((r) => r.block_id && setTimeout(() => flash(r.block_id!), 300))
      .catch((e) => notify(e.message));
  };

  const card = (b: Block, i: number) => (
    <BlockCard
      key={b.id}
      block={b}
      index={i + 1}
      fresh={fresh.has(b.id)}
      onPin={(p) => api.pin(b.id, p).catch((e) => notify(e.message))}
      onRemove={() => api.removeBlock(b.id).catch((e) => notify(e.message))}
      onAsk={ask}
      onAskAbout={() => composer.current?.focus(`Sobre el bloque ${i + 1} («${b.title}»): `)}
      onFreeze={() => api.freezeBlock(b.id).catch((e) => notify(e.message))}
    />
  );

  // Blocks no turn produced come first; then each turn: its question, the blocks it produced, the agent's answer,
  // and what the user wrote after it (a note or a cell added once the turn ended goes below the answer, not inside it).
  function renderFeed() {
    const starts = turnStarts(blocks, turns);
    const slice = (from: number, to: number) => blocks.slice(from, to).map((b, j) => card(b, from + j));
    return (
      <>
        {slice(0, starts[0] ?? blocks.length)}
        {turns.map((t, k) => {
          const from = starts[k];
          const to = starts[k + 1] ?? blocks.length;
          const mine = t.running ? -1 : blocks.slice(from, to).findIndex((b) => b.source === "user");
          const split = mine < 0 ? to : from + mine;
          return (
            <Fragment key={t.key}>
              <TurnHead turn={t} blocks={blocks} onOpenObject={openObject} onOpenWorkspace={() => openWorkspace(t.key)} />
              {slice(from, split)}
              <TurnAnswer turn={t} />
              {slice(split, to)}
            </Fragment>
          );
        })}
        {draft && <DraftCell key={draft} kind={draft} onClose={() => setDraft(null)} />}
      </>
    );
  }

  const model = settings?.profiles.find((p) => p.name === settings.active_profile) ?? settings?.profiles[0];
  const isLocal = settings?.active_is_local ?? true;

  return (
    <div className={`app ${wsOpen ? "ws-open" : ""}`}>
      <aside className="rail">
        <div className="brand">
          <span className="logo">q</span>
          <div>
            <strong>qtrix</strong>
            <span className="muted small">econometría agéntica local</span>
          </div>
        </div>
        <NotebookList
          notebooks={notebooks}
          activeId={activeId}
          locked={busy}
          onSelect={selectNotebook}
          onCreate={createNotebook}
          onRename={renameNotebook}
          onDuplicate={duplicateNotebook}
          onDelete={deleteNotebook}
        />
        <DatasetList datasets={datasets} onChange={refreshDatasets} onOpen={setOpenDataset} onError={notify} />
        <div className="rail-foot">
          <div className={`badge ${isLocal ? "local" : "cloud"}`} title="Destino de los prompts">
            {isLocal ? <HardDrive size={13} /> : <Cloud size={13} />}
            {isLocal ? "Modelo local" : `Prompts → ${model?.provider}`}
          </div>
          <div className="badge" title="Memoria semántica (turbovec)">
            <Brain size={13} /> memoria: {memory}
          </div>
          <button className="ghost wide" onClick={() => setShowSettings(true)}>
            <Settings size={15} /> Ajustes
          </button>
        </div>
      </aside>

      <main className="panel">
        <header className="panel-head">
          <div>
            {active ? <EditableTitle value={active.title} onSave={(t) => renameNotebook(active.id, t)} /> : <h1>qtrix</h1>}
            <span className="muted small">
              {blocks.length} bloques · v{version}
              {busy && (
                <span className="working">
                  <Sparkles size={12} /> el equipo está trabajando…
                </span>
              )}
            </span>
          </div>
          <div className="panel-tools">
            <button
              className={`ghost ws-toggle ${wsOpen ? "on" : ""} ${busy ? "live" : ""}`}
              title={wsOpen ? "Ocultar el espacio de trabajo" : "Espacio de trabajo del agente"}
              aria-pressed={wsOpen}
              onClick={() => (wsOpen ? setWsOpen(false) : openWorkspace(wsTurn))}
            >
              <PanelRight size={16} />
              {busy && <span className="ws-live-dot" />}
            </button>
          </div>
        </header>
        <NotebookRibbon draft={draft} onAdd={setDraft} onAskAI={() => composer.current?.focus()}>
          <button className="ghost" title="Deshacer" onClick={() => api.undo()}>
            <Undo2 size={15} />
          </button>
          <button className="ghost" title="Rehacer" onClick={() => api.redo()}>
            <Redo2 size={15} />
          </button>
          <button className="ghost" title="Limpiar (conserva fijados)" onClick={() => api.clear()}>
            <Eraser size={15} />
          </button>
        </NotebookRibbon>
        <div
          className="blocks"
          ref={scroller}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 240;
          }}
        >
          {blocks.length === 0 && turns.length === 0 && (
            <div className="panel-empty">
              <span className="empty-mark">
                <Sparkles size={22} />
              </span>
              <h2>El cuaderno se construye mientras conversas</h2>
              <p>
                Importa un dataset a la izquierda y pregunta abajo en lenguaje natural. El agente elegirá estimadores, ejecutará los
                modelos en tu máquina y sus tablas, contrastes y gráficos aparecerán aquí. Cada acción del agente es una celda de Python
                que puedes editar y re-ejecutar.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} className="suggestion" onClick={() => ask(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {renderFeed()}
        </div>
        <Composer
          ref={composer}
          busy={busy}
          onSend={ask}
          onStop={stop}
          onReset={reset}
          modelPicker={
            <ModelPicker
              settings={settings}
              onChange={setSettings}
              onOpenModels={() => setShowModels(true)}
              onOpenSettings={() => setShowSettings(true)}
            />
          }
        />
      </main>

      <Workspace
        open={wsOpen}
        turns={turns}
        turnKey={wsTurn}
        onSelect={setWsTurn}
        onClose={closeWorkspace}
        onOpenObject={openObject}
      />

      {openDataset && (
        <DatasetModal
          dataset={openDataset}
          onClose={() => setOpenDataset(null)}
          onChange={() => refreshDatasets().then(() => setOpenDataset(null))}
        />
      )}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} onSaved={setSettings} />}
      {showModels && <ModelsModal onClose={() => setShowModels(false)} />}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
