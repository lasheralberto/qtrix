// Notebooks (E8·04): the rail lists them like chat sessions; opening one shows everything that lives in it.
import { BookOpen, Copy, MoreHorizontal, Pencil, Plus, Search, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Notebook } from "./api";

/** Where a row menu opens: under its "..." button. */
export type MenuAt = { top: number; right: number };
export const menuAt = (e: { currentTarget: HTMLElement }): MenuAt => {
  const r = e.currentTarget.getBoundingClientRect();
  return { top: r.bottom + 2, right: window.innerWidth - r.right };
};

/** Rendered in body: the sidebar lists scroll (and the rail is transformed), which would clip or misplace it. */
export function RowMenu(props: { at: MenuAt; children: React.ReactNode }) {
  return createPortal(
    <div className="nb-menu" role="menu" style={{ top: props.at.top, right: props.at.right }}>
      {props.children}
    </div>,
    document.body,
  );
}

export const UNTITLED = "Cuaderno sin título";

/** "hace 5 min", "ayer", or the date, from an epoch in seconds. */
export function ago(epoch: number, now = Date.now()): string {
  const min = Math.max(0, Math.round((now - epoch * 1000) / 60000));
  if (min < 1) return "ahora";
  if (min < 60) return `hace ${min} min`;
  const h = Math.round(min / 60);
  if (h < 24) return `hace ${h} h`;
  const d = Math.round(h / 24);
  if (d === 1) return "ayer";
  if (d < 7) return `hace ${d} días`;
  return new Date(epoch * 1000).toLocaleDateString("es-ES", { day: "numeric", month: "short" });
}

const counts = (nb: Notebook) =>
  [nb.n_datasets ? `${nb.n_datasets} ${nb.n_datasets === 1 ? "dataset" : "datasets"}` : "vacío"]
    .concat(nb.n_objects ? [`${nb.n_objects} obj.`] : [])
    .join(" · ");

/** A title that turns into an input on click; Enter or leaving saves it, Escape cancels. */
export function EditableTitle(props: {
  value: string;
  onSave(title: string): void;
  className?: string;
  start?: boolean;
  onDone?(): void;
}) {
  const [editing, setEditing] = useState(Boolean(props.start));
  const [text, setText] = useState(props.value);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => setText(props.value), [props.value]);
  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  const finish = (save: boolean) => {
    setEditing(false);
    const title = text.trim();
    if (save && title && title !== props.value) props.onSave(title);
    else setText(props.value);
    props.onDone?.();
  };

  if (!editing)
    return (
      <h1 className={`${props.className ?? ""} editable`} title="Clic para renombrar" onClick={() => setEditing(true)}>
        {props.value}
        <Pencil size={13} />
      </h1>
    );
  return (
    <input
      ref={input}
      className="title-input"
      value={text}
      maxLength={80}
      aria-label="Título del cuaderno"
      onChange={(e) => setText(e.target.value)}
      onBlur={() => finish(true)}
      onKeyDown={(e) => {
        if (e.key === "Enter") finish(true);
        if (e.key === "Escape") finish(false);
      }}
    />
  );
}

export function NotebookList(props: {
  notebooks: Notebook[];
  activeId: string | null;
  /** A chat turn is running: switching would cut it off. */
  locked: boolean;
  onSelect(id: string): void;
  onCreate(): void;
  onRename(id: string, title: string): void;
  onDuplicate(id: string): void;
  onDelete(id: string): void;
}) {
  const [query, setQuery] = useState("");
  const [menu, setMenu] = useState<string | null>(null);
  const [at, setAt] = useState<MenuAt>({ top: 0, right: 0 });
  const [renaming, setRenaming] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(null);
    window.addEventListener("click", close);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("click", close);
      window.removeEventListener("scroll", close, true);
    };
  }, [menu]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? props.notebooks.filter((n) => n.title.toLowerCase().includes(q)) : props.notebooks;
  }, [props.notebooks, query]);

  return (
    <div className="notebooks">
      <div className="side-title">
        <span>
          Cuadernos <span className="count">{props.notebooks.length}</span>
        </span>
        <button className="ghost" title="Nuevo cuaderno" aria-label="Nuevo cuaderno" onClick={props.onCreate}>
          <Plus size={15} />
        </button>
      </div>
      {props.notebooks.length > 5 && (
        <label className="nb-search">
          <Search size={13} />
          <input value={query} placeholder="Buscar cuaderno" onChange={(e) => setQuery(e.target.value)} />
        </label>
      )}
      <ul className="nb-list">
        {shown.map((nb) => {
          const active = nb.id === props.activeId;
          if (confirming === nb.id)
            return (
              <li key={nb.id} className="nb-confirm">
                <span>¿Borrar «{nb.title}»? Va a la papelera de qtrix.</span>
                <span className="nb-confirm-actions">
                  <button
                    className="danger"
                    onClick={() => {
                      setConfirming(null);
                      props.onDelete(nb.id);
                    }}
                  >
                    <Trash2 size={13} /> Borrar
                  </button>
                  <button className="ghost" onClick={() => setConfirming(null)}>
                    <X size={13} /> Cancelar
                  </button>
                </span>
              </li>
            );
          return (
            <li key={nb.id} className={`nb ${active ? "active" : ""}`}>
              {renaming === nb.id ? (
                <div className="nb-item editing">
                  <BookOpen size={14} />
                  <EditableTitle
                    start
                    className="nb-title"
                    value={nb.title}
                    onSave={(t) => props.onRename(nb.id, t)}
                    onDone={() => setRenaming(null)}
                  />
                </div>
              ) : (
                <button
                  className="nb-item"
                  aria-current={active}
                  disabled={props.locked && !active}
                  title={props.locked && !active ? "Espera a que termine el turno en curso" : nb.title}
                  onClick={() => props.onSelect(nb.id)}
                >
                  <BookOpen size={14} />
                  <span className="nb-text">
                    <span className="nb-title">{nb.title}</span>
                    <span className="nb-meta">
                      {counts(nb)} · {ago(nb.opened_at)}
                    </span>
                  </span>
                </button>
              )}
              <button
                className="ghost nb-more"
                aria-label={`Acciones de ${nb.title}`}
                aria-haspopup="menu"
                onClick={(e) => {
                  e.stopPropagation();
                  setAt(menuAt(e));
                  setMenu(menu === nb.id ? null : nb.id);
                }}
              >
                <MoreHorizontal size={15} />
              </button>
              {menu === nb.id && (
                <RowMenu at={at}>
                  <button
                    role="menuitem"
                    onClick={() => {
                      setRenaming(nb.id);
                      setMenu(null);
                    }}
                  >
                    <Pencil size={13} /> Renombrar
                  </button>
                  <button role="menuitem" onClick={() => props.onDuplicate(nb.id)}>
                    <Copy size={13} /> Duplicar
                  </button>
                  <button
                    role="menuitem"
                    className="danger"
                    onClick={() => {
                      setConfirming(nb.id);
                      setMenu(null);
                    }}
                  >
                    <Trash2 size={13} /> Borrar
                  </button>
                </RowMenu>
              )}
            </li>
          );
        })}
        {shown.length === 0 && <li className="muted small nb-none">Ningún cuaderno coincide.</li>}
      </ul>
    </div>
  );
}
