// Conversations of the active notebook: each one has its own chat and panel, all of them work on the notebook's data.
import { MessageSquare, MoreHorizontal, Pencil, Plus, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import type { Thread } from "./api";
import { EditableTitle, type MenuAt, RowMenu, ago, menuAt } from "./notebooks";

const meta = (t: Thread) => ago(t.updated_at);

export function ThreadList(props: {
  threads: Thread[];
  activeId: string | null;
  /** A chat turn is running: switching would hide it. */
  locked: boolean;
  onSelect(id: string): void;
  onCreate(): void;
  onRename(id: string, title: string): void;
  onDelete(id: string): void;
}) {
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

  // The newest first: the conversation one is working in is usually the latest.
  const shown = [...props.threads].sort((a, b) => b.created_at - a.created_at);
  const last = props.threads.length === 1;

  return (
    <div className="notebooks threads">
      <div className="side-title">
        <span>
          Conversaciones <span className="count">{props.threads.length}</span>
        </span>
        <button
          className="ghost"
          title={props.locked ? "Espera a que termine el turno en curso" : "Nueva conversación (mismos datos)"}
          aria-label="Nueva conversación"
          disabled={props.locked}
          onClick={props.onCreate}
        >
          <Plus size={15} />
        </button>
      </div>
      <ul className="nb-list">
        {shown.map((t) => {
          const active = t.id === props.activeId;
          if (confirming === t.id)
            return (
              <li key={t.id} className="nb-confirm">
                <span>¿Borrar «{t.title}»? Los datasets y objetos del cuaderno se conservan.</span>
                <span className="nb-confirm-actions">
                  <button
                    className="danger"
                    onClick={() => {
                      setConfirming(null);
                      props.onDelete(t.id);
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
            <li key={t.id} className={`nb ${active ? "active" : ""}`}>
              {renaming === t.id ? (
                <div className="nb-item editing">
                  <MessageSquare size={14} />
                  <EditableTitle
                    start
                    className="nb-title"
                    value={t.title}
                    onSave={(title) => props.onRename(t.id, title)}
                    onDone={() => setRenaming(null)}
                  />
                </div>
              ) : (
                <button
                  className="nb-item"
                  aria-current={active}
                  disabled={props.locked && !active}
                  title={props.locked && !active ? "Espera a que termine el turno en curso" : t.title}
                  onClick={() => props.onSelect(t.id)}
                >
                  <MessageSquare size={14} />
                  <span className="nb-text">
                    <span className="nb-title">{t.title}</span>
                    <span className="nb-meta">{meta(t)}</span>
                  </span>
                </button>
              )}
              <button
                className="ghost nb-more"
                aria-label={`Acciones de ${t.title}`}
                aria-haspopup="menu"
                onClick={(e) => {
                  e.stopPropagation();
                  setAt(menuAt(e));
                  setMenu(menu === t.id ? null : t.id);
                }}
              >
                <MoreHorizontal size={15} />
              </button>
              {menu === t.id && (
                <RowMenu at={at}>
                  <button
                    role="menuitem"
                    onClick={() => {
                      setRenaming(t.id);
                      setMenu(null);
                    }}
                  >
                    <Pencil size={13} /> Renombrar
                  </button>
                  <button
                    role="menuitem"
                    className="danger"
                    disabled={last}
                    title={last ? "Un cuaderno necesita al menos una conversación" : undefined}
                    onClick={() => {
                      setConfirming(t.id);
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
      </ul>
    </div>
  );
}
