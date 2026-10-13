// Autocomplete for the composer: as the user types a word, it suggests the analysis methods of the catalog (GET /methods)
// and the variables of the loaded datasets that match it. Suggestions only fill the message in: the agent is still the
// one that runs anything.
import { useEffect, useMemo, useState } from "react";
import type { Dataset, MethodArea } from "./api";

export interface Suggestion {
  text: string;
  kind: "variable" | "method" | "tool";
  /** Where it comes from: the dataset (and type) of a variable, the area or method of a method. */
  detail: string;
}

const KIND_LABEL: Record<Suggestion["kind"], string> = {
  variable: "variable",
  method: "método",
  tool: "qx",
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
    const vars = new Map<string, { type: string; from: string[] }>();
    for (const d of datasets)
      for (const c of d.columns) {
        const v = vars.get(c.name) ?? { type: c.type, from: [] };
        v.from.push(d.name);
        vars.set(c.name, v);
      }
    for (const [name, v] of vars)
      add({ text: name, kind: "variable", detail: `${v.from.join(", ")} · ${v.type}` });
    for (const a of areas ?? [])
      for (const m of a.methods) {
        add({ text: m.label, kind: "method", detail: a.label });
        for (const variant of m.variants) add({ text: variant, kind: "method", detail: m.label });
        for (const t of m.tools) add({ text: t, kind: "tool", detail: m.label });
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

/** The list that opens above the composer. Mouse down accepts (so the textarea keeps its focus). */
export function SuggestionList({
  id,
  items,
  word,
  active,
  onHover,
  onPick,
}: {
  id: string;
  items: Suggestion[];
  word: string;
  active: number;
  onHover(i: number): void;
  onPick(s: Suggestion): void;
}) {
  useEffect(() => {
    document.getElementById(`${id}-${active}`)?.scrollIntoView?.({ block: "nearest" });
  }, [id, active]);
  return (
    <ul className="suggest" id={id} role="listbox" aria-label="Sugerencias">
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
  );
}

/** Which entry is highlighted: reset to the first whenever the list changes. */
export function useActiveIndex(items: Suggestion[]) {
  const [active, setActive] = useState(0);
  const key = items.map((s) => s.text).join("\u0000");
  useEffect(() => setActive(0), [key]);
  return [active, setActive] as const;
}
