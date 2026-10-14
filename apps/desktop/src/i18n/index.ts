// Translation lookup. Only English for now: to add a language, create i18n_<lang>.ts with the same keys and select it here.
import { i18n_en } from "./i18n_en";

export type TextKey = keyof typeof i18n_en;

const dict: Record<TextKey, string> = i18n_en;

/** Text for `key`; `{name}` placeholders are replaced with `vars.name`. */
export function T(key: TextKey, vars?: Record<string, string | number>): string {
  const text = dict[key];
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m));
}
