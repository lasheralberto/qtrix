// English texts of the app, one file per group of modules under ./en. Keys are "<module>.<name>".
import { app } from "./en/app";
import { cells } from "./en/cells";
import { chat } from "./en/chat";
import { data } from "./en/data";
import { models } from "./en/models";
import { notebooks } from "./en/notebooks";
import { plan } from "./en/plan";
import { team } from "./en/team";

export const i18n_en = { ...app, ...chat, ...cells, ...data, ...models, ...notebooks, ...plan, ...team } as const;
