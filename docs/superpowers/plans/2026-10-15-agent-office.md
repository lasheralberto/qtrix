# Agent Office Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A full-screen low-poly 3D office where the agent team is seen working; the user walks a chosen agent around with the arrow keys and reads workspaces at desks and results at the whiteboard.

**Architecture:** A pure tile grid (`grid.ts`) and a module-level store (`state.ts`) hold positions and movement; React Three Fiber renders the room and avatars from that store plus the chat state the app already has; an HTML overlay carries the bottom bar and the readable side panel, reusing the workspace components. No backend changes.

**Tech Stack:** React 19, TypeScript, Vite 8, `three@0.186`, `@react-three/fiber@9`, `@react-three/drei@10`, `vitest@5` (new, tests of the pure modules only).

**Spec:** `docs/superpowers/specs/2026-10-15-agent-office-design.md`

## Global Constraints

- All UI strings go through `src/i18n` (`T("office.…")`); keys live in `apps/desktop/src/i18n/en/office.ts` and are merged in `src/i18n/i18n_en.ts`.
- Keep diffs small; match the surrounding style (2-space, double quotes, trailing commas, `//` header comment per file explaining its role).
- `cd apps/desktop && npm run typecheck && npm run build` must stay green after every task; `tsconfig` has `noUnusedLocals`/`noUnusedParameters`.
- No external 3D assets (models, images, fonts): everything procedural or canvas-drawn.
- Room grid is 12 × 9 tiles; at most `MAX_TEAM` (10) desks.
- Working directory for all commands: `apps/desktop` unless stated.
- Do not commit unless the user asks; the commit steps below are what to run *when* they do.

## Review Focus

1. Team disabled or `settings.team` undefined → the room must still show the lead at the first desk (Task 1 test `layoutDesks` with one id; Task 4 falls back to `[LEAD]`).
2. A member's chair tile occupied by another agent when a turn starts → `walkHome` returns false and nothing crashes; the agent stays where it is (Task 2 test).
3. A path tile becomes blocked mid-walk (cat or another agent steps in) → `tick` re-plans instead of walking through (Task 2 test).
4. Agent name/emoji empty → face falls back to 🧑‍💼 and the label to "—" (Task 5 code; manual check in Task 8).
5. Report with empty `abstract` or lead turn with empty `text` → whiteboard reads the i18n "(nothing on the board yet)" and no writing walk is triggered (Task 7: `boardLines` returns the fallback, the effect skips the walk).

---

### Task 1: Grid: tiles, desk layout and paths

**Files:**
- Create: `apps/desktop/src/office/grid.ts`
- Create: `apps/desktop/src/office/grid.test.ts`
- Create: `apps/desktop/vitest.config.ts`
- Modify: `apps/desktop/package.json` (devDependency `vitest`, script `test`)

**Interfaces:**
- Produces: `COLS`, `ROWS`, `Tile`, `Dir`, `Desk`, `key`, `same`, `opposite`, `inRoom`, `world`, `WHITEBOARD_TILES`, `READING_ZONE`, `COFFEE_TILES`, `COFFEE_SPOT`, `BOARD_SPOT`, `layoutDesks(ids)`, `blockedTiles(desks)`, `walkable(t, blocked)`, `step(t, dir)`, `dirBetween(a, b)`, `findPath(from, to, blocked)`.

- [ ] **Step 1: Install vitest and add the test script**

Run: `npm i -D vitest@5`
Create `apps/desktop/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";

// Unit tests of the pure modules only (grid, office state); the UI is checked by typecheck, build and by hand.
export default defineConfig({ test: { include: ["src/**/*.test.ts"] } });
```

In `package.json` `"scripts"` add `"test": "vitest run"`.

- [ ] **Step 2: Write the failing tests**

`apps/desktop/src/office/grid.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  BOARD_SPOT,
  blockedTiles,
  COLS,
  findPath,
  inRoom,
  key,
  layoutDesks,
  ROWS,
  step,
  walkable,
  WHITEBOARD_TILES,
  type Tile,
} from "./grid";

const ids = (n: number) => Array.from({ length: n }, (_, i) => `ag-${i}`);

describe("layoutDesks", () => {
  it("gives the lead the first desk, by the whiteboard wall", () => {
    const [lead] = layoutDesks(["lead"]);
    expect(lead.id).toBe("lead");
    expect(lead.chair).toEqual({ x: 1, z: 2 });
  });

  it.each([1, 5, 10])("places %i desks inside the room without overlaps", (n) => {
    const desks = layoutDesks(ids(n));
    expect(desks).toHaveLength(n);
    const used = new Set<string>();
    for (const d of desks) {
      for (const t of [...d.tiles, d.chair]) {
        expect(inRoom(t)).toBe(true);
        expect(used.has(key(t))).toBe(false);
        used.add(key(t));
      }
    }
  });

  it("keeps every chair walkable and connected to the whiteboard", () => {
    const desks = layoutDesks(ids(10));
    const blocked = blockedTiles(desks);
    for (const d of desks) {
      expect(walkable(d.chair, blocked)).toBe(true);
      expect(findPath(d.chair, BOARD_SPOT, blocked)).not.toBeNull();
    }
  });

  it("caps the team at the desks the room has", () => {
    expect(layoutDesks(ids(12))).toHaveLength(10);
  });
});

describe("walkable", () => {
  it("stops at the walls", () => {
    const blocked = blockedTiles([]);
    expect(walkable({ x: -1, z: 0 }, blocked)).toBe(false);
    expect(walkable({ x: COLS, z: 0 }, blocked)).toBe(false);
    expect(walkable({ x: 0, z: ROWS }, blocked)).toBe(false);
    expect(walkable({ x: 3, z: 3 }, blocked)).toBe(true);
  });

  it("treats the whiteboard, the desks and anything in the set as furniture", () => {
    const desks = layoutDesks(["lead"]);
    const blocked = blockedTiles(desks);
    expect(walkable(WHITEBOARD_TILES[0], blocked)).toBe(false);
    expect(walkable(desks[0].tiles[0], blocked)).toBe(false);
    blocked.add(key({ x: 6, z: 6 }));
    expect(walkable({ x: 6, z: 6 }, blocked)).toBe(false);
  });
});

describe("findPath", () => {
  it("returns the shortest path, excluding the start and including the goal", () => {
    const path = findPath({ x: 2, z: 2 }, { x: 4, z: 2 }, new Set());
    expect(path).toEqual([
      { x: 3, z: 2 },
      { x: 4, z: 2 },
    ]);
  });

  it("goes around a desk", () => {
    const desks = layoutDesks(ids(10));
    const blocked = blockedTiles(desks);
    // The first south desk sits on (2,8)-(3,8): from its left to its right one must pass above it.
    const path = findPath({ x: 1, z: 8 }, { x: 4, z: 8 }, blocked)!;
    expect(path).not.toBeNull();
    expect(path.every((t) => !blocked.has(key(t)))).toBe(true);
    expect(path[path.length - 1]).toEqual({ x: 4, z: 8 });
  });

  it("is empty when already there and null when boxed in", () => {
    expect(findPath({ x: 5, z: 5 }, { x: 5, z: 5 }, new Set())).toEqual([]);
    const box = new Set<string>();
    for (const t of [step({ x: 5, z: 5 }, "n"), step({ x: 5, z: 5 }, "s"), step({ x: 5, z: 5 }, "e"), step({ x: 5, z: 5 }, "w")])
      box.add(key(t));
    expect(findPath({ x: 5, z: 5 }, { x: 8, z: 8 }, box)).toBeNull();
  });

  it("refuses a blocked goal", () => {
    const goal: Tile = { x: 7, z: 7 };
    expect(findPath({ x: 5, z: 5 }, goal, new Set([key(goal)]))).toBeNull();
  });
});
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `npm test`
Expected: FAIL — cannot resolve `./grid`.

- [ ] **Step 4: Implement `grid.ts`**

```ts
// The office floor as a grid of tiles: where the desks stand, what blocks the way, and how an agent walks from A
// to B. Pure functions; the store in state.ts keeps who is where.
export const COLS = 12;
export const ROWS = 9;

export type Tile = { x: number; z: number };
/** Compass directions; north is the whiteboard wall (z = 0), west the other wall (x = 0). */
export type Dir = "n" | "s" | "e" | "w";

/** A desk: the two tiles it stands on, the tile of its chair and the direction from the desk to the chair. */
export interface Desk {
  id: string;
  tiles: [Tile, Tile];
  chair: Tile;
  facing: Dir;
}

export const key = (t: Tile) => `${t.x},${t.z}`;
export const same = (a: Tile, b: Tile) => a.x === b.x && a.z === b.z;
export const opposite: Record<Dir, Dir> = { n: "s", s: "n", e: "w", w: "e" };
export const inRoom = (t: Tile) => t.x >= 0 && t.x < COLS && t.z >= 0 && t.z < ROWS;
/** The centre of a tile in scene units, the room centred on the origin. */
export const world = (t: Tile): [number, number] => [t.x - COLS / 2 + 0.5, t.z - ROWS / 2 + 0.5];

const span = (xs: number[], z: number): Tile[] => xs.map((x) => ({ x, z }));
/** The whiteboard hangs on the north wall over these tiles (a marker tray keeps them free). */
export const WHITEBOARD_TILES: Tile[] = span([4, 5, 6, 7], 0);
/** Standing here reads the whiteboard. */
export const READING_ZONE: Tile[] = span([4, 5, 6, 7], 1);
export const COFFEE_TILES: Tile[] = span([10, 11], 0);
/** Where an agent stands for a coffee, and where the lead stands to write. */
export const COFFEE_SPOT: Tile = { x: 10, z: 1 };
export const BOARD_SPOT: Tile = { x: 5, z: 1 };

// Desk slots in the order they are handed out: the west wall (nearest the whiteboard first), the south edge, then
// the east edge. The centre of the room stays free to walk.
const slot = (tiles: [Tile, Tile], chair: Tile, facing: Dir): Omit<Desk, "id"> => ({ tiles, chair, facing });
const DESK_SLOTS: Omit<Desk, "id">[] = [
  ...[2, 4, 6].map((z) => slot([{ x: 0, z }, { x: 0, z: z + 1 }], { x: 1, z }, "e")),
  ...[2, 5, 8, 10].map((x) => slot([{ x, z: 8 }, { x: x + 1, z: 8 }], { x, z: 7 }, "n")),
  ...[2, 4, 6].map((z) => slot([{ x: 11, z }, { x: 11, z: z + 1 }], { x: 10, z }, "w")),
];

/** One desk per id, in order, as many as the room has slots. */
export function layoutDesks(ids: string[]): Desk[] {
  return ids.slice(0, DESK_SLOTS.length).map((id, i) => ({ id, ...DESK_SLOTS[i] }));
}

/** The tiles furniture stands on: whiteboard tray, coffee machine and the desks. */
export function blockedTiles(desks: Desk[]): Set<string> {
  const s = new Set<string>();
  for (const t of [...WHITEBOARD_TILES, ...COFFEE_TILES]) s.add(key(t));
  for (const d of desks) for (const t of d.tiles) s.add(key(t));
  return s;
}

export const walkable = (t: Tile, blocked: Set<string>) => inRoom(t) && !blocked.has(key(t));

const DELTA: Record<Dir, [number, number]> = { n: [0, -1], s: [0, 1], e: [1, 0], w: [-1, 0] };
export const step = (t: Tile, d: Dir): Tile => ({ x: t.x + DELTA[d][0], z: t.z + DELTA[d][1] });

/** The direction from `a` to its neighbour `b`. */
export function dirBetween(a: Tile, b: Tile): Dir {
  if (b.x > a.x) return "e";
  if (b.x < a.x) return "w";
  if (b.z < a.z) return "n";
  return "s";
}

/** Shortest walk from `from` to `to` (breadth first), the goal included and the start left out; `[]` when already
 * there, `null` when there is no way. */
export function findPath(from: Tile, to: Tile, blocked: Set<string>): Tile[] | null {
  if (same(from, to)) return [];
  if (!walkable(to, blocked)) return null;
  const prev = new Map<string, Tile | null>([[key(from), null]]);
  const queue: Tile[] = [from];
  while (queue.length) {
    const t = queue.shift()!;
    for (const d of ["n", "s", "e", "w"] as Dir[]) {
      const n = step(t, d);
      if (!walkable(n, blocked) || prev.has(key(n))) continue;
      prev.set(key(n), t);
      if (same(n, to)) {
        const path: Tile[] = [];
        for (let c: Tile | null = n; c && !same(c, from); c = prev.get(key(c)) ?? null) path.unshift(c);
        return path;
      }
      queue.push(n);
    }
  }
  return null;
}
```

- [ ] **Step 5: Run the tests**

Run: `npm test`
Expected: all `grid.test.ts` tests PASS. Then `npm run typecheck` → clean.

- [ ] **Step 6: Commit (when asked)**

```bash
git add apps/desktop/package.json apps/desktop/package-lock.json apps/desktop/vitest.config.ts apps/desktop/src/office/grid.ts apps/desktop/src/office/grid.test.ts
git commit -m "feat(office): tile grid, desk layout and pathfinding"
```

---

### Task 2: Office store: positions, selection, walking

**Files:**
- Create: `apps/desktop/src/office/state.ts`
- Create: `apps/desktop/src/office/state.test.ts`

**Interfaces:**
- Consumes: everything from `grid.ts`.
- Produces: `AgentPos { tile; facing; seated; path; home }`, `OfficeState { positions; selected; cat }`, `office` object with `get`, `subscribe`, `setDesks(desks)`, `select(id)`, `move(id, dir): boolean`, `walkTo(id, to, home?): boolean`, `walkHome(id): boolean`, `tick(id)`, `sit(id): boolean`, `catStep()`, `reset()`; `deskOf(id)`, `blockedFor(id)`; hook `useOffice()`.

- [ ] **Step 1: Write the failing tests**

`apps/desktop/src/office/state.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { BOARD_SPOT, layoutDesks } from "./grid";
import { office } from "./state";

const desks = layoutDesks(["lead", "a", "b"]);

beforeEach(() => {
  office.reset();
  office.setDesks(desks);
});

describe("setDesks", () => {
  it("seats newcomers at their chair and selects the first", () => {
    const s = office.get();
    expect(s.positions.lead).toMatchObject({ tile: desks[0].chair, seated: true, facing: "w", path: [] });
    expect(s.selected).toBe("lead");
  });

  it("forgets agents that left and keeps the others where they were", () => {
    office.move("a", "e");
    const moved = office.get().positions.a.tile;
    office.setDesks(layoutDesks(["lead", "a"]));
    expect(office.get().positions.b).toBeUndefined();
    expect(office.get().positions.a.tile).toEqual(moved);
  });
});

describe("move", () => {
  it("steps one free tile, standing up", () => {
    expect(office.move("lead", "e")).toBe(true);
    expect(office.get().positions.lead).toMatchObject({ tile: { x: 2, z: 2 }, seated: false, facing: "e" });
  });

  it("is a no-op into furniture, keeping the new facing", () => {
    expect(office.move("lead", "w")).toBe(false); // the desk itself
    expect(office.get().positions.lead).toMatchObject({ tile: desks[0].chair, facing: "w" });
  });

  it("is blocked by another agent", () => {
    office.move("lead", "e"); // (2,2)
    office.move("lead", "s"); // (2,3)
    office.walkTo("a", { x: 2, z: 4 });
    while (office.get().positions.a.path.length) office.tick("a");
    expect(office.move("lead", "s")).toBe(false);
  });
});

describe("walking", () => {
  it("walkTo plans a path and tick follows it", () => {
    expect(office.walkTo("lead", BOARD_SPOT)).toBe(true);
    const n = office.get().positions.lead.path.length;
    expect(n).toBeGreaterThan(0);
    office.tick("lead");
    expect(office.get().positions.lead.path).toHaveLength(n - 1);
    expect(office.get().positions.lead.seated).toBe(false);
  });

  it("walkHome sits the agent down on arrival", () => {
    office.walkTo("lead", BOARD_SPOT);
    while (office.get().positions.lead.path.length) office.tick("lead");
    expect(office.walkHome("lead")).toBe(true);
    while (office.get().positions.lead.path.length) office.tick("lead");
    expect(office.get().positions.lead).toMatchObject({ tile: desks[0].chair, seated: true, facing: "w" });
  });

  it("walkHome fails when someone else is on the chair", () => {
    office.walkTo("lead", BOARD_SPOT);
    while (office.get().positions.lead.path.length) office.tick("lead");
    office.walkTo("a", desks[0].chair);
    while (office.get().positions.a.path.length) office.tick("a");
    expect(office.walkHome("lead")).toBe(false);
  });

  it("re-plans when the next tile gets blocked", () => {
    office.walkTo("lead", { x: 5, z: 2 }); // straight east along row 2
    const next = office.get().positions.lead.path[0];
    office.walkTo("a", next);
    while (office.get().positions.a.path.length) office.tick("a");
    office.tick("lead");
    expect(office.get().positions.lead.tile).toEqual(desks[0].chair); // did not walk into "a"
    expect(office.get().positions.lead.path[0]).not.toEqual(next);
  });

  it("sit works only on its own chair", () => {
    expect(office.sit("lead")).toBe(true);
    office.move("lead", "e");
    expect(office.sit("lead")).toBe(false);
  });
});

describe("cat", () => {
  it("strolls to a free neighbouring tile", () => {
    const before = office.get().cat;
    office.catStep();
    const after = office.get().cat;
    expect(Math.abs(after.x - before.x) + Math.abs(after.z - before.z)).toBe(1);
  });
});
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `npm test`
Expected: FAIL — cannot resolve `./state`.

- [ ] **Step 3: Implement `state.ts`**

```ts
// Who is where in the office: a small store outside React (positions survive closing the office) read through
// useOffice(). Movement is in whole tiles; the avatars animate between them and call tick() on arrival.
import { useSyncExternalStore } from "react";
import {
  blockedTiles,
  dirBetween,
  findPath,
  key,
  opposite,
  same,
  step,
  walkable,
  type Desk,
  type Dir,
  type Tile,
} from "./grid";

export interface AgentPos {
  tile: Tile;
  facing: Dir;
  seated: boolean;
  /** Tiles still to walk, the goal last. */
  path: Tile[];
  /** The walk ends at the agent's own chair: sit down on arrival. */
  home: boolean;
}

export interface OfficeState {
  positions: Record<string, AgentPos>;
  /** The agent the keyboard moves. */
  selected: string | null;
  cat: Tile;
}

const CAT_START: Tile = { x: 6, z: 4 };
let state: OfficeState = { positions: {}, selected: null, cat: CAT_START };
let desks: Desk[] = [];
let furniture = new Set<string>();
const listeners = new Set<() => void>();

function set(next: OfficeState) {
  state = next;
  listeners.forEach((l) => l());
}
const patch = (id: string, p: Partial<AgentPos>) =>
  set({ ...state, positions: { ...state.positions, [id]: { ...state.positions[id], ...p } } });

export const deskOf = (id: string) => desks.find((d) => d.id === id);

/** Everything `id` cannot walk into: furniture, the cat and the other agents. */
export function blockedFor(id: string | null): Set<string> {
  const s = new Set(furniture);
  s.add(key(state.cat));
  for (const [other, p] of Object.entries(state.positions)) if (other !== id) s.add(key(p.tile));
  return s;
}

const seatedAt = (d: Desk): AgentPos => ({ tile: d.chair, facing: opposite[d.facing], seated: true, path: [], home: false });

export const office = {
  get: () => state,
  subscribe(l: () => void) {
    listeners.add(l);
    return () => void listeners.delete(l);
  },
  /** The desks of the room, one agent each: newcomers start seated, agents that left are forgotten. */
  setDesks(next: Desk[]) {
    desks = next;
    furniture = blockedTiles(next);
    const positions: Record<string, AgentPos> = {};
    for (const d of next) positions[d.id] = state.positions[d.id] ?? seatedAt(d);
    const selected = state.selected && positions[state.selected] ? state.selected : (next[0]?.id ?? null);
    set({ ...state, positions, selected });
  },
  select(id: string | null) {
    set({ ...state, selected: id });
  },
  /** One tile in `dir` if it is free; the agent stands up and drops any walk. Returns whether it moved. */
  move(id: string, dir: Dir): boolean {
    const p = state.positions[id];
    if (!p) return false;
    const to = step(p.tile, dir);
    const free = walkable(to, blockedFor(id));
    patch(id, { tile: free ? to : p.tile, facing: dir, seated: false, path: [], home: false });
    return free;
  },
  /** Plans a walk to `to`; with `home`, the agent sits down when it gets there. Returns whether there is a way. */
  walkTo(id: string, to: Tile, home = false): boolean {
    const p = state.positions[id];
    if (!p) return false;
    const path = findPath(p.tile, to, blockedFor(id));
    if (!path) return false;
    const d = deskOf(id);
    if (!path.length) {
      if (home && d) patch(id, seatedAt(d));
      return true;
    }
    patch(id, { path, home, seated: false });
    return true;
  },
  walkHome(id: string): boolean {
    const d = deskOf(id);
    return !!d && office.walkTo(id, d.chair, true);
  },
  /** Takes the next tile of the walk (the avatar reached the current one); a tile taken meanwhile re-plans. */
  tick(id: string) {
    const p = state.positions[id];
    if (!p || !p.path.length) return;
    const [next, ...rest] = p.path;
    const blocked = blockedFor(id);
    if (!walkable(next, blocked)) {
      patch(id, { path: findPath(p.tile, p.path[p.path.length - 1], blocked) ?? [] });
      return;
    }
    const d = deskOf(id);
    const arrivedHome = rest.length === 0 && p.home && d && same(next, d.chair);
    patch(id, arrivedHome ? seatedAt(d) : { tile: next, facing: dirBetween(p.tile, next), path: rest });
  },
  /** Sits the agent down if it stands on its own chair. */
  sit(id: string): boolean {
    const d = deskOf(id);
    const p = state.positions[id];
    if (!d || !p || !same(p.tile, d.chair)) return false;
    patch(id, seatedAt(d));
    return true;
  },
  /** The cat strolls one tile in a random free direction. */
  catStep() {
    const blocked = blockedFor(null);
    const options = (["n", "s", "e", "w"] as Dir[]).map((d) => step(state.cat, d)).filter((t) => walkable(t, blocked));
    if (options.length) set({ ...state, cat: options[Math.floor(Math.random() * options.length)] });
  },
  /** Tests only. */
  reset() {
    desks = [];
    furniture = new Set();
    set({ positions: {}, selected: null, cat: CAT_START });
  },
};

export const useOffice = () => useSyncExternalStore(office.subscribe, office.get);
```

- [ ] **Step 4: Run the tests**

Run: `npm test`
Expected: all tests in both files PASS. `npm run typecheck` → clean.

- [ ] **Step 5: Commit (when asked)**

```bash
git add apps/desktop/src/office/state.ts apps/desktop/src/office/state.test.ts
git commit -m "feat(office): position store with walking and the cat"
```

---

### Task 3: Canvas textures (faces, boards) and text wrapping

**Files:**
- Create: `apps/desktop/src/office/textures.ts`
- Create: `apps/desktop/src/office/textures.test.ts` (for `wrapLines` and `pastel` only)
- Modify: `apps/desktop/package.json` (dependencies `three`, `@react-three/fiber`, `@react-three/drei`; devDependency `@types/three`)

**Interfaces:**
- Produces: `MARKER_FONT`, `emojiTexture(emoji): CanvasTexture`, `textBoard(width, height, opts): { texture: CanvasTexture; draw(lines: string[], chars?: number): void }`, `wrapLines(text, cols, max): string[]`, `pastel(id): string`.

- [ ] **Step 1: Install the 3D dependencies**

Run: `npm i three@0.186.1 @react-three/fiber@9.8.2 @react-three/drei@10.7.9 && npm i -D @types/three@0.186.0`

- [ ] **Step 2: Write the failing test**

`apps/desktop/src/office/textures.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { pastel, wrapLines } from "./textures";

describe("wrapLines", () => {
  it("wraps on words and caps the number of lines", () => {
    const lines = wrapLines("The quick brown fox jumps over the lazy dog again and again", 16, 3);
    expect(lines).toEqual(["The quick brown", "fox jumps over", "the lazy dog ag…"]);
  });

  it("drops markdown marks and joins paragraphs (the board is a summary, not a document)", () => {
    expect(wrapLines("## Title\n\n**bold** text\n", 40, 4)).toEqual(["Title bold text"]);
  });

  it("returns nothing for empty text", () => {
    expect(wrapLines("   \n", 20, 3)).toEqual([]);
  });
});

describe("pastel", () => {
  it("is stable per id and a hex colour", () => {
    expect(pastel("ag-1")).toBe(pastel("ag-1"));
    expect(pastel("lead")).toMatch(/^#[0-9a-f]{6}$/);
  });
});
```

- [ ] **Step 3: Run it to see it fail**

Run: `npm test`
Expected: FAIL — cannot resolve `./textures`.

- [ ] **Step 4: Implement `textures.ts`**

```ts
// Canvas-drawn textures for the office (faces, whiteboard, poster) and the small text helpers they need. Nothing
// here touches React; the scene asks for a texture and redraws the boards through draw().
import { CanvasTexture, SRGBColorSpace } from "three";

const EMOJI_FONT = "'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif";
/** A marker-like hand: the fonts most machines have, in order of charm. */
export const MARKER_FONT = "'Segoe Print', 'Bradley Hand', 'Comic Sans MS', 'Chalkboard SE', cursive";

const faces = new Map<string, CanvasTexture>();

/** A square face with the emoji on a skin-toned background (cached per emoji). */
export function emojiTexture(emoji: string): CanvasTexture {
  const hit = faces.get(emoji);
  if (hit) return hit;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  g.fillStyle = "#ffdcb3";
  g.fillRect(0, 0, 128, 128);
  g.font = `84px ${EMOJI_FONT}`;
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(emoji, 64, 70);
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  faces.set(emoji, t);
  return t;
}

/** A board that can be redrawn: draw() shows the first `chars` characters of `lines`, the first line in bold. */
export function textBoard(width: number, height: number, opts: { bg: string; ink: string; size: number; font?: string }) {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const g = c.getContext("2d")!;
  const texture = new CanvasTexture(c);
  texture.colorSpace = SRGBColorSpace;
  const font = opts.font ?? MARKER_FONT;
  const draw = (lines: string[], chars = Infinity) => {
    g.fillStyle = opts.bg;
    g.fillRect(0, 0, width, height);
    g.fillStyle = opts.ink;
    g.textBaseline = "top";
    const pad = opts.size * 0.8;
    let left = chars;
    lines.forEach((line, i) => {
      if (left <= 0) return;
      g.font = `${i === 0 ? "bold " : ""}${opts.size}px ${font}`;
      g.fillText(line.slice(0, left), pad, pad + i * opts.size * 1.4, width - 2 * pad);
      left -= line.length;
    });
    texture.needsUpdate = true;
  };
  draw([]);
  return { texture, draw };
}

/** `text` as at most `max` lines of about `cols` characters, markdown marks dropped and paragraphs run together;
 * the last line ends with "…" when text was left over. */
export function wrapLines(text: string, cols: number, max: number): string[] {
  const words = text
    .replace(/[#*_`>]+/g, "")
    .split(/\s+/)
    .filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    if (line && (line + " " + w).length > cols) {
      lines.push(line);
      line = w;
    } else line = line ? `${line} ${w}` : w;
  }
  if (line) lines.push(line);
  if (lines.length > max) return [...lines.slice(0, max - 1), lines.slice(max - 1).join(" ").slice(0, cols - 1) + "…"];
  return lines;
}

const PASTELS = ["#f6a6b2", "#f9c784", "#ffe599", "#b5e7a0", "#9ad0ec", "#c3b1e1", "#f4b6c2", "#a0e7e5", "#ffcbf2", "#b4f8c8"];

/** A body colour that stays the same for an agent id. */
export function pastel(id: string): string {
  let h = 0;
  for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return PASTELS[h % PASTELS.length];
}
```

- [ ] **Step 5: Run the tests**

Run: `npm test`
Expected: PASS. `npm run typecheck` → clean (vitest runs in node; `document` is only touched inside functions the tests do not call).

- [ ] **Step 6: Commit (when asked)**

```bash
git add apps/desktop/package.json apps/desktop/package-lock.json apps/desktop/src/office/textures.ts apps/desktop/src/office/textures.test.ts
git commit -m "feat(office): canvas textures and text helpers; add three/r3f/drei"
```

---

### Task 4: The room, the overlay and the button

**Files:**
- Create: `apps/desktop/src/office/room.tsx`
- Create: `apps/desktop/src/office/Office.tsx`
- Create: `apps/desktop/src/i18n/en/office.ts`
- Modify: `apps/desktop/src/i18n/i18n_en.ts`
- Modify: `apps/desktop/src/App.tsx` (lucide import line 4; state near line 74; rail foot lines 481–494; overlays near line 632)
- Modify: `apps/desktop/src/styles.css` (append)

**Interfaces:**
- Consumes: `grid.ts`, `state.ts`, `textures.ts`.
- Produces: `room.tsx` exports `YAW: Record<Dir, number>`, `Floor`, `Walls`, `Whiteboard({ lines, chars })`, `DeskMesh({ desk, lead, running, lit, failed })`, `CoffeeMachine`, `Poster`, `Cat({ tile })`; `Office.tsx` exports `Office(props: OfficeProps)` with `OfficeProps { team?: TeamConfig; turns: Turn[]; turnKey: string | null; onSelectTurn(key: string | null): void; reporting: Reporting; onOpenObject(name: string, blockId?: string | null): void; onClose(): void }`.

- [ ] **Step 1: i18n keys**

`apps/desktop/src/i18n/en/office.ts`:

```ts
// English texts: the 3D office (office/)
export const office = {
  "office.button": "Office",
  "office.title": "The office",
  "office.close": "Close the office",
  "office.hint": "Pick a teammate below and walk with the arrow keys. Stand at a desk to read its screen, or in front of the whiteboard for the results.",
  "office.whiteboard": "Whiteboard",
  "office.results": "Results",
  "office.leadAnswer": "The lead's answer",
  "office.nothingYet": "(nothing on the board yet)",
  "office.noReport": "Nothing to show yet: ask something and the team will write it up here.",
  "office.noWebgl": "This machine cannot draw the 3D office (no WebGL). The team keeps working in the workspace panel.",
  "office.poster": "KEEP CALM AND CLUSTER YOUR STANDARD ERRORS",
  "office.seated": "sitting (arrows to get up)",
  "office.prevTurn": "Previous turn",
  "office.nextTurn": "Next turn",
  "office.closePanel": "Close panel",
} as const;
```

In `src/i18n/i18n_en.ts` add `import { office } from "./en/office";` and `...office` to the spread (keep alphabetical: after `notebooks`).

- [ ] **Step 2: `room.tsx`**

```tsx
// The furniture of the office: floor, walls, whiteboard, desks, the coffee machine, the poster and the cat. Every
// piece is boxes and planes in flat toon colours; positions come from grid.ts tiles.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import type { Group, Mesh } from "three";
import { T } from "../i18n";
import { COLS, ROWS, world, type Desk, type Dir, type Tile } from "./grid";
import { MARKER_FONT, textBoard } from "./textures";

/** Rotation about y that turns a model's front (+z) toward a direction. */
export const YAW: Record<Dir, number> = { s: 0, e: Math.PI / 2, n: Math.PI, w: -Math.PI / 2 };

const at = (t: Tile, y = 0): [number, number, number] => {
  const [x, z] = world(t);
  return [x, y, z];
};

export function Floor() {
  const tiles = useMemo(() => {
    const out: { pos: [number, number, number]; dark: boolean }[] = [];
    for (let x = 0; x < COLS; x++) for (let z = 0; z < ROWS; z++) out.push({ pos: at({ x, z }, -0.05), dark: (x + z) % 2 === 0 });
    return out;
  }, []);
  return (
    <group>
      {tiles.map((t, i) => (
        <mesh key={i} position={t.pos} receiveShadow>
          <boxGeometry args={[0.98, 0.1, 0.98]} />
          <meshToonMaterial color={t.dark ? "#e9d8c3" : "#f5e9d9"} />
        </mesh>
      ))}
    </group>
  );
}

export function Walls() {
  const h = 2.6;
  return (
    <group>
      <mesh position={[0, h / 2, -ROWS / 2 - 0.1]} receiveShadow>
        <boxGeometry args={[COLS + 0.4, h, 0.2]} />
        <meshToonMaterial color="#cfe3f5" />
      </mesh>
      <mesh position={[-COLS / 2 - 0.1, h / 2, 0]} receiveShadow>
        <boxGeometry args={[0.2, h, ROWS + 0.4]} />
        <meshToonMaterial color="#bcd6ef" />
      </mesh>
      {/* a skirting line so the walls do not float */}
      <mesh position={[0, 0.12, -ROWS / 2 - 0.02]}>
        <boxGeometry args={[COLS + 0.4, 0.24, 0.06]} />
        <meshToonMaterial color="#8fb3d9" />
      </mesh>
    </group>
  );
}

/** The whiteboard on the north wall: `lines` written up to `chars` characters. */
export function Whiteboard({ lines, chars }: { lines: string[]; chars: number }) {
  const board = useMemo(() => textBoard(1024, 512, { bg: "#fbfbf7", ink: "#2b3a67", size: 46 }), []);
  useEffect(() => board.draw(lines, chars), [board, lines, chars]);
  return (
    <group position={[0, 1.55, -ROWS / 2 + 0.03]}>
      <mesh>
        <boxGeometry args={[4.2, 2.1, 0.06]} />
        <meshToonMaterial color="#6b7a90" />
      </mesh>
      <mesh position={[0, 0, 0.035]}>
        <planeGeometry args={[4, 1.9]} />
        <meshBasicMaterial map={board.texture} toneMapped={false} />
      </mesh>
      <mesh position={[0, -1.1, 0.12]}>
        <boxGeometry args={[4.2, 0.08, 0.24]} />
        <meshToonMaterial color="#56657b" />
      </mesh>
    </group>
  );
}

/** The monitor's picture: coloured bars that pulse while the agent works; dark until it ever worked. */
function Screen({ running, lit }: { running: boolean; lit: boolean }) {
  const bars = useRef<(Mesh | null)[]>([]);
  useFrame(({ clock }) => {
    if (!running) return;
    bars.current.forEach((m, i) => {
      if (m) m.scale.y = 0.45 + 0.55 * Math.abs(Math.sin(clock.elapsedTime * 3 + i * 0.9));
    });
  });
  const colors = ["#ff8fab", "#ffd166", "#06d6a0", "#118ab2", "#9b5de5", "#f15bb5", "#00bbf9"];
  return (
    <group>
      <mesh>
        <boxGeometry args={[0.7, 0.46, 0.03]} />
        <meshBasicMaterial color={lit ? "#1b2a4a" : "#2a2a2e"} />
      </mesh>
      {lit &&
        colors.map((c, i) => (
          <mesh key={c} ref={(m) => void (bars.current[i] = m)} position={[-0.27 + i * 0.09, -0.08, 0.02]}>
            <boxGeometry args={[0.06, 0.24, 0.01]} />
            <meshBasicMaterial color={c} />
          </mesh>
        ))}
    </group>
  );
}

export function DeskMesh({ desk, lead, running, lit, failed }: { desk: Desk; lead: boolean; running: boolean; lit: boolean; failed: boolean }) {
  const [x0, z0] = world(desk.tiles[0]);
  const [x1, z1] = world(desk.tiles[1]);
  const centre: [number, number, number] = [(x0 + x1) / 2, 0, (z0 + z1) / 2];
  const yaw = YAW[desk.facing];
  const alongZ = desk.tiles[0].x === desk.tiles[1].x;
  return (
    <group>
      {/* table top and legs, the long side along the wall */}
      <group position={centre} rotation={[0, alongZ ? Math.PI / 2 : 0, 0]}>
        <mesh position={[0, 0.72, 0]} castShadow receiveShadow>
          <boxGeometry args={[1.9, 0.08, 0.8]} />
          <meshToonMaterial color={lead ? "#d9a066" : "#e8c39e"} />
        </mesh>
        {[-0.85, 0.85].flatMap((lx) =>
          [-0.3, 0.3].map((lz) => (
            <mesh key={`${lx}${lz}`} position={[lx, 0.36, lz]}>
              <boxGeometry args={[0.08, 0.72, 0.08]} />
              <meshToonMaterial color="#8c6a4a" />
            </mesh>
          )),
        )}
      </group>
      {/* monitor over the first tile, facing the chair */}
      <group position={[x0, 0.76, z0]} rotation={[0, yaw, 0]}>
        <mesh position={[0, 0.08, 0]}>
          <boxGeometry args={[0.3, 0.16, 0.2]} />
          <meshToonMaterial color="#555c6b" />
        </mesh>
        <group position={[0, 0.42, 0.1]}>
          <Screen running={running} lit={lit} />
        </group>
        {failed && <Smoke />}
        <mesh position={[0.5, 0.1, 0.15]}>
          <cylinderGeometry args={[0.07, 0.06, 0.14, 10]} />
          <meshToonMaterial color="#ffffff" />
        </mesh>
      </group>
      {lead && (
        <group position={[x1, 0.76, z1]}>
          <mesh position={[0, 0.08, 0]}>
            <cylinderGeometry args={[0.1, 0.08, 0.16, 10]} />
            <meshToonMaterial color="#c9744a" />
          </mesh>
          <mesh position={[0, 0.3, 0]}>
            <sphereGeometry args={[0.2, 8, 6]} />
            <meshToonMaterial color="#5fa85a" />
          </mesh>
        </group>
      )}
      {/* the chair, its back away from the desk */}
      <group position={at(desk.chair)} rotation={[0, yaw, 0]}>
        <mesh position={[0, 0.4, 0]} castShadow>
          <boxGeometry args={[0.5, 0.08, 0.5]} />
          <meshToonMaterial color="#4a5a7a" />
        </mesh>
        <mesh position={[0, 0.62, 0.22]}>
          <boxGeometry args={[0.5, 0.5, 0.06]} />
          <meshToonMaterial color="#4a5a7a" />
        </mesh>
        <mesh position={[0, 0.2, 0]}>
          <cylinderGeometry args={[0.04, 0.04, 0.4, 6]} />
          <meshToonMaterial color="#2f3a4f" />
        </mesh>
      </group>
    </group>
  );
}

/** A puff of smoke over a monitor whose agent failed. */
function Smoke() {
  const g = useRef<Group>(null);
  useFrame(({ clock }) => {
    if (!g.current) return;
    g.current.children.forEach((c, i) => {
      const t = (clock.elapsedTime * 0.6 + i * 0.33) % 1;
      c.position.y = 0.7 + t * 0.8;
      c.scale.setScalar(0.5 + t * 0.8);
      c.visible = t < 0.9;
    });
  });
  return (
    <group ref={g}>
      {[0, 1, 2].map((i) => (
        <mesh key={i} position={[(i - 1) * 0.1, 0.7, 0]}>
          <sphereGeometry args={[0.12, 6, 5]} />
          <meshToonMaterial color="#9aa3ad" transparent opacity={0.8} />
        </mesh>
      ))}
    </group>
  );
}

export function CoffeeMachine() {
  const [x, z] = world({ x: 10, z: 0 });
  return (
    <group position={[x + 0.5, 0, z]}>
      <mesh position={[0, 0.45, 0]} castShadow>
        <boxGeometry args={[1.6, 0.9, 0.8]} />
        <meshToonMaterial color="#e3e8ef" />
      </mesh>
      <mesh position={[-0.3, 1.25, -0.1]} castShadow>
        <boxGeometry args={[0.6, 0.7, 0.5]} />
        <meshToonMaterial color="#3b3f48" />
      </mesh>
      <mesh position={[-0.3, 1.0, 0.2]}>
        <cylinderGeometry args={[0.08, 0.07, 0.14, 10]} />
        <meshToonMaterial color="#ffffff" />
      </mesh>
      <mesh position={[0.4, 1.0, 0]}>
        <cylinderGeometry args={[0.18, 0.18, 0.2, 12]} />
        <meshToonMaterial color="#d97757" />
      </mesh>
    </group>
  );
}

export function Poster() {
  const board = useMemo(() => textBoard(512, 640, { bg: "#ffd166", ink: "#2b3a67", size: 56, font: MARKER_FONT }), []);
  useEffect(() => board.draw(T("office.poster").split(" ")), [board]);
  return (
    <mesh position={[-COLS / 2 + 0.02, 1.6, 1.5]} rotation={[0, Math.PI / 2, 0]}>
      <planeGeometry args={[1.1, 1.4]} />
      <meshBasicMaterial map={board.texture} toneMapped={false} />
    </mesh>
  );
}

export function Cat({ tile }: { tile: Tile }) {
  const g = useRef<Group>(null);
  const target = at(tile, 0.12);
  useFrame((_, dt) => {
    if (!g.current) return;
    g.current.position.x += (target[0] - g.current.position.x) * Math.min(1, dt * 4);
    g.current.position.z += (target[2] - g.current.position.z) * Math.min(1, dt * 4);
  });
  return (
    <group ref={g} position={target}>
      <mesh castShadow>
        <boxGeometry args={[0.36, 0.2, 0.22]} />
        <meshToonMaterial color="#f0a35e" />
      </mesh>
      <mesh position={[0.2, 0.12, 0]}>
        <boxGeometry args={[0.18, 0.18, 0.2]} />
        <meshToonMaterial color="#f0a35e" />
      </mesh>
      <mesh position={[-0.24, 0.12, 0]} rotation={[0, 0, 0.6]}>
        <boxGeometry args={[0.2, 0.05, 0.05]} />
        <meshToonMaterial color="#d88a45" />
      </mesh>
    </group>
  );
}
```

- [ ] **Step 3: `Office.tsx` (room only; avatars, bar and panel come in later tasks)**

```tsx
// The office: a full-screen 3D view of the team at its desks. The room and the avatars live in the canvas; the
// bottom bar and the readable side panel are plain HTML over it. Movement rules are in state.ts.
import { OrthographicCamera } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { LEAD_ID, type TeamConfig } from "../api";
import type { Turn } from "../chat";
import { T } from "../i18n";
import type { Reporting } from "../reporting";
import { layoutDesks } from "./grid";
import { Cat, CoffeeMachine, DeskMesh, Floor, Poster, Walls, Whiteboard } from "./room";
import { office, useOffice } from "./state";

export interface OfficeProps {
  team?: TeamConfig;
  turns: Turn[];
  /** The turn shown in the panels (null = the latest), shared with the workspace. */
  turnKey: string | null;
  onSelectTurn(key: string | null): void;
  reporting: Reporting;
  onOpenObject(name: string, blockId?: string | null): void;
  onClose(): void;
}

/** The lead when settings have no team yet. */
const LEAD = { id: LEAD_ID, name: "", emoji: "", enabled: true };

function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Fits the room to the viewport: an isometric orthographic camera whose zoom follows the window and the wheel. */
function CameraRig({ zoom }: { zoom: number }) {
  const size = useThree((s) => s.size);
  const base = Math.min(size.width, size.height) / 15;
  return <OrthographicCamera makeDefault position={[12, 12, 12]} zoom={base * zoom} near={0.1} far={100} onUpdate={(c) => c.lookAt(0, 0.4, 0)} />;
}

export function Office(props: OfficeProps) {
  const { onClose } = props;
  const webgl = useMemo(hasWebGL, []);
  const [zoom, setZoom] = useState(1);
  const members = useMemo(() => {
    const agents = props.team?.agents.length ? props.team.agents : [LEAD];
    return props.team?.enabled ? agents.filter((a) => a.enabled) : agents.slice(0, 1);
  }, [props.team]);
  const desks = useMemo(() => layoutDesks(members.map((m) => m.id)), [members]);
  useEffect(() => office.setDesks(desks), [desks]);
  const state = useOffice();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    const id = setInterval(office.catStep, 2500);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="office" role="dialog" aria-label={T("office.title")}>
      <header className="office-top">
        <span className="office-title">{T("office.title")}</span>
        <span className="muted small office-hint">{T("office.hint")}</span>
        <span className="spacer" />
        <button className="ghost" title={T("office.close")} onClick={onClose}>
          <X size={16} />
        </button>
      </header>
      {webgl ? (
        <Canvas shadows onWheel={(e) => setZoom((z) => Math.min(1.6, Math.max(0.7, z - e.deltaY * 0.001)))}>
          <color attach="background" args={["#dbeafe"]} />
          <CameraRig zoom={zoom} />
          <hemisphereLight args={["#ffffff", "#c9b8a8", 0.9]} />
          <directionalLight position={[6, 12, 4]} intensity={1.6} castShadow shadow-mapSize={[2048, 2048]} shadow-camera-left={-10} shadow-camera-right={10} shadow-camera-top={10} shadow-camera-bottom={-10} />
          <Floor />
          <Walls />
          <Whiteboard lines={[T("office.nothingYet")]} chars={Infinity} />
          <CoffeeMachine />
          <Poster />
          {desks.map((d, i) => (
            <DeskMesh key={d.id} desk={d} lead={i === 0} running={false} lit={false} failed={false} />
          ))}
          <Cat tile={state.cat} />
        </Canvas>
      ) : (
        <div className="office-nowebgl">
          <p>{T("office.noWebgl")}</p>
          <button className="primary" onClick={onClose}>
            {T("office.close")}
          </button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Wire the button in `App.tsx`**

- Line 4: add `Building2` to the `lucide-react` import (keep alphabetical).
- After `import { TeamModal } from "./team";` (line 31) add `import { Office } from "./office/Office";`.
- After `const [showTeam, setShowTeam] = useState(false);` (line 74) add `const [showOffice, setShowOffice] = useState(false);` and, next to `closeWorkspace`, `const closeOffice = useCallback(() => setShowOffice(false), []);`.
- In the rail foot, directly **before** the team button (line 487) insert:

```tsx
          <button className="ghost wide" disabled={!settings} onClick={() => setShowOffice(true)}>
            <Building2 size={15} /> {T("office.button")}
          </button>
```

- Next to the other overlays (after the `showTeam` line, 632) add:

```tsx
      {showOffice && (
        <Office
          team={settings?.team}
          turns={turns}
          turnKey={wsTurn}
          onSelectTurn={setWsTurn}
          reporting={reporting}
          onOpenObject={(name, blockId) => {
            setShowOffice(false);
            openObject(name, blockId);
          }}
          onClose={closeOffice}
        />
      )}
```

- [ ] **Step 5: CSS (append to `styles.css`)**

```css
/* The 3D office: full-screen overlay; the canvas fills it, the bar and the panel float over it. */
.office { position: fixed; inset: 0; z-index: 60; background: #dbeafe; display: flex; flex-direction: column; }
.office canvas { flex: 1; min-height: 0; outline: none; }
.office-top { display: flex; align-items: center; gap: 12px; padding: 8px 14px; background: var(--surface); border-bottom: 1px solid var(--line); }
.office-title { font-weight: 600; }
.office-hint { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.office-nowebgl { margin: auto; text-align: center; max-width: 420px; display: flex; flex-direction: column; gap: 12px; align-items: center; }
```

- [ ] **Step 6: Typecheck, build and look**

Run: `npm run typecheck && npm run build`
Expected: clean. Then `npm run dev` (from `apps/desktop`; the engine via `npm start` at the repo root if the app needs it): click **Office** above **Agent team** → a bright room with checkered floor, two walls, whiteboard reading "(nothing on the board yet)", one desk per enabled member (lead's with a plant), coffee machine, poster, a cat wandering. Wheel zooms. Escape closes. Turn the team on with 3 members → 3 desks.

- [ ] **Step 7: Commit (when asked)**

```bash
git add apps/desktop/src/office apps/desktop/src/i18n apps/desktop/src/App.tsx apps/desktop/src/styles.css
git commit -m "feat(office): room, overlay and the Office button"
```

---

### Task 5: Avatars, the bottom bar and keyboard movement

**Files:**
- Create: `apps/desktop/src/office/avatar.tsx`
- Modify: `apps/desktop/src/office/Office.tsx`
- Modify: `apps/desktop/src/styles.css` (append)

**Interfaces:**
- Consumes: `office`, `useOffice`, `AgentPos` (state.ts); `world`, `YAW`; `emojiTexture`, `pastel`.
- Produces: `Avatar({ id, name, emoji, pos, selected, live })` and `AgentLive { running: boolean; status?: string; failed: boolean }`.

- [ ] **Step 1: `avatar.tsx`**

```tsx
// One agent in the office: a pastel capsule with an emoji face. It glides between the tiles the store gives it
// (hopping as it walks, calling office.tick on arrival) and shows what the agent is doing in a bubble.
import { Html } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Group } from "three";
import { T } from "../i18n";
import { world } from "./grid";
import { YAW } from "./room";
import { office, type AgentPos } from "./state";
import { emojiTexture, pastel } from "./textures";

/** What the agent is doing right now, from the chat state. */
export interface AgentLive {
  running: boolean;
  status?: string;
  failed: boolean;
}

const WALK_SPEED = 5.5; // tiles per second ≈ 180 ms per tile
const FALLBACK_FACE = "🧑‍💼";

export function Avatar({ id, name, emoji, pos, selected, live }: { id: string; name: string; emoji: string; pos: AgentPos; selected: boolean; live: AgentLive }) {
  const root = useRef<Group>(null);
  const body = useRef<Group>(null);
  const arms = useRef<Group>(null);
  const face = useMemo(() => emojiTexture(emoji || FALLBACK_FACE), [emoji]);
  const colour = useMemo(() => pastel(id), [id]);
  const target = world(pos.tile);
  const travel = useRef(0); // 0..1 progress of the current hop

  // When a run ends well: ✓ for a few seconds.
  const [done, setDone] = useState(false);
  const wasRunning = useRef(live.running);
  useEffect(() => {
    const ended = wasRunning.current && !live.running && !live.failed;
    wasRunning.current = live.running;
    if (!ended) return;
    setDone(true);
    const t = setTimeout(() => setDone(false), 4000);
    return () => clearTimeout(t);
  }, [live.running, live.failed]);

  useFrame((s, dt) => {
    const g = root.current;
    if (!g) return;
    const dx = target[0] - g.position.x;
    const dz = target[1] - g.position.z;
    const dist = Math.hypot(dx, dz);
    if (dist > 0.01) {
      const stepLen = Math.min(dist, WALK_SPEED * dt);
      g.position.x += (dx / dist) * stepLen;
      g.position.z += (dz / dist) * stepLen;
      travel.current = Math.min(1, travel.current + dt * WALK_SPEED);
      g.position.y = Math.sin(travel.current * Math.PI) * 0.18;
    } else {
      if (travel.current > 0) {
        travel.current = 0;
        g.position.set(target[0], 0, target[1]);
        office.tick(id);
      }
      g.position.y = pos.seated ? -0.12 : Math.sin(s.clock.elapsedTime * 2 + g.position.x) * 0.02;
    }
    const yaw = YAW[pos.facing];
    g.rotation.y += (yaw - g.rotation.y) * Math.min(1, dt * 10);
    if (arms.current) arms.current.rotation.x = live.running && pos.seated ? Math.sin(s.clock.elapsedTime * 14) * 0.25 - 0.9 : 0;
    if (body.current) body.current.rotation.z = pos.seated && !live.running ? Math.sin(s.clock.elapsedTime * 1.3) * 0.04 : 0;
  });

  const bubble = live.running ? `… ${live.status ?? ""}`.trim() : live.failed ? "💥" : done ? "✓" : null;

  return (
    <group ref={root} position={[target[0], 0, target[1]]}>
      {selected && (
        <mesh position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.32, 0.42, 24]} />
          <meshBasicMaterial color="#2f62d6" />
        </mesh>
      )}
      <group ref={body}>
        <mesh position={[0, 0.5, 0]} castShadow>
          <capsuleGeometry args={[0.22, 0.4, 4, 10]} />
          <meshToonMaterial color={colour} />
        </mesh>
        <group ref={arms} position={[0, 0.62, 0]}>
          {[-0.3, 0.3].map((x) => (
            <mesh key={x} position={[x, -0.12, 0]} castShadow>
              <boxGeometry args={[0.1, 0.34, 0.1]} />
              <meshToonMaterial color={colour} />
            </mesh>
          ))}
        </group>
        <mesh position={[0, 1.08, 0]} castShadow>
          <boxGeometry args={[0.44, 0.44, 0.44]} />
          {[0, 1, 2, 3, 5].map((i) => (
            <meshToonMaterial key={i} attach={`material-${i}`} color="#ffdcb3" />
          ))}
          <meshToonMaterial attach="material-4" map={face} />
        </mesh>
      </group>
      <Html position={[0, 1.55, 0]} center className="office-label" zIndexRange={[10, 0]}>
        <span className="office-name">{name || "—"}</span>
        {bubble && <span className={`office-bubble ${live.failed ? "bad" : ""}`}>{bubble}</span>}
        {!bubble && pos.seated && selected && <span className="office-hint-tag">{T("office.seated")}</span>}
      </Html>
    </group>
  );
}
```

- [ ] **Step 2: Wire avatars, bar and keys in `Office.tsx`**

Add imports: `import { Avatar, type AgentLive } from "./avatar";`, `LoaderCircle` to the lucide import, `type Dir` to the grid import.

Inside `Office`, after `const state = useOffice();`:

```tsx
  // Live state of each agent, from the latest turn: who is working, on what, who failed.
  const last = props.turns[props.turns.length - 1];
  const live = useMemo(() => {
    const out: Record<string, AgentLive> = {};
    for (const m of members) out[m.id] = { running: false, failed: false };
    if (last) {
      out[LEAD_ID] = { running: last.running, status: last.status, failed: !!last.error };
      for (const a of last.agents ?? []) if (out[a.id]) out[a.id] = { running: a.running, status: a.status, failed: a.ok === false || !!a.error };
    }
    return out;
  }, [members, last]);

  // Monitors light up once their agent has worked in any turn.
  const worked = useMemo(() => {
    const s = new Set<string>();
    if (props.turns.length) s.add(LEAD_ID);
    for (const t of props.turns) for (const a of t.agents ?? []) s.add(a.id);
    return s;
  }, [props.turns]);

  // Arrow keys walk the selected agent; Enter sits it down.
  useEffect(() => {
    const DIRS: Record<string, Dir> = { ArrowUp: "n", ArrowDown: "s", ArrowLeft: "w", ArrowRight: "e" };
    const onKey = (e: KeyboardEvent) => {
      const id = office.get().selected;
      if (!id) return;
      if (DIRS[e.key]) {
        e.preventDefault();
        office.move(id, DIRS[e.key]);
      } else if (e.key === "Enter") office.sit(id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
```

Replace the `DeskMesh` map and add avatars inside the `<Canvas>`:

```tsx
          {desks.map((d, i) => (
            <DeskMesh key={d.id} desk={d} lead={i === 0} running={live[d.id]?.running ?? false} lit={worked.has(d.id)} failed={live[d.id]?.failed ?? false} />
          ))}
          {members.map((m) =>
            state.positions[m.id] ? (
              <Avatar key={m.id} id={m.id} name={m.name} emoji={m.emoji} pos={state.positions[m.id]} selected={state.selected === m.id} live={live[m.id]} />
            ) : null,
          )}
```

After the `<Canvas>`/no-WebGL block, still inside `.office`, add the bar:

```tsx
      <nav className="office-bar" aria-label={T("office.title")}>
        {members.map((m) => (
          <button key={m.id} className={`office-chip ${state.selected === m.id ? "on" : ""}`} onClick={() => office.select(m.id)} title={m.name}>
            {live[m.id]?.running ? <LoaderCircle size={13} className="spin" /> : <span aria-hidden>{m.emoji || "🧑‍💼"}</span>}
            <span className="office-chip-name">{m.name || "—"}</span>
          </button>
        ))}
      </nav>
```

(The whiteboard chip is added in Task 7 with the panel.)

- [ ] **Step 3: CSS (append)**

```css
.office-bar { position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%); display: flex; gap: 6px; padding: 6px; background: var(--surface); border: 1px solid var(--line); border-radius: 999px; box-shadow: var(--shadow); max-width: calc(100% - 32px); overflow-x: auto; }
.office-chip { border-radius: 999px; padding: 6px 12px 6px 9px; border-color: transparent; }
.office-chip.on { background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
.office-chip-name { max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.office-label { pointer-events: none; display: flex; flex-direction: column; align-items: center; gap: 3px; white-space: nowrap; }
.office-name { font-size: 11px; font-weight: 600; color: #2b3a67; background: rgba(255, 255, 255, 0.85); padding: 1px 7px; border-radius: 999px; }
.office-bubble { font-size: 12px; background: #fff; border: 1.5px solid #2b3a67; border-radius: 10px; padding: 3px 8px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; animation: enter 0.25s ease-out; }
.office-bubble.bad { border-color: var(--warm); background: var(--warm-soft); }
.office-hint-tag { font-size: 10px; color: var(--ink-3); }
```

- [ ] **Step 4: Typecheck, build, look**

Run: `npm run typecheck && npm run build`, then `npm run dev`. Expected: avatars sit at their chairs facing the desks, names above heads; the first chip is selected (ring under the lead). Arrow keys walk it off the chair with a hop; desks, walls, the cat and other agents block; Enter back on the chair sits it. Clicking another chip moves the ring. Send a request: the lead's bubble shows "… status", the monitor bars pulse; with the team on, members' bubbles appear as they work; ✓ for a few seconds when done. Close and reopen the office: everyone is where they were.

- [ ] **Step 5: Commit (when asked)**

```bash
git add apps/desktop/src/office apps/desktop/src/styles.css
git commit -m "feat(office): avatars, bottom bar and arrow-key walking"
```

---

### Task 6: Coming back to work, coffee trips

**Files:**
- Modify: `apps/desktop/src/office/Office.tsx`

**Interfaces:**
- Consumes: `office.walkHome`, `office.walkTo`, `COFFEE_SPOT`.

- [ ] **Step 1: Walk home when a turn starts or a member starts working**

In `Office`, after the `live` memo:

```tsx
  // A new turn, or a member starting on its task, sends the agents back to their desks (the user may walk them
  // off again; they just stand up).
  const runningIds = Object.entries(live)
    .filter(([, l]) => l.running)
    .map(([id]) => id)
    .join(",");
  useEffect(() => {
    for (const id of runningIds.split(",").filter(Boolean)) {
      const p = office.get().positions[id];
      if (p && !p.seated) office.walkHome(id);
    }
  }, [runningIds]);
```

- [ ] **Step 2: Coffee trips**

```tsx
  // Every so often an idle agent nobody is steering goes for a coffee and comes back.
  useEffect(() => {
    const trip = () => {
      const s = office.get();
      const idle = members.filter((m) => s.positions[m.id]?.seated && s.selected !== m.id && !live[m.id]?.running);
      if (!idle.length) return;
      const id = idle[Math.floor(Math.random() * idle.length)].id;
      if (office.walkTo(id, COFFEE_SPOT)) setTimeout(() => office.walkHome(id), 6000);
    };
    const timer = setInterval(trip, 20000);
    return () => clearInterval(timer);
  }, [members, live]);
```

Add `COFFEE_SPOT` to the grid import.

- [ ] **Step 3: Typecheck, build, look**

Run: `npm run typecheck && npm run build`, then `npm run dev`. Expected: walk two agents away from their desks, send a request → they walk back (around furniture) and sit; wait ~20 s with the office idle → one unselected agent walks to the coffee machine and returns.

- [ ] **Step 4: Commit (when asked)**

```bash
git add apps/desktop/src/office/Office.tsx
git commit -m "feat(office): agents return to their desks and take coffee breaks"
```

---

### Task 7: The side panel and the whiteboard

**Files:**
- Create: `apps/desktop/src/office/panel.tsx`
- Modify: `apps/desktop/src/office/Office.tsx`
- Modify: `apps/desktop/src/workspace.tsx:332,351,413` (export `MemberConclusions`, `mergeAgents`, `TurnWork`)
- Modify: `apps/desktop/src/styles.css` (append)

**Interfaces:**
- Consumes: `TurnWork({ turn: Work; onOpenObject })`, `MemberConclusions({ agent: AgentWork })`, `mergeAgents(turns): AgentWork[]` from `workspace.tsx`; `Markdown` from `blocks.tsx`; `ConversationReport` from `api.ts`; `wrapLines`, `BOARD_SPOT`, `READING_ZONE`, `same`.
- Produces: `OfficePanel(props)`, `type Panel = { kind: "desk"; id: string } | { kind: "board" }`, `boardLines(report, leadText): string[]`.

- [ ] **Step 1: Export the workspace pieces**

In `workspace.tsx` prefix `function MemberConclusions` (line 332), `function mergeAgents` (351) and `function TurnWork` (413) with `export`. No other change.

- [ ] **Step 2: `panel.tsx`**

```tsx
// The readable side of the office: what the controlled agent is looking at. At a desk, that agent's workspace (the
// same components as the workspace panel); at the whiteboard, the report, or the lead's answer when reporting is off.
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import type { ConversationReport } from "../api";
import { Markdown } from "../blocks";
import type { AgentWork, Turn } from "../chat";
import { T } from "../i18n";
import { MemberConclusions, TurnWork } from "../workspace";
import { wrapLines } from "./textures";

export type Panel = { kind: "desk"; id: string } | { kind: "board" };

/** The lines the 3D whiteboard shows: the report's title and abstract, else the lead's answer, else a note. */
export function boardLines(report: ConversationReport | null, leadText: string): string[] {
  if (report) {
    const lines = [report.title, ...wrapLines(report.abstract, 38, 6)].filter(Boolean);
    if (lines.length) return lines;
  }
  const lines = wrapLines(leadText, 38, 7);
  return lines.length ? lines : [T("office.nothingYet")];
}

export function OfficePanel({
  panel,
  turn,
  member,
  memberName,
  report,
  at,
  total,
  onGo,
  onClose,
  onOpenObject,
}: {
  panel: Panel;
  turn: Turn | undefined;
  member: AgentWork | undefined;
  memberName: string;
  /** The report to show on the board, already null when reporting is off. */
  report: ConversationReport | null;
  at: number;
  total: number;
  onGo(i: number): void;
  onClose(): void;
  onOpenObject(name: string, blockId?: string | null): void;
}) {
  const title = panel.kind === "board" ? T("office.results") : memberName;
  return (
    <aside className="office-panel" aria-label={title}>
      <header className="ws-head">
        <div className="ws-head-row">
          <span className="ws-title">{title}</span>
          <span className="spacer" />
          <button className="ghost" title={T("office.prevTurn")} disabled={at <= 0} onClick={() => onGo(at - 1)}>
            <ChevronLeft size={15} />
          </button>
          <span className="ws-pos muted small">{total ? `${at + 1}/${total}` : "—"}</span>
          <button className="ghost" title={T("office.nextTurn")} disabled={at >= total - 1} onClick={() => onGo(at + 1)}>
            <ChevronRight size={15} />
          </button>
          <button className="ghost" title={T("office.closePanel")} onClick={onClose}>
            <X size={15} />
          </button>
        </div>
        {panel.kind === "desk" && turn && (
          <p className="ws-question" title={member ? member.task : turn.question}>
            {member && <span className="ws-label">{T("team.task")}</span>}
            {member ? member.task : turn.question}
          </p>
        )}
      </header>
      <div className="ws-body">
        {panel.kind === "board" ? (
          <Board report={report} leadText={turn?.text ?? ""} />
        ) : member ? (
          <>
            <MemberConclusions agent={member} />
            <TurnWork turn={member} onOpenObject={onOpenObject} />
          </>
        ) : turn ? (
          <TurnWork turn={turn} onOpenObject={onOpenObject} />
        ) : (
          <p className="muted small">{T("office.noReport")}</p>
        )}
      </div>
    </aside>
  );
}

function Board({ report, leadText }: { report: ConversationReport | null; leadText: string }) {
  if (report)
    return (
      <div className="ws-card conclusion office-report">
        <h3>{report.title}</h3>
        <Markdown text={report.abstract} />
        {report.sections.map((s) => (
          <section key={s.heading}>
            <h4>{s.heading}</h4>
            <Markdown text={s.body_md} />
          </section>
        ))}
        <h4>{T("workspace.provisionalConclusion")}</h4>
        <Markdown text={report.conclusions_md} />
      </div>
    );
  if (!leadText) return <p className="muted small">{T("office.noReport")}</p>;
  return (
    <div className="ws-card conclusion">
      <span className="ws-label">{T("office.leadAnswer")}</span>
      <Markdown text={leadText} />
    </div>
  );
}
```

- [ ] **Step 3: Panel state, whiteboard chip, writing animation in `Office.tsx`**

Imports: `import { mergeAgents } from "../workspace";`, `import { boardLines, OfficePanel, type Panel } from "./panel";`, add `BOARD_SPOT`, `READING_ZONE`, `same` to the grid import, `ClipboardList` to the lucide import, `useRef` to the React import.

Inside `Office`, replace the Escape effect from Task 4 with the block below and add the rest after the `live` memo:

```tsx
  // The turn the panels show (the latest unless one was picked), with the members' work folded up to it.
  const index = props.turnKey ? props.turns.findIndex((t) => t.key === props.turnKey) : props.turns.length - 1;
  const at = index < 0 ? props.turns.length - 1 : index;
  const turn = props.turns[at];
  const merged = useMemo(() => mergeAgents(props.turns.slice(0, at + 1)), [props.turns, at]);
  const go = (i: number) => props.onSelectTurn(i >= props.turns.length - 1 ? null : props.turns[i].key);

  // What the controlled agent is looking at: it opens on arriving at a chair or at the whiteboard, and closes on
  // Escape or on walking away.
  const [panel, setPanel] = useState<Panel | null>(null);
  const sel = state.selected ? state.positions[state.selected] : undefined;
  const selTile = sel ? `${sel.tile.x},${sel.tile.z}` : "";
  useEffect(() => {
    if (!sel) return;
    const desk = desks.find((d) => same(d.chair, sel.tile));
    if (desk) setPanel({ kind: "desk", id: desk.id });
    else if (READING_ZONE.some((t) => same(t, sel.tile))) setPanel({ kind: "board" });
    else setPanel(null);
    // The tile is the trigger; `sel` itself changes on every store update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selTile, desks]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (panel) setPanel(null);
      else onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, panel]);

  // The whiteboard: the report (reporting on) or the lead's answer of the shown turn. When it changes while the
  // office follows the latest turn, the lead walks over and writes it; browsing old turns shows it at once.
  const report = props.reporting.enabled ? props.reporting.report : null;
  const lines = useMemo(() => boardLines(report, turn?.text ?? ""), [report, turn?.text]);
  const text = lines.join("\n");
  const [chars, setChars] = useState(Infinity);
  const firstBoard = useRef(true);
  useEffect(() => {
    const nothing = lines.length === 1 && lines[0] === T("office.nothingYet");
    if (firstBoard.current || props.turnKey !== null || nothing) {
      firstBoard.current = false;
      setChars(Infinity);
      return;
    }
    const lead = members[0]?.id;
    if (lead) office.walkTo(lead, BOARD_SPOT);
    setChars(0);
    const total = text.length;
    const timer = setInterval(() => setChars((c) => (c + 1 >= total ? Infinity : c + 1)), 40);
    const back = setTimeout(() => lead && office.walkHome(lead), total * 40 + 1500);
    return () => {
      clearInterval(timer);
      clearTimeout(back);
    };
    // Only a change of the text itself starts a writing session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);
```

Replace `<Whiteboard lines={[T("office.nothingYet")]} chars={Infinity} />` with `<Whiteboard lines={lines} chars={chars} />`.

In the bar, after the member chips:

```tsx
        <button className={`office-chip ${panel?.kind === "board" ? "on" : ""}`} onClick={() => setPanel({ kind: "board" })}>
          <ClipboardList size={13} /> <span className="office-chip-name">{T("office.whiteboard")}</span>
        </button>
```

After the bar, render the panel:

```tsx
      {panel && (
        <OfficePanel
          panel={panel}
          turn={turn}
          member={panel.kind === "desk" && panel.id !== LEAD_ID ? merged.find((a) => a.id === panel.id) : undefined}
          memberName={panel.kind === "desk" ? `${members.find((m) => m.id === panel.id)?.emoji ?? ""} ${members.find((m) => m.id === panel.id)?.name ?? ""}`.trim() || "—" : ""}
          report={report}
          at={at}
          total={props.turns.length}
          onGo={go}
          onClose={() => setPanel(null)}
          onOpenObject={props.onOpenObject}
        />
      )}
```

- [ ] **Step 4: CSS (append)**

```css
.office-panel { position: absolute; top: 52px; right: 14px; bottom: 76px; width: 420px; max-width: calc(100% - 28px); display: flex; flex-direction: column; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); animation: enter 0.25s ease-out; }
.office-panel .ws-head { padding: 10px 12px 6px; border-bottom: 1px solid var(--line); }
.office-report h3 { margin: 0 0 6px; font-family: var(--serif); font-size: 17px; }
.office-report h4 { margin: 12px 0 4px; font-size: 12.5px; }
```

- [ ] **Step 5: Typecheck, build, look**

Run: `npm run typecheck && npm run build`, then `npm run dev`. Expected:
- Walk the selected agent onto a member's chair → the panel opens with that member's task, conclusions and work; onto the lead's chair → the lead's work of the shown turn; ‹ › change the turn (and the workspace follows when reopened).
- Walk in front of the whiteboard, or click the **Whiteboard** chip → the report (Reporting on) or the lead's answer; with no turns, the "nothing to show" note.
- Escape closes the panel, a second Escape closes the office.
- With the office open and following the latest turn, finish a request → the lead walks to the board, the text appears character by character, the lead walks back and sits.
- Click an object link in the panel → the office closes and the block flashes in the notebook.

- [ ] **Step 6: Commit (when asked)**

```bash
git add apps/desktop/src/office apps/desktop/src/workspace.tsx apps/desktop/src/styles.css
git commit -m "feat(office): side panel with workspaces and results; whiteboard writing"
```

---

### Task 8: Final checks

**Files:** none new.

- [ ] **Step 1: Full verification**

Run, from `apps/desktop`: `npm test && npm run typecheck && npm run build`. Expected: all green.

- [ ] **Step 2: Manual checklist (record the outcome in the final report)**

1. Team off → only the lead; team on with 1, 5 and 10 enabled members → that many desks, none overlapping, every chair reachable.
2. Member with empty name/emoji → "—" label, 🧑‍💼 face.
3. Remove a member in Agent team while the office is open → it vanishes from the room.
4. Member failing (e.g. a task that errors) → 💥 bubble and smoke over its monitor.
5. Resize the window → the room keeps fitting; wheel zoom clamps.
6. Agents never walk through furniture, each other or the cat; a blocked path re-plans.

- [ ] **Step 3: Commit (when asked)**

Nothing new to commit unless fixes were needed; then `git commit -m "fix(office): …"`.
