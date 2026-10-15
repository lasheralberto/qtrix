// Who is where in the office: a small store outside React (positions survive closing the office) read through
// useOffice(). Movement is in whole tiles; the avatars animate between them and call tick() on arrival.
import { useSyncExternalStore } from "react";
import {
  BOARD_SPOT,
  blockedTiles,
  COFFEE_SPOT,
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
  /** The selected agent once the user has moved it: the office leaves it alone (no walks home, no errands). */
  steered: string | null;
  cat: Tile;
}

const CAT_START: Tile = { x: 6, z: 4 };
/** The fastest a held arrow key moves an agent: one tile per step of the walk (see WALK_SPEED in avatar.tsx). */
const STEP_MS = 180;
const empty = (): OfficeState => ({ positions: {}, selected: null, steered: null, cat: CAT_START });
let state: OfficeState = empty();
let desks: Desk[] = [];
let furniture = new Set<string>();
let clock: () => number = () => Date.now();
const lastMove = new Map<string, number>();
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
    for (const d of next) {
      const p = state.positions[d.id];
      // Newcomers start seated; so does an agent seated at a chair that is no longer its own (the layout moved
      // with the team) or one standing where furniture now stands.
      const misplaced = p && ((p.seated && !same(p.tile, d.chair)) || furniture.has(key(p.tile)));
      positions[d.id] = !p || misplaced ? seatedAt(d) : p;
    }
    const selected = state.selected && positions[state.selected] ? state.selected : (next[0]?.id ?? null);
    const steered = state.steered === selected ? state.steered : null;
    const cat = furniture.has(key(state.cat)) ? CAT_START : state.cat;
    set({ ...state, positions, selected, steered, cat });
  },
  select(id: string | null) {
    set({ ...state, selected: id, steered: null });
  },
  /** One tile in `dir` if it is free; the agent stands up and drops any walk. Returns whether it moved. Steps
   * closer together than the walk (a held key repeats faster) are dropped, so the avatar is never left behind. */
  move(id: string, dir: Dir): boolean {
    const p = state.positions[id];
    if (!p) return false;
    const now = clock();
    if (now - (lastMove.get(id) ?? -Infinity) < STEP_MS) return false;
    lastMove.set(id, now);
    const to = step(p.tile, dir);
    const free = walkable(to, blockedFor(id));
    set({
      ...state,
      steered: id,
      positions: { ...state.positions, [id]: { ...p, tile: free ? to : p.tile, facing: dir, seated: false, path: [], home: false } },
    });
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
  /** Takes the next tile of the walk (the avatar reached the current one). A tile taken meanwhile re-plans; with
   * no way through for now (someone stands on the goal) the walk waits and tries again on the next tick. */
  tick(id: string) {
    const p = state.positions[id];
    if (!p || !p.path.length) return;
    const [next, ...rest] = p.path;
    const blocked = blockedFor(id);
    if (!walkable(next, blocked)) {
      const path = findPath(p.tile, p.path[p.path.length - 1], blocked);
      if (path?.length) patch(id, { path });
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
  /** The cat strolls one tile in a random free direction, never onto a chair or a spot the agents walk to. */
  catStep() {
    const blocked = blockedFor(null);
    for (const t of [...desks.map((d) => d.chair), BOARD_SPOT, COFFEE_SPOT]) blocked.add(key(t));
    const options = (["n", "s", "e", "w"] as Dir[]).map((d) => step(state.cat, d)).filter((t) => walkable(t, blocked));
    if (options.length) set({ ...state, cat: options[Math.floor(Math.random() * options.length)] });
  },
  /** Tests only: a fresh office, on a clock that moves a second per reading (so moves are never throttled). */
  reset() {
    desks = [];
    furniture = new Set();
    lastMove.clear();
    let t = 0;
    clock = () => (t += 1000);
    set(empty());
  },
  /** Tests only. */
  setClock(c: () => number) {
    clock = c;
  },
};

export const useOffice = () => useSyncExternalStore(office.subscribe, office.get);
