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
