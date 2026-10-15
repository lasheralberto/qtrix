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
