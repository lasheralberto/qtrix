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

describe("review fixes", () => {
  it("I-2: selecting is not steering; moving is", () => {
    expect(office.get().steered).toBeNull();
    office.move("lead", "e");
    expect(office.get().steered).toBe("lead");
    office.select("a");
    expect(office.get().steered).toBeNull();
  });

  it("I-3: a walk home whose chair is taken for a while waits and then arrives", () => {
    office.walkTo("lead", BOARD_SPOT);
    while (office.get().positions.lead.path.length) office.tick("lead");
    expect(office.walkHome("lead")).toBe(true);
    // someone sits on the lead's chair while it walks
    office.walkTo("a", desks[0].chair);
    for (let i = 0; i < 50 && office.get().positions.a.path.length; i++) office.tick("a");
    for (let i = 0; i < 20; i++) office.tick("lead");
    expect(office.get().positions.lead.home).toBe(true);
    expect(office.get().positions.lead.path.length).toBeGreaterThan(0);
    // the chair frees up: the lead gets there and sits
    office.walkHome("a");
    for (let i = 0; i < 50 && office.get().positions.a.path.length; i++) office.tick("a");
    for (let i = 0; i < 50 && office.get().positions.lead.path.length; i++) office.tick("lead");
    expect(office.get().positions.lead).toMatchObject({ tile: desks[0].chair, seated: true });
  });

  it("I-3: the cat never sits on a chair or a meeting spot", () => {
    for (let i = 0; i < 2000; i++) office.catStep();
    const cat = office.get().cat;
    for (const d of desks) expect(cat).not.toEqual(d.chair);
  });

  it("I-4: moves faster than the walk speed are dropped", () => {
    let now = 1000;
    office.setClock(() => now);
    expect(office.move("lead", "e")).toBe(true);
    now += 50;
    expect(office.move("lead", "e")).toBe(false);
    expect(office.get().positions.lead.tile).toEqual({ x: 2, z: 2 });
    now += 200;
    expect(office.move("lead", "e")).toBe(true);
  });

  it("I-5: a seated agent whose desk moved is re-seated at its new chair", () => {
    office.setDesks(layoutDesks(["lead", "b"])); // b takes a's old slot
    const b = layoutDesks(["lead", "b"])[1];
    expect(office.get().positions.b).toMatchObject({ tile: b.chair, seated: true });
  });

  it("I-5: an agent standing where a new desk now stands is moved to its chair", () => {
    office.setDesks(layoutDesks(["lead"]));
    office.walkTo("lead", { x: 2, z: 8 }); // where the first south desk will stand
    while (office.get().positions.lead.path.length) office.tick("lead");
    expect(office.get().positions.lead.tile).toEqual({ x: 2, z: 8 });
    const big = layoutDesks(["lead", "a", "b", "c"]); // c's desk covers (2,8)-(3,8)
    office.setDesks(big);
    expect(office.get().positions.lead).toMatchObject({ tile: big[0].chair, seated: true });
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
