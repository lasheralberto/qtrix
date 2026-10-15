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
