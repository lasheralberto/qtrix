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

/** The tag over an agent's head, drawn on a canvas (a DOM overlay per agent would be a React root each): its name
 * in a pill and, above it, a speech bubble or a small hint. */
export function labelBoard() {
  const W = 512;
  const H = 192;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;
  const texture = new CanvasTexture(c);
  texture.colorSpace = SRGBColorSpace;
  const pill = (text: string, y: number, h: number, font: string, fill: string, ink: string, border?: string) => {
    g.font = font;
    const w = Math.min(W - 8, g.measureText(text).width + h);
    const x = (W - w) / 2;
    g.beginPath();
    g.roundRect(x, y, w, h, h / 2.6);
    g.fillStyle = fill;
    g.fill();
    if (border) {
      g.lineWidth = 4;
      g.strokeStyle = border;
      g.stroke();
    }
    g.fillStyle = ink;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, W / 2, y + h / 2 + 1, w - h / 2);
  };
  const draw = (name: string, bubble: string | null, bad: boolean, hint: string | null) => {
    g.clearRect(0, 0, W, H);
    if (bubble) pill(bubble.length > 30 ? bubble.slice(0, 29) + "…" : bubble, 14, 70, `32px 'Segoe UI', system-ui, ${EMOJI_FONT}`, bad ? "#fbece5" : "#ffffff", "#2b3a67", bad ? "#c2562d" : "#2b3a67");
    else if (hint) pill(hint, 44, 40, "22px sans-serif", "rgba(255,255,255,0.75)", "#4b5567");
    pill(name, 124, 50, "bold 28px sans-serif", "rgba(255,255,255,0.9)", "#2b3a67");
    texture.needsUpdate = true;
  };
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
