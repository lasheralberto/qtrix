# Agent office: a 3D view of the team at work

**Date:** 2026-10-15 · **Status:** approved design, pending implementation plan

## Goal

A full-screen 3D office, in a playful low-poly "indie" style, where the user sees the agent team working. Each
agent has a desk whose monitor stands for its workspace; a whiteboard in the middle of the back wall stands for the
team's results. The user picks an agent in a bottom bar and walks it around the room with the arrow keys, Habbo
Hotel style; standing at a desk or in front of the whiteboard opens the real content (workspace or report) in an
HTML side panel.

It is a second view over state the app already has (`settings.team`, the chat `turns`, `reporting`); it adds no
backend work and no persistence.

## Decisions taken with the user

- Opens full screen, from a new button in the rail foot placed directly above "Agent team".
- The whiteboard shows the Reporting report when reporting is on; otherwise the lead's answer of the shown turn.
- Navigation is Habbo-like: select an agent in the bottom bar, move it tile by tile with the arrow keys. No free
  orbit; fixed isometric camera with wheel zoom.
- Stack: `three` + `@react-three/fiber` v9 + `@react-three/drei`, everything procedural (no model or texture
  assets).

## Scope

In:

- Room, furniture, whiteboard, avatars, idle/working animations, bottom bar, keyboard movement, side panel.
- One desk per enabled team member (the lead always), up to `MAX_TEAM`.
- Tests for the pure grid logic; typecheck and build green.

Out (YAGNI):

- Saving positions across app restarts; multiplayer; sound; editing the team from the office; mobile/touch.
- Readable text rendered inside the 3D monitors (the side panel carries the readable content).

## Room

- Grid of 12 × 9 tiles (x: 0..11, z: 0..8), tile = 1 unit. Floor and walls on the north (z = 0) and west (x = 0)
  sides only, so the isometric camera looks in over the missing walls.
- Camera: orthographic, isometric (rotated 45° around y, pitched ~35°), fixed target at the room centre; mouse
  wheel zooms between 0.7× and 1.6×. No orbit.
- Whiteboard: on the north wall, centred (tiles x = 4..7, z = 0). The tile row z = 1 in front of it is the "reading"
  zone.
- Desks: one per agent, placed by a layout function (see `grid.ts`): along the west wall first (facing east), then
  along the south edge (facing north), leaving the centre free to walk. Each desk occupies 2 × 1 tiles plus a chair
  tile in front; the chair tile is where an agent sits and where the controlled agent "reads" that desk. The lead's
  desk is first and carries a plant.
- Props that cost nothing: a coffee machine in the north-east corner (2 tiles), a cat that wanders over free
  tiles one step every few seconds, a poster on the west wall.
- Look: flat pastel palette, `MeshToonMaterial`, one directional light with soft shadows plus a hemisphere light,
  slightly rounded boxes. Colours follow the app's light/dark theme only for the overlay chrome; the room itself
  stays bright.

## Avatars

- Capsule body in a colour derived from the agent id (stable hash into a pastel list), box head whose front face
  is a canvas texture with the agent's emoji (fallback 🧑‍💼), a floating name label (drei `Html`, non-interactive).
- One avatar per enabled member of `settings.team.agents`; with the team disabled, only the lead. Members that are
  not enabled are not drawn.
- Driven by real state (per agent, from `mergeAgents(turns)` for members and the turn itself for the lead):
  - `running`: sits at its desk, arms wiggle ("typing"), a speech bubble shows "…" and the current `status`.
  - finished with `ok !== false`: bubble ✓ for a few seconds after it stops.
  - `ok === false` or `error`: bubble 💥 and a small puff of smoke over the monitor.
  - idle: gentle bob; every ~20 s one idle agent walks to the coffee machine and back (only if not controlled).
- When a turn starts, every agent that is not already at its chair walks back to it (BFS path on the grid, one
  tile per 180 ms). The user may still move an agent; it simply walks off again.
- Walking animation: a small hop per tile and a turn toward the direction of travel.

## Whiteboard

- 3D: a white board with a dark frame. Its face is a canvas texture showing, in a marker-like font, the report
  title and the first ~6 lines of the abstract (or the first lines of the lead's answer). When the content changes,
  the lead walks to the reading zone and the text is "written" line by line (~40 ms per character), then it walks
  back. With nothing to show, the board reads "(nothing yet)" from i18n.
- The readable version is in the side panel (see Interaction).

## Monitors

- 3D: a screen plane with 6–8 coloured bars whose heights pulse while the agent is running and stay still
  otherwise; dark when the agent never worked. Purely atmospheric.

## Interaction

- Bottom bar: one chip per drawn agent (emoji + name; a spinner while running) plus a "Whiteboard" chip. Clicking a
  chip selects that agent (a ring under its feet) and gives it the keyboard. Clicking the whiteboard chip opens
  the results panel without moving anyone.
- Arrow keys move the selected agent one tile in that direction; holding a key repeats at the walk speed. Furniture,
  walls, the cat and other agents block. Enter when standing on its own chair tile sits it down (it also sits
  automatically when a path brings it home).
- Side panel (right, ~420 px, over the canvas): opens when the controlled agent stands on a chair tile, showing that
  desk's agent workspace: `MemberConclusions` + `TurnWork` for members, `TurnWork` of the shown turn for the lead,
  both exported from `workspace.tsx`. In the reading zone in front of the whiteboard (or via its chip) it shows the
  report: `ReportSection`s and conclusions rendered with the existing `Markdown`, or the lead's `text`. Object links
  use the `onOpenObject` the workspace already has.
- The panel has ‹ › arrows to move between turns, mirroring the workspace; the shown turn is shared with the
  workspace (`wsTurn`).
- Escape closes the panel; with no panel open it closes the office. The office does not pause the chat: turns keep
  streaming and the avatars react live.
- Positions and selection are kept in a module-level store so reopening the office finds everyone where they
  were. Closing the app forgets them.

## Code

```
apps/desktop/src/office/
  Office.tsx     full-screen overlay: <Canvas>, bottom bar, side panel, keyboard wiring
  room.tsx       floor, walls, desks, whiteboard, coffee machine, poster, cat
  avatar.tsx     one agent: meshes, bubble, label, animation from its state
  grid.ts        pure: tile map, desk layout for N agents, walkability, BFS path, neighbours
  state.ts       store + hook: positions, selection, walking queues, who sits where
  textures.ts    canvas textures: emoji face, whiteboard text, monitor bars
apps/desktop/src/i18n/en/office.ts
apps/desktop/src/styles.css        (.office, .office-bar, .office-panel)
```

- `App.tsx`: `showOffice` state; button `<Building2 /> {T("office.button")}` above the team button; renders
  `<Office team={settings.team} turns={turns} turnKey={wsTurn} onSelectTurn={setWsTurn} reporting={reporting}
  onOpenObject={openObject} onClose=… />`.
- `workspace.tsx`: export `TurnWork`, `MemberConclusions`, `mergeAgents` (no behaviour change).
- Dependencies: `three`, `@react-three/fiber`, `@react-three/drei`, `@types/three`; `vitest` as a dev dependency
  with a `test` script in `apps/desktop/package.json` (the desktop app has no test runner today).

## Error handling

- WebGL unavailable (old GPU, headless): the overlay shows an i18n message with a close button instead of a
  blank canvas (`Canvas` `onCreated` failure / `WebGLRenderingContext` probe before mounting).
- A team member removed from settings while the office is open disappears from the room; its stored position is
  dropped.
- Agents whose desk cannot be placed (more than the layout supports) are not drawn; the layout supports
  `MAX_TEAM`.

## Testing

- `grid.test.ts` (vitest): desk layout for 1, 5 and 10 agents produces non-overlapping desks inside the room with
  a walkable chair tile and a path to the whiteboard zone; BFS finds the shortest path around a desk and returns
  none when boxed in; walkability respects walls, furniture and occupied tiles.
- `state.test.ts`: selecting, moving into a blocked tile is a no-op, walking home drains the queue.
- `npm run typecheck`, `npm run build`.
- Manual: open the office with the team on and off, run a request, watch avatars return to desks and the
  whiteboard update, read a member's workspace from its chair, read the report, Escape twice.
