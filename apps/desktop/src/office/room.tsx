// The furniture of the office: floor, walls, whiteboard, desks, the coffee machine, the poster and the cat. Every
// piece is boxes and planes in flat toon colours; positions come from grid.ts tiles.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
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
  // Only the first position goes through props: a new array on every render would make it jump instead of walk.
  const [start] = useState(target);
  useFrame((_, dt) => {
    if (!g.current) return;
    g.current.position.x += (target[0] - g.current.position.x) * Math.min(1, dt * 4);
    g.current.position.z += (target[2] - g.current.position.z) * Math.min(1, dt * 4);
  });
  return (
    <group ref={g} position={start}>
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
