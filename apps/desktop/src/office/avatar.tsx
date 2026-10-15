// One agent in the office: a pastel capsule with an emoji face. It glides toward the tile the store gives it,
// hopping as it walks, takes the next tile of its walk when it gets there, and shows what the agent does in a bubble.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Group } from "three";
import { T } from "../i18n";
import { world } from "./grid";
import { YAW } from "./room";
import { office, type AgentPos } from "./state";
import { emojiTexture, labelBoard, pastel } from "./textures";

/** What the agent is doing right now, from the chat state. */
export interface AgentLive {
  running: boolean;
  status?: string;
  failed: boolean;
}

const WALK_SPEED = 5.5; // tiles per second ≈ 180 ms per tile
const FALLBACK_FACE = "🧑‍💼";

/** The shortest turn from angle `a` to angle `b`. */
const turn = (a: number, b: number) => ((((b - a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;

export function Avatar({ id, name, emoji, pos, selected, live }: { id: string; name: string; emoji: string; pos: AgentPos; selected: boolean; live: AgentLive }) {
  const root = useRef<Group>(null);
  const body = useRef<Group>(null);
  const arms = useRef<Group>(null);
  const face = useMemo(() => emojiTexture(emoji || FALLBACK_FACE), [emoji]);
  const colour = useMemo(() => pastel(id), [id]);
  // Only the first position and heading go through props: the frame loop moves the avatar from then on.
  const [start] = useState(() => {
    const [x, z] = world(pos.tile);
    return { position: [x, 0, z] as [number, number, number], rotation: [0, YAW[pos.facing], 0] as [number, number, number] };
  });

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
    // The store, not the props of the last render: a tick earlier in this frame already moved the agent on.
    const p = office.get().positions[id];
    if (!g || !p) return;
    const [tx, tz] = world(p.tile);
    const dx = tx - g.position.x;
    const dz = tz - g.position.z;
    const dist = Math.hypot(dx, dz);
    if (dist > 0.01) {
      const stepLen = Math.min(dist, WALK_SPEED * dt);
      g.position.x += (dx / dist) * stepLen;
      g.position.z += (dz / dist) * stepLen;
      g.position.y = Math.sin(Math.min(1, dist) * Math.PI) * 0.18;
    } else {
      g.position.set(tx, 0, tz);
      if (p.path.length) office.tick(id);
      g.position.y = p.seated ? -0.12 : Math.sin(s.clock.elapsedTime * 2 + tx) * 0.02;
    }
    g.rotation.y += turn(g.rotation.y, YAW[p.facing]) * Math.min(1, dt * 10);
    if (arms.current) arms.current.rotation.x = live.running && p.seated ? Math.sin(s.clock.elapsedTime * 14) * 0.25 - 0.9 : 0;
    if (body.current) body.current.rotation.z = p.seated && !live.running ? Math.sin(s.clock.elapsedTime * 1.3) * 0.04 : 0;
  });

  const bubble = live.running ? `${(live.status ?? "").replace(/…$/, "")}…` : live.failed ? "💥" : done ? "✓" : null;

  return (
    <group ref={root} position={start.position} rotation={start.rotation}>
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
      <Label name={name || "—"} bubble={bubble} bad={live.failed} hint={!bubble && pos.seated && selected ? T("office.seated") : null} />
    </group>
  );
}

/** Name and speech bubble over the head, always facing the camera and drawn over the furniture. */
function Label({ name, bubble, bad, hint }: { name: string; bubble: string | null; bad: boolean; hint: string | null }) {
  const board = useMemo(labelBoard, []);
  useEffect(() => board.draw(name, bubble, bad, hint), [board, name, bubble, bad, hint]);
  useEffect(() => () => board.texture.dispose(), [board]);
  return (
    <sprite position={[0, 1.85, 0]} scale={[3.2, 1.2, 1]} renderOrder={10}>
      <spriteMaterial map={board.texture} transparent depthTest={false} />
    </sprite>
  );
}
