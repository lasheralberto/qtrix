import { defineConfig } from "vitest/config";

// Unit tests of the pure modules only (grid, office state); the UI is checked by typecheck, build and by hand.
export default defineConfig({ test: { include: ["src/**/*.test.ts"] } });
