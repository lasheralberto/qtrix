import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Electron loads the built files from disk, so assets must use relative paths.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  build: { outDir: "dist", chunkSizeWarningLimit: 6000 },
});
