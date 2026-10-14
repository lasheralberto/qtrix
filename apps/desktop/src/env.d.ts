/// <reference types="vite/client" />

interface QtrixBridge {
  engineInfo(): Promise<{ baseUrl: string; token: string }>;
  openDataFiles(): Promise<string[]>;
  pathForFile(file: File): string;
  openPath(path: string): Promise<string>;
  getHome(): Promise<{ home: string; defaultHome: string }>;
  chooseFolder(): Promise<string | null>;
  setHome(path: string, move: boolean): Promise<{ ok: boolean; error?: string }>;
  exportPdf(name: string): Promise<{ ok: boolean; path?: string; error?: string }>;
}

interface Window {
  qtrix?: QtrixBridge;
}

declare module "plotly.js-dist-min" {
  const Plotly: {
    react(el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown): Promise<unknown>;
    purge(el: HTMLElement): void;
    Plots: { resize(el: HTMLElement): void };
  };
  export default Plotly;
}
