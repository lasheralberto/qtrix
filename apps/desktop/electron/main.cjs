// Electron main process: starts the local Python engine as a sidecar and opens the workbench window.
// The engine listens on 127.0.0.1 only and requires a random per-session token.
const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn, execFile } = require("node:child_process");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const readline = require("node:readline");

const isDev = Boolean(process.env.QTRIX_DEV_URL);
const token = crypto.randomBytes(24).toString("hex");
// Everything qtrix stores lives in one folder; location.json (in userData) remembers where it is.
const HOME_ENTRIES = ["settings.json", "LEEME.txt", "data", "memory", "models", "runtime", "logs", "uploads", "exports"];
let engine = null;
let enginePort = null;
let mainWindow = null;

const locationFile = () => path.join(app.getPath("userData"), "location.json");
const defaultHome = () => path.join(os.homedir(), "qtrix");

function qtrixHome() {
  try {
    const home = JSON.parse(fs.readFileSync(locationFile(), "utf-8")).home;
    if (home) return home;
  } catch {
    /* first run: no pointer yet */
  }
  return defaultHome();
}

function engineCommand() {
  if (process.env.QTRIX_ENGINE_CMD) {
    const [cmd, ...args] = process.env.QTRIX_ENGINE_CMD.split(" ");
    return { cmd, args, cwd: undefined };
  }
  const bundled = path.join(process.resourcesPath || "", "engine", process.platform === "win32" ? "qtrix-engine.exe" : "qtrix-engine");
  if (!isDev && fs.existsSync(bundled)) {
    return { cmd: bundled, args: ["--port", "0"], cwd: undefined };
  }
  // Development: run the engine from the monorepo with uv.
  const engineDir = path.resolve(__dirname, "..", "..", "..", "engine");
  return { cmd: "uv", args: ["run", "--project", engineDir, "python", "-m", "qtrix_engine", "--port", "0"], cwd: engineDir };
}

function startEngine() {
  enginePort = null;
  return new Promise((resolve, reject) => {
    const { cmd, args, cwd } = engineCommand();
    // The engine resolves the qtrix folder itself (same pointer file) and logs to <qtrix>/logs/engine.log.
    engine = spawn(cmd, args, {
      cwd,
      env: {
        ...process.env,
        QTRIX_TOKEN: token,
        QTRIX_PARENT_PID: String(process.pid), // the engine exits by itself if this process dies
        PYTHONIOENCODING: "utf-8",
        PYTHONUNBUFFERED: "1",
      },
      windowsHide: true,
    });
    let stderrTail = "";
    engine.stderr.on("data", (chunk) => {
      stderrTail = (stderrTail + chunk.toString()).slice(-1500);
    });
    const timer = setTimeout(() => reject(new Error("The qtrix engine did not start within 120 s")), 120_000);
    readline.createInterface({ input: engine.stdout }).on("line", (line) => {
      try {
        const msg = JSON.parse(line);
        if (msg.event === "ready") {
          clearTimeout(timer);
          enginePort = msg.port;
          resolve(msg.port);
        }
      } catch {
        /* not a control line */
      }
    });
    engine.on("exit", (code) => {
      clearTimeout(timer);
      if (enginePort === null) {
        const where = path.join(qtrixHome(), "logs", "engine.log");
        reject(new Error(`The qtrix engine exited with code ${code}\n\n${stderrTail.trim()}\n\nLog: ${where}`));
      }
      engine = null;
    });
  });
}

/** Stop the engine (and its llama-server) and resolve once the process is gone. */
function stopEngine() {
  return new Promise((resolve) => {
    if (!engine) return resolve();
    const proc = engine;
    engine = null;
    const done = setTimeout(resolve, 10_000);
    proc.once("exit", () => {
      clearTimeout(done);
      resolve();
    });
    if (process.platform === "win32") {
      // uv spawns python as a child: kill the whole tree.
      execFile("taskkill", ["/pid", String(proc.pid), "/T", "/F"], () => {});
    } else {
      proc.kill("SIGTERM");
    }
  });
}

function isInside(child, parent) {
  const rel = path.relative(path.resolve(parent), path.resolve(child));
  return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
}

function moveEntry(src, dst) {
  try {
    fs.renameSync(src, dst);
  } catch (err) {
    if (err.code !== "EXDEV") throw err;
    fs.cpSync(src, dst, { recursive: true }); // different drive: copy, then delete
    fs.rmSync(src, { recursive: true, force: true });
  }
}

function moveHome(from, to) {
  if (isInside(to, from)) throw new Error("La nueva carpeta no puede estar dentro de la actual");
  if (fs.existsSync(path.join(to, "data", "workspace.duckdb"))) {
    throw new Error("La carpeta destino ya contiene datos de qtrix. Elige 'Usar sin mover' para abrirla tal cual.");
  }
  fs.mkdirSync(to, { recursive: true });
  for (const name of HOME_ENTRIES) {
    const src = path.join(from, name);
    const dst = path.join(to, name);
    if (!fs.existsSync(src)) continue;
    if (fs.existsSync(dst)) {
      if (fs.statSync(dst).isDirectory() && fs.readdirSync(dst).length === 0) fs.rmdirSync(dst);
      else if (name === "LEEME.txt") continue;
      else throw new Error(`Ya existe ${dst}`);
    }
    moveEntry(src, dst);
  }
  try {
    fs.rmdirSync(from); // only if nothing else was left behind
  } catch {
    /* the user kept other files there */
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1560,
    height: 960,
    minWidth: 1100,
    minHeight: 700,
    title: "qtrix",
    backgroundColor: "#f6f5f1",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("https://")) shell.openExternal(url);
    return { action: "deny" };
  });
  if (isDev) mainWindow.loadURL(process.env.QTRIX_DEV_URL);
  else mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
}

ipcMain.handle("engine:info", () => ({ baseUrl: `http://127.0.0.1:${enginePort}`, token }));

ipcMain.handle("dialog:open-data", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Importar datos",
    properties: ["openFile", "multiSelections"],
    filters: [
      { name: "Datos", extensions: ["csv", "tsv", "txt", "parquet", "pq", "xlsx", "xls", "xlsm"] },
      { name: "Todos", extensions: ["*"] },
    ],
  });
  return result.canceled ? [] : result.filePaths;
});

ipcMain.handle("dialog:choose-folder", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Carpeta de qtrix",
    defaultPath: qtrixHome(),
    properties: ["openDirectory", "createDirectory"],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("shell:open-path", (_event, target) => {
  if (!isInside(target, qtrixHome())) return "Ruta fuera de la carpeta de qtrix";
  return shell.openPath(target);
});

ipcMain.handle("home:get", () => ({ home: qtrixHome(), defaultHome: defaultHome() }));

ipcMain.handle("home:set", async (_event, { path: requested, move }) => {
  const current = path.resolve(qtrixHome());
  const target = path.resolve(requested);
  if (target === current) return { ok: true };
  try {
    await stopEngine();
    if (move) moveHome(current, target);
    fs.mkdirSync(target, { recursive: true });
    fs.mkdirSync(path.dirname(locationFile()), { recursive: true });
    fs.writeFileSync(locationFile(), JSON.stringify({ home: target }, null, 2));
    await startEngine();
    mainWindow?.reload();
    return { ok: true };
  } catch (err) {
    // Whatever failed, bring the engine back on the folder the pointer now designates.
    if (!engine) await startEngine().catch(() => {});
    mainWindow?.reload();
    return { ok: false, error: String(err.message || err) };
  }
});

// Two instances would start two engines fighting over the same workspace database.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) app.quit();
app.on("second-instance", () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.focus();
});

(gotLock ? app.whenReady() : Promise.resolve()).then(async () => {
  if (!gotLock) return;
  try {
    await startEngine();
  } catch (err) {
    dialog.showErrorBox("qtrix", String(err.message || err));
    app.quit();
    return;
  }
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
app.on("before-quit", () => {
  stopEngine();
});
process.on("exit", () => {
  stopEngine();
});
