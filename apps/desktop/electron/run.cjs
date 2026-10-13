// Launches Electron with ELECTRON_RUN_AS_NODE removed. VS Code / Claude Code terminals set it to 1,
// which makes electron.exe behave as plain Node (require("electron") has no ipcMain/app) and crash.
const { spawn } = require("node:child_process");
const electron = require("electron");

const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;

const child = spawn(electron, process.argv.slice(2).length ? process.argv.slice(2) : ["."], { stdio: "inherit", env });
child.on("exit", (code) => process.exit(code ?? 0));
