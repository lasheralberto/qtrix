// Minimal, explicit bridge between the sandboxed renderer and the main process.
const { contextBridge, ipcRenderer, webUtils } = require("electron");

contextBridge.exposeInMainWorld("qtrix", {
  engineInfo: () => ipcRenderer.invoke("engine:info"),
  openDataFiles: () => ipcRenderer.invoke("dialog:open-data"),
  pathForFile: (file) => webUtils.getPathForFile(file),
  openPath: (p) => ipcRenderer.invoke("shell:open-path", p),
  getHome: () => ipcRenderer.invoke("home:get"),
  chooseFolder: () => ipcRenderer.invoke("dialog:choose-folder"),
  setHome: (p, move) => ipcRenderer.invoke("home:set", { path: p, move }),
});
