# qtrix

**Un equipo de econometristas con IA que trabaja en tu ordenador.**

Le hablas en lenguaje natural ("estima un panel con efectos fijos y dime si hay heterocedasticidad") y qtrix
ejecuta estimadores reales (statsmodels, linearmodels, arch…) y construye un panel vivo con tablas de
regresión, contrastes, gráficos y notas. **Tus datos nunca salen de tu máquina** (a menos que elijas un modelo
en la nube).


<p align="center">
  <a href="docs/media/qtrix-demo.mp4">
    <img src="docs/media/qtrix-demo.gif" alt="qtrix en 20 segundos: haces una pregunta, el agente construye tablas, gráficos y contrastes, y el informe se escribe solo" width="760">
  </a>
  <br><sub>▶ <a href="docs/media/qtrix-demo.mp4">Ver el vídeo completo (21 s, con sonido)</a>: la conversación, cómo el agente construye los resultados y el informe final.</sub>
</p>

---

## Instalación en 3 pasos

### 1. Instala los requisitos (una sola vez)

| Herramienta | Para qué | Descarga |
|---|---|---|
| **Node.js 20+** | Interfaz de escritorio | https://nodejs.org |
| **uv** | Gestiona Python 3.12 y las librerías solas | https://docs.astral.sh/uv/getting-started/installation/ |
| **Git** | Descargar el proyecto | https://git-scm.com |

> No necesitas instalar Python: `uv` lo descarga automáticamente.

Comprueba que están listos:

```bash
node --version    # v20 o superior
uv --version
git --version
```

### 2. Descarga e instala qtrix

```bash
git clone https://github.com/lasheralberto/qtrix.git
cd qtrix
npm run setup
```

`npm run setup` instala todo: el motor Python con la pila econométrica y la app de escritorio. Tarda unos
minutos la primera vez.

### 3. Arranca

```bash
npm start
```

Se abre la ventana de qtrix. Listo.

---

## Primeros pasos dentro de la app

1. **Elige un modelo** en **Ajustes**. Tienes dos caminos:
   - **100 % local y gratis:** abre **Modelos locales** (barra izquierda), descarga uno de los modelos
     recomendados y qtrix lo ejecuta solo. También funciona con [Ollama](https://ollama.com) o LM Studio.
   - **En la nube:** pega tu clave de Anthropic, OpenAI, Gemini o Mistral (se guarda en el llavero del sistema).
     Activa **Solo modelos locales** si quieres bloquear cualquier proveedor remoto.
2. **Carga tus datos** arrastrando un CSV o Excel a la barra de datos.
3. **Pregunta en el chat**, por ejemplo: *"Haz una regresión de salario sobre educación y experiencia con
   errores robustos"*.
4. Mira cómo el panel central se llena de tablas, gráficos y notas. Sigue conversando y se transforma.

---

## ¿Algo no funciona?

| Síntoma | Solución |
|---|---|
| `uv: command not found` | Reinicia la terminal tras instalar uv, o revisa el paso 1. |
| La ventana de Electron no abre | Ejecuta `node apps/desktop/node_modules/electron/install.js` y vuelve a `npm start`. |
| Error desde la terminal de VS Code | Usa `npm start` (ya limpia la variable `ELECTRON_RUN_AS_NODE` que rompe Electron). |
| El agente no responde | Revisa que hay un modelo configurado en **Ajustes** y mira `~/qtrix/logs/engine.log`. |

---

## Dónde guarda qtrix tus cosas

Todo vive en una única carpeta, `~/qtrix` por defecto (cámbiala en **Ajustes > Carpeta de qtrix**):

```
qtrix/
  settings.json   data/ (DuckDB)   memory/   models/ (GGUF)   runtime/ (llama.cpp)
  logs/           uploads/         exports/  LEEME.txt
```

---

## Para desarrolladores

```
apps/desktop   Electron + React (barra de datos · panel dinámico · chat con agentes)
engine         Motor Python: FastAPI, agente PydanticAI, registro de herramientas MCP,
               DuckDB, memoria de proyecto con turbovec
docs           Catálogo de librerías y decisiones de arquitectura (ADR)
```

**Solo la interfaz en el navegador** (con el motor lanzado a mano):

```bash
cd engine && uv run python -m qtrix_engine --port 8765
cd apps/desktop && npm run dev:web        # http://localhost:5173
```

**Usar las herramientas desde cualquier cliente MCP** (Claude Desktop, etc.):

```bash
cd engine && uv run qtrix-mcp --workspace ruta/al/workspace.duckdb
```

Expone el mismo registro que usan los agentes (`data_*`, `ols_*`, `panel_*`, `iv_*`, `ts_*`, `viz_*`, `ui_*`,
`memory_*`). La app en marcha también acepta `POST /tools/{nombre}` para llamadas directas sin LLM.

**Tests:**

```bash
npm test                                           # motor (pytest)
cd engine && uv run ruff check src tests           # lint
cd apps/desktop && npm run typecheck && npm run build
```

Las estimaciones se validan contra statsmodels/linearmodels (coeficientes y errores estándar a 1e-10) y el
bucle chat → herramienta → panel se prueba con un modelo simulado.

**Empaquetar un instalador** (`.exe`, `.dmg`, `.AppImage`): `cd apps/desktop && npm run dist`.

---

## Contribuir

¡Las contribuciones son bienvenidas! Lee [CONTRIBUTING.md](CONTRIBUTING.md) (en inglés), abre un issue para
comentar tu idea y envía un pull request. Los nuevos estimadores deben validarse contra una implementación de
referencia. En tu primer PR un bot te pedirá firmar el [CLA](CLA.md). Vulnerabilidades: [SECURITY.md](SECURITY.md).

## Licencia

qtrix es **código fuente disponible** bajo la [Business Source License 1.1](LICENSE): puedes usarlo, estudiarlo y
modificarlo libremente (también en producción, para investigación, docencia, uso interno o consultoría), pero no
ofrecerlo a terceros como servicio alojado o producto competidor. Cada versión pasa automáticamente a
**Apache-2.0** cuatro años después de su publicación. Para otros usos, contacta con el autor.

English version: [README.en.md](README.en.md).
