// What the agent is doing, in plain words: each tool call described in a short sentence (10 words at most) for the
// progress line, instead of the tool's technical name.
import type { CellInfo } from "./api";

type Args = Record<string, unknown>;
/** [while it runs, once done]; either may use the call's arguments. */
type Phrase = [(a: Args) => string, (a: Args) => string];

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : "");
const lhs = (a: Args) => str(a.formula).split("~")[0].trim();
const of = (a: Args) => {
  const v = lhs(a) || str(a.series) || str(a.column) || str(a.y) || str(a.name);
  return v ? ` de ${v}` : "";
};
const fixed = (running: string, done: string): Phrase => [() => running, () => done];

const PHRASES: Record<string, Phrase> = {
  data_list_datasets: fixed("Revisando los datos disponibles", "Datos disponibles revisados"),
  data_describe_structure: fixed("Identificando la estructura de los datos", "Estructura de los datos identificada"),
  data_set_structure: fixed("Declarando la estructura de los datos", "Estructura de los datos declarada"),
  data_set_panel_structure: fixed("Declarando entidad y tiempo del panel", "Estructura de panel declarada"),
  data_set_sample: fixed("Acotando la muestra de estimación", "Muestra de estimación acotada"),
  data_import_file: fixed("Importando el archivo de datos", "Archivo de datos importado"),
  data_summary_stats: fixed("Calculando estadísticos descriptivos", "Estadísticos descriptivos calculados"),
  data_correlation: fixed("Calculando correlaciones entre variables", "Correlaciones calculadas"),
  data_preview: fixed("Mirando las primeras filas", "Primeras filas revisadas"),
  data_sql: fixed("Consultando los datos con SQL", "Consulta SQL realizada"),
  data_genr: [(a) => `Creando la serie ${str(a.name) || "nueva"}`, (a) => `Serie ${str(a.name) || "nueva"} creada`],
  data_transform: fixed("Creando variables derivadas", "Variables derivadas creadas"),
  ols_fit: [(a) => `Estimando regresión OLS${of(a)}`, (a) => `Regresión OLS${of(a)} estimada`],
  ols_binary: [(a) => `Estimando modelo logit/probit${of(a)}`, (a) => `Modelo logit/probit${of(a)} estimado`],
  ols_count: [(a) => `Estimando modelo de conteo${of(a)}`, (a) => `Modelo de conteo${of(a)} estimado`],
  ols_ordered: [(a) => `Estimando modelo de respuesta ordenada${of(a)}`, (a) => `Modelo ordenado${of(a)} estimado`],
  ols_multinomial: [(a) => `Estimando logit multinomial${of(a)}`, (a) => `Logit multinomial${of(a)} estimado`],
  ols_quantile: [(a) => `Estimando regresión cuantílica${of(a)}`, (a) => `Regresión cuantílica${of(a)} estimada`],
  ols_diagnostics: fixed("Comprobando los supuestos del modelo", "Supuestos del modelo comprobados"),
  ols_compare: fixed("Comparando modelos en una tabla", "Modelos comparados"),
  panel_fit: [(a) => `Estimando modelo de panel${of(a)}`, (a) => `Modelo de panel${of(a)} estimado`],
  panel_event_study: [(a) => `Estimando estudio de eventos${of(a)}`, (a) => `Estudio de eventos${of(a)} estimado`],
  causal_rdd: [(a) => `Estimando regresión discontinua${of(a)}`, (a) => `Regresión discontinua${of(a)} estimada`],
  causal_dml: [(a) => `Estimando double machine learning${of(a)}`, (a) => `Double ML${of(a)} estimado`],
  causal_rd_density: fixed("Contrastando manipulación en el umbral", "Test de manipulación realizado"),
  causal_forest: [(a) => `Estimando efectos heterogéneos${of(a)}`, (a) => `Efectos heterogéneos${of(a)} estimados`],
  causal_pds: [(a) => `Seleccionando controles con lasso${of(a)}`, (a) => `Selección con lasso${of(a)} estimada`],
  causal_sensitivity: fixed("Midiendo la sensibilidad a variables omitidas", "Sensibilidad a variables omitidas medida"),
  ols_margins: fixed("Calculando efectos marginales", "Efectos marginales calculados"),
  causal_synth: [(a) => `Construyendo control sintético${of(a)}`, (a) => `Control sintético${of(a)} construido`],
  panel_hausman: fixed("Contrastando efectos fijos frente a aleatorios", "Test de Hausman realizado"),
  iv_fit: [(a) => `Estimando con variables instrumentales${of(a)}`, (a) => `Estimación IV${of(a)} terminada`],
  ts_unit_root: fixed("Contrastando raíz unitaria de la serie", "Test de raíz unitaria realizado"),
  ts_arima: fixed("Ajustando un modelo ARIMA", "Modelo ARIMA ajustado"),
  ts_var: fixed("Ajustando un modelo VAR", "Modelo VAR ajustado"),
  ts_ardl: fixed("Ajustando un modelo ARDL", "Modelo ARDL ajustado"),
  ts_markov: fixed("Ajustando un modelo de cambio de régimen", "Modelo de cambio de régimen ajustado"),
  ts_granger: fixed("Contrastando causalidad de Granger", "Test de causalidad de Granger realizado"),
  ts_vecm: fixed("Ajustando un modelo de corrección de error", "Modelo VECM ajustado"),
  ts_cointegration: fixed("Contrastando cointegración entre series", "Test de cointegración realizado"),
  ts_acf_pacf: fixed("Calculando autocorrelaciones", "Autocorrelaciones calculadas"),
  ts_breaks: fixed("Buscando cambios estructurales", "Búsqueda de cambios estructurales terminada"),
  ts_forecast: fixed("Generando la predicción", "Predicción generada"),
  vol_garch: fixed("Ajustando un modelo de volatilidad", "Modelo de volatilidad ajustado"),
  obj_list: fixed("Revisando los modelos del proyecto", "Modelos del proyecto revisados"),
  obj_get: [(a) => `Consultando ${str(a.name) || "un modelo"}`, (a) => `${str(a.name) || "Modelo"} consultado`],
  obj_versions: fixed("Revisando versiones del modelo", "Versiones del modelo revisadas"),
  obj_views: fixed("Viendo qué vistas ofrece el modelo", "Vistas del modelo revisadas"),
  obj_view: fixed("Calculando una vista del modelo", "Vista del modelo calculada"),
  obj_freeze: fixed("Guardando una instantánea del resultado", "Instantánea guardada"),
  obj_rename: fixed("Renombrando un objeto", "Objeto renombrado"),
  obj_copy: fixed("Copiando un objeto", "Objeto copiado"),
  obj_delete: fixed("Eliminando un objeto", "Objeto eliminado"),
  obj_group: fixed("Agrupando series", "Series agrupadas"),
  obj_lock: fixed("Bloqueando un objeto", "Objeto bloqueado"),
  obj_reestimate: fixed("Re-estimando el modelo con cambios", "Modelo re-estimado"),
  viz_scatter: fixed("Dibujando un diagrama de dispersión", "Diagrama de dispersión dibujado"),
  viz_histogram: fixed("Dibujando un histograma", "Histograma dibujado"),
  viz_line: fixed("Dibujando un gráfico de líneas", "Gráfico de líneas dibujado"),
  viz_coefplot: fixed("Dibujando los coeficientes con intervalos", "Gráfico de coeficientes dibujado"),
  viz_residuals: fixed("Dibujando los residuos del modelo", "Residuos del modelo dibujados"),
  ui_list_blocks: fixed("Revisando el panel de análisis", "Panel de análisis revisado"),
  ui_add_view: fixed("Añadiendo una vista al panel", "Vista añadida al panel"),
  ui_freeze_block: fixed("Congelando un bloque del panel", "Bloque del panel congelado"),
  ui_add_equation: fixed("Escribiendo la ecuación estimada", "Ecuación escrita"),
  ui_remove_block: fixed("Quitando un bloque obsoleto", "Bloque obsoleto quitado"),
  ui_reorder: fixed("Reordenando el panel", "Panel reordenado"),
  memory_search: fixed("Buscando en la memoria del proyecto", "Memoria del proyecto consultada"),
  py_run: fixed("Ejecutando un cálculo en Python", "Cálculo en Python terminado"),
  search_tools: fixed("Buscando la herramienta adecuada", "Herramienta encontrada"),
  load_capability: fixed("Consultando una guía de método", "Guía de método consultada"),
  read_file: fixed("Leyendo una guía de método", "Guía de método leída"),
  search_files: fixed("Buscando en las guías de método", "Búsqueda en las guías terminada"),
};

const FALLBACK: [string, Phrase][] = [
  ["data_", fixed("Trabajando con los datos", "Datos procesados")],
  ["ts_", fixed("Analizando la serie temporal", "Análisis de series terminado")],
  ["obj_", fixed("Gestionando los modelos", "Modelos gestionados")],
  ["viz_", fixed("Preparando un gráfico", "Gráfico preparado")],
  ["ui_", fixed("Ordenando el panel", "Panel ordenado")],
];

/** At most ``n`` words, with an ellipsis when cut. */
export function words(text: string, n = 10): string {
  const w = text.trim().split(/\s+/);
  return w.length <= n ? w.join(" ") : w.slice(0, n).join(" ") + "…";
}

/** The progress sentence of one action: what it does while it runs, what it did once done, or that it is retrying. */
export function describeAction(cell: CellInfo): string {
  const phrase =
    PHRASES[cell.name] ??
    FALLBACK.find(([prefix]) => cell.name.startsWith(prefix))?.[1] ??
    fixed("Trabajando en el análisis", "Paso del análisis terminado");
  if (cell.ok === false) return "Corrigiendo un error y reintentando";
  const args = (cell.args ?? {}) as Args;
  return words(cell.ok === undefined ? phrase[0](args) : phrase[1](args));
}
