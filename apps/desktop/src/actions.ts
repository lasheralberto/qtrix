// What the agent is doing, in plain words: each tool call described in a short sentence (10 words at most) for the
// progress line, instead of the tool's technical name.
import type { CellInfo } from "./api";
import { T } from "./i18n";

type Args = Record<string, unknown>;
/** [while it runs, once done]; either may use the call's arguments. */
type Phrase = [(a: Args) => string, (a: Args) => string];

const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : "");
const lhs = (a: Args) => str(a.formula).split("~")[0].trim();
const of = (a: Args) => {
  const v = lhs(a) || str(a.series) || str(a.column) || str(a.y) || str(a.name);
  return v ? T("actions.of", { name: v }) : "";
};
const fixed = (running: string, done: string): Phrase => [() => running, () => done];

const PHRASES: Record<string, Phrase> = {
  data_list_datasets: fixed(T("actions.dataListDatasetsRunning"), T("actions.dataListDatasetsDone")),
  data_describe_structure: fixed(T("actions.dataDescribeStructureRunning"), T("actions.dataDescribeStructureDone")),
  data_set_structure: fixed(T("actions.dataSetStructureRunning"), T("actions.dataSetStructureDone")),
  data_set_panel_structure: fixed(T("actions.dataSetPanelStructureRunning"), T("actions.dataSetPanelStructureDone")),
  data_set_sample: fixed(T("actions.dataSetSampleRunning"), T("actions.dataSetSampleDone")),
  data_import_file: fixed(T("actions.dataImportFileRunning"), T("actions.dataImportFileDone")),
  data_summary_stats: fixed(T("actions.dataSummaryStatsRunning"), T("actions.dataSummaryStatsDone")),
  data_correlation: fixed(T("actions.dataCorrelationRunning"), T("actions.dataCorrelationDone")),
  data_preview: fixed(T("actions.dataPreviewRunning"), T("actions.dataPreviewDone")),
  data_sql: fixed(T("actions.dataSqlRunning"), T("actions.dataSqlDone")),
  data_genr: [(a) => T("actions.dataGenrRunning", { name: str(a.name) || T("actions.newName") }), (a) => T("actions.dataGenrDone", { name: str(a.name) || T("actions.newName") })],
  data_transform: fixed(T("actions.dataTransformRunning"), T("actions.dataTransformDone")),
  ols_fit: [(a) => T("actions.olsFitRunning", { of: of(a) }), (a) => T("actions.olsFitDone", { of: of(a) })],
  ols_binary: [(a) => T("actions.olsBinaryRunning", { of: of(a) }), (a) => T("actions.olsBinaryDone", { of: of(a) })],
  ols_count: [(a) => T("actions.olsCountRunning", { of: of(a) }), (a) => T("actions.olsCountDone", { of: of(a) })],
  ols_ordered: [(a) => T("actions.olsOrderedRunning", { of: of(a) }), (a) => T("actions.olsOrderedDone", { of: of(a) })],
  ols_multinomial: [(a) => T("actions.olsMultinomialRunning", { of: of(a) }), (a) => T("actions.olsMultinomialDone", { of: of(a) })],
  ols_quantile: [(a) => T("actions.olsQuantileRunning", { of: of(a) }), (a) => T("actions.olsQuantileDone", { of: of(a) })],
  ols_diagnostics: fixed(T("actions.olsDiagnosticsRunning"), T("actions.olsDiagnosticsDone")),
  ols_compare: fixed(T("actions.olsCompareRunning"), T("actions.olsCompareDone")),
  panel_fit: [(a) => T("actions.panelFitRunning", { of: of(a) }), (a) => T("actions.panelFitDone", { of: of(a) })],
  panel_event_study: [(a) => T("actions.panelEventStudyRunning", { of: of(a) }), (a) => T("actions.panelEventStudyDone", { of: of(a) })],
  causal_rdd: [(a) => T("actions.causalRddRunning", { of: of(a) }), (a) => T("actions.causalRddDone", { of: of(a) })],
  causal_dml: [(a) => T("actions.causalDmlRunning", { of: of(a) }), (a) => T("actions.causalDmlDone", { of: of(a) })],
  causal_rd_density: fixed(T("actions.causalRdDensityRunning"), T("actions.causalRdDensityDone")),
  causal_forest: [(a) => T("actions.causalForestRunning", { of: of(a) }), (a) => T("actions.causalForestDone", { of: of(a) })],
  causal_pds: [(a) => T("actions.causalPdsRunning", { of: of(a) }), (a) => T("actions.causalPdsDone", { of: of(a) })],
  causal_sensitivity: fixed(T("actions.causalSensitivityRunning"), T("actions.causalSensitivityDone")),
  ols_margins: fixed(T("actions.olsMarginsRunning"), T("actions.olsMarginsDone")),
  causal_synth: [(a) => T("actions.causalSynthRunning", { of: of(a) }), (a) => T("actions.causalSynthDone", { of: of(a) })],
  panel_hausman: fixed(T("actions.panelHausmanRunning"), T("actions.panelHausmanDone")),
  iv_fit: [(a) => T("actions.ivFitRunning", { of: of(a) }), (a) => T("actions.ivFitDone", { of: of(a) })],
  ts_unit_root: fixed(T("actions.tsUnitRootRunning"), T("actions.tsUnitRootDone")),
  ts_arima: fixed(T("actions.tsArimaRunning"), T("actions.tsArimaDone")),
  ts_var: fixed(T("actions.tsVarRunning"), T("actions.tsVarDone")),
  ts_ardl: fixed(T("actions.tsArdlRunning"), T("actions.tsArdlDone")),
  ts_markov: fixed(T("actions.tsMarkovRunning"), T("actions.tsMarkovDone")),
  ts_granger: fixed(T("actions.tsGrangerRunning"), T("actions.tsGrangerDone")),
  ts_vecm: fixed(T("actions.tsVecmRunning"), T("actions.tsVecmDone")),
  ts_cointegration: fixed(T("actions.tsCointegrationRunning"), T("actions.tsCointegrationDone")),
  ts_acf_pacf: fixed(T("actions.tsAcfPacfRunning"), T("actions.tsAcfPacfDone")),
  ts_breaks: fixed(T("actions.tsBreaksRunning"), T("actions.tsBreaksDone")),
  ts_forecast: fixed(T("actions.tsForecastRunning"), T("actions.tsForecastDone")),
  vol_garch: fixed(T("actions.volGarchRunning"), T("actions.volGarchDone")),
  obj_list: fixed(T("actions.objListRunning"), T("actions.objListDone")),
  obj_get: [(a) => T("actions.objGetRunning", { name: str(a.name) || T("actions.aModel") }), (a) => T("actions.objGetDone", { name: str(a.name) || T("actions.model") })],
  obj_versions: fixed(T("actions.objVersionsRunning"), T("actions.objVersionsDone")),
  obj_views: fixed(T("actions.objViewsRunning"), T("actions.objViewsDone")),
  obj_view: fixed(T("actions.objViewRunning"), T("actions.objViewDone")),
  obj_freeze: fixed(T("actions.objFreezeRunning"), T("actions.objFreezeDone")),
  obj_rename: fixed(T("actions.objRenameRunning"), T("actions.objRenameDone")),
  obj_copy: fixed(T("actions.objCopyRunning"), T("actions.objCopyDone")),
  obj_delete: fixed(T("actions.objDeleteRunning"), T("actions.objDeleteDone")),
  obj_group: fixed(T("actions.objGroupRunning"), T("actions.objGroupDone")),
  obj_lock: fixed(T("actions.objLockRunning"), T("actions.objLockDone")),
  obj_reestimate: fixed(T("actions.objReestimateRunning"), T("actions.objReestimateDone")),
  viz_scatter: fixed(T("actions.vizScatterRunning"), T("actions.vizScatterDone")),
  viz_histogram: fixed(T("actions.vizHistogramRunning"), T("actions.vizHistogramDone")),
  viz_line: fixed(T("actions.vizLineRunning"), T("actions.vizLineDone")),
  viz_coefplot: fixed(T("actions.vizCoefplotRunning"), T("actions.vizCoefplotDone")),
  viz_residuals: fixed(T("actions.vizResidualsRunning"), T("actions.vizResidualsDone")),
  ui_list_blocks: fixed(T("actions.uiListBlocksRunning"), T("actions.uiListBlocksDone")),
  ui_add_view: fixed(T("actions.uiAddViewRunning"), T("actions.uiAddViewDone")),
  ui_freeze_block: fixed(T("actions.uiFreezeBlockRunning"), T("actions.uiFreezeBlockDone")),
  ui_add_equation: fixed(T("actions.uiAddEquationRunning"), T("actions.uiAddEquationDone")),
  ui_remove_block: fixed(T("actions.uiRemoveBlockRunning"), T("actions.uiRemoveBlockDone")),
  ui_reorder: fixed(T("actions.uiReorderRunning"), T("actions.uiReorderDone")),
  memory_search: fixed(T("actions.memorySearchRunning"), T("actions.memorySearchDone")),
  py_run: fixed(T("actions.pyRunRunning"), T("actions.pyRunDone")),
  search_tools: fixed(T("actions.searchToolsRunning"), T("actions.searchToolsDone")),
  load_capability: fixed(T("actions.loadCapabilityRunning"), T("actions.loadCapabilityDone")),
  read_file: fixed(T("actions.readFileRunning"), T("actions.readFileDone")),
  search_files: fixed(T("actions.searchFilesRunning"), T("actions.searchFilesDone")),
};

const FALLBACK: [string, Phrase][] = [
  ["data_", fixed(T("actions.fallbackDataRunning"), T("actions.fallbackDataDone"))],
  ["ts_", fixed(T("actions.fallbackTsRunning"), T("actions.fallbackTsDone"))],
  ["obj_", fixed(T("actions.fallbackObjRunning"), T("actions.fallbackObjDone"))],
  ["viz_", fixed(T("actions.fallbackVizRunning"), T("actions.fallbackVizDone"))],
  ["ui_", fixed(T("actions.fallbackUiRunning"), T("actions.fallbackUiDone"))],
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
    fixed(T("actions.defaultRunning"), T("actions.defaultDone"));
  if (cell.ok === false) return T("actions.retrying");
  const args = (cell.args ?? {}) as Args;
  return words(cell.ok === undefined ? phrase[0](args) : phrase[1](args));
}
