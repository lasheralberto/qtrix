"""The catalog of analysis methods the user sees before talking to the agent.

The tool registry groups capabilities by code namespace; an analyst thinks in areas and methods (fixed effects, unit
roots, propensity score matching). This module is that second view: area -> method -> variants, each method pointing
at the registry tools (or the causal library run in a Python cell) that implement it. Availability and the data
structures a method needs are derived from the registry, so the catalog cannot claim what the engine cannot run.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from qtrix_engine.tools.registry import get_tool


@dataclass(frozen=True)
class Method:
    id: str
    label: str
    summary: str
    tools: tuple[str, ...]
    prompt: str  # an example request; «…» marks what the user fills in
    variants: tuple[str, ...] = ()
    library: tuple[str, ...] = ()  # functions of contrib.econometrics.econometric_algorithm, run via py_run
    structures: tuple[str, ...] | None = None  # overrides the structures derived from the tools' kinds


@dataclass(frozen=True)
class Area:
    id: str
    label: str
    summary: str
    methods: tuple[Method, ...]


PANEL = ("panel", "panel_undated")

AREAS: tuple[Area, ...] = (
    Area("data", "Datos y descriptivos", "Importar, explorar, transformar y estructurar los datos", (
        Method("import", "Importar datos", "CSV, Excel o Parquet como una página del cuaderno", ("data_import_file",),
               "Importa el fichero «ruta» y describe su estructura"),
        Method("structure", "Estructura de los datos", "Declarar corte transversal, serie temporal o panel",
               ("data_set_structure", "data_set_panel_structure", "data_describe_structure"),
               "Declara «dataset» como panel con «id» como individuo y «año» como periodo",
               ("Corte transversal", "Serie temporal", "Panel")),
        Method("descriptives", "Estadísticos descriptivos", "Media, dispersión, cuantiles y valores perdidos",
               ("data_summary_stats", "data_preview"), "Describe las variables de «dataset»"),
        Method("correlation", "Correlaciones", "Matriz de correlaciones entre variables", ("data_correlation",),
               "Muestra la matriz de correlaciones de «variables»"),
        Method("transform", "Transformaciones y series nuevas", "Logaritmos, diferencias, retardos, dummies",
               ("data_genr", "data_transform"), "Genera log(«variable») y su primera diferencia"),
        Method("sample", "Muestra activa", "Restringir el análisis a un subconjunto de observaciones",
               ("data_set_sample",), "Restringe la muestra a «condición»"),
        Method("sql", "Consultas SQL", "Filtrar, agregar y cruzar datasets con DuckDB", ("data_sql",),
               "Agrega «dataset» por «grupo» con la media de «variable»"),
    )),
    Area("cross_section", "Corte transversal", "Regresión lineal, respuesta binaria y contrastes de especificación", (
        Method("ols", "Mínimos cuadrados (MCO)", "Regresión lineal con errores robustos, cluster o HAC", ("ols_fit",),
               "Estima por MCO «y» sobre «x1, x2» con errores robustos",
               ("Errores clásicos", "Robustos HC0–HC3", "Cluster", "HAC (Newey-West)")),
        Method("binary", "Respuesta binaria", "Modelos de probabilidad para una variable 0/1", ("ols_binary",),
               "Estima un logit de «y» sobre «x1, x2» y da los efectos marginales", ("Logit", "Probit")),
        Method("count", "Datos de conteo", "Número de eventos: Poisson (PPML), binomial negativa, inflado en ceros",
               ("ols_count",), "Estima un modelo de Poisson de «y» sobre «x1, x2» con errores robustos",
               ("Poisson / PPML", "Binomial negativa", "Poisson inflado en ceros", "Binomial negativa inflada")),
        Method("ordered", "Respuesta ordenada", "Categorías con un orden natural (valoraciones, niveles)",
               ("ols_ordered",), "Estima un logit ordenado de «y» sobre «x1, x2»",
               ("Logit ordenado", "Probit ordenado")),
        Method("multinomial", "Respuesta multinomial", "Elección entre categorías sin orden", ("ols_multinomial",),
               "Estima un logit multinomial de «y» sobre «x1, x2»"),
        Method("quantile", "Regresión cuantílica", "Efecto sobre un cuantil de la variable, no sobre la media",
               ("ols_quantile",), "Estima la regresión cuantílica de «y» sobre «x» en los cuantiles 0,1, 0,5 y 0,9",
               ("Mediana", "Cuantiles extremos")),
        Method("diagnostics", "Contrastes de especificación", "Heterocedasticidad, autocorrelación, normalidad, RESET",
               ("ols_diagnostics",), "¿Hay heterocedasticidad en «modelo»? Comprueba los supuestos"),
        Method("compare", "Comparar modelos", "Tabla de regresiones lado a lado", ("ols_compare",),
               "Compara «modelo1» y «modelo2» en una tabla"),
    )),
    Area("panel", "Datos de panel", "Estimadores para individuos observados a lo largo del tiempo", (
        Method("panel_fit", "Estimadores de panel", "Heterogeneidad no observada por individuo y periodo",
               ("panel_fit",), "Estima efectos fijos de «y» sobre «x1, x2» con errores cluster por individuo",
               ("Efectos fijos", "Efectos fijos bidireccionales", "Efectos aleatorios", "Primeras diferencias",
                "Between", "Pooled")),
        Method("hausman", "Test de Hausman", "Elegir entre efectos fijos y aleatorios", ("panel_hausman",),
               "Compara efectos fijos y aleatorios con el test de Hausman"),
    )),
    Area("causal", "Inferencia causal", "Estrategias de identificación de efectos causales", (
        Method("iv", "Variables instrumentales", "Regresor endógeno con instrumentos externos", ("iv_fit",),
               "Estima el efecto de «x» sobre «y» instrumentando con «z»", ("2SLS", "LIML", "GMM")),
        Method("propensity", "Selección en observables", "Puntuación de propensión: emparejamiento y ponderación",
               ("py_run",), "Estima el efecto de «tratamiento» sobre «y» con propensity score matching",
               ("Matching", "IPW", "Doblemente robusto"),
               library=("propensity_score_construction", "propensity_score_matching",
                        "propensity_score_inverse_probability_weighting",
                        "propensity_score_double_robust_estimator_augmented_IPW")),
        Method("did", "Diferencias en diferencias", "Política con una fecha de adopción común", ("panel_fit",),
               "Estima un diferencias en diferencias de «y» con «tratado» y «post»", ("TWFE",),
               library=("Static_Diff_in_Diff_regression",)),
        Method("event_study", "Estudio de eventos", "Adopción escalonada y dinámica del efecto", ("py_run",),
               "Estima un estudio de eventos de «y» con adopción escalonada y muestra las tendencias previas",
               library=("Staggered_Diff_in_Diff_Event_Study_regression",
                        "Staggered_Diff_in_Diff_Event_Study_visualization"), structures=PANEL),
        Method("rdd", "Regresión discontinua", "Umbral en una variable de asignación", ("py_run",),
               "Estima una regresión discontinua de «y» en el umbral «c» de «variable»", ("Sharp", "Fuzzy"),
               library=("Sharp_Regression_Discontinuity_Design_regression",
                        "Fuzzy_Regression_Discontinuity_Design_regression")),
    )),
    Area("time_series", "Series temporales", "Dinámica, estacionariedad, sistemas y predicción", (
        Method("correlogram", "Correlograma", "Funciones de autocorrelación simple y parcial", ("ts_acf_pacf",),
               "Muestra el correlograma de «serie»", ("ACF", "PACF")),
        Method("unit_root", "Raíces unitarias", "¿Es la serie estacionaria?", ("ts_unit_root",),
               "Contrasta si «serie» tiene raíz unitaria",
               ("ADF", "KPSS", "Phillips-Perron", "Zivot-Andrews", "DF-GLS", "Razón de varianzas")),
        Method("arima", "ARIMA", "Modelos univariantes, con estacionalidad", ("ts_arima",),
               "Estima un ARIMA(1,1,1) para «serie»", ("ARIMA", "SARIMA")),
        Method("ardl", "ARDL y contraste de límites", "Efectos dinámicos y relación a largo plazo, sin exigir I(1)",
               ("ts_ardl",), "Estima un ARDL de «y» sobre «x» y contrasta si hay relación a largo plazo",
               ("ARDL", "Contraste de límites (bounds test)")),
        Method("markov", "Cambio de régimen", "Regímenes ocultos (expansión/recesión, calma/turbulencia)",
               ("ts_markov",), "Estima un modelo de cambio de régimen de Markov de «serie» con dos regímenes",
               ("Media cambiante", "Autorregresivo de Markov")),
        Method("granger", "Causalidad de Granger", "¿Ayuda el pasado de una serie a predecir otra?",
               ("ts_granger",), "Contrasta si «x» causa en el sentido de Granger a «y»"),
        Method("var", "VAR", "Vectores autorregresivos e impulso-respuesta", ("ts_var",),
               "Estima un VAR con «series» y muestra las funciones impulso-respuesta"),
        Method("cointegration", "Cointegración y VECM", "Relaciones de equilibrio a largo plazo",
               ("ts_cointegration", "ts_vecm"), "¿Están cointegradas «series»? Si lo están, estima un VECM",
               ("Johansen", "Engle-Granger", "VECM")),
        Method("breaks", "Cambios estructurales", "Detectar rupturas en la serie", ("ts_breaks",),
               "Busca cambios estructurales en «serie»"),
        Method("forecast", "Predicción", "Pronóstico automático con intervalos", ("ts_forecast",),
               "Predice «serie» 8 periodos hacia delante", ("Auto-ARIMA", "Auto-ETS", "Naive", "Naive estacional")),
    )),
    Area("volatility", "Volatilidad", "Varianza condicional de rentabilidades", (
        Method("garch", "Modelos GARCH", "Agrupamiento de la volatilidad y asimetrías", ("vol_garch",),
               "Estima un GARCH(1,1) con errores t para «rentabilidad»", ("GARCH", "EGARCH", "GJR-GARCH")),
    )),
    Area("charts", "Gráficos", "Visualizaciones en el panel de análisis", (
        Method("line", "Líneas", "Evolución temporal de una o varias series", ("viz_line",),
               "Dibuja «series» a lo largo del tiempo"),
        Method("scatter", "Dispersión", "Relación entre dos variables con recta de ajuste", ("viz_scatter",),
               "Dibuja «y» frente a «x»"),
        Method("histogram", "Histograma", "Distribución de una variable", ("viz_histogram",),
               "Muestra el histograma de «variable»"),
        Method("coefplot", "Coeficientes", "Estimaciones con intervalos de confianza", ("viz_coefplot",),
               "Dibuja los coeficientes de «modelo»"),
        Method("residuals", "Residuos", "Residuos frente a ajustados, QQ e histograma", ("viz_residuals",),
               "Dibuja los residuos de «modelo»", ("Ajustados", "QQ", "Histograma")),
    )),
)


def _method_entry(m: Method) -> dict[str, Any]:
    specs = [get_tool(name) for name in m.tools]
    missing_tools = [name for name, spec in zip(m.tools, specs, strict=True) if spec is None]
    present = [s for s in specs if s is not None]
    missing_modules = sorted({mod for s in present for mod in s.requires if find_spec(mod) is None})
    if m.structures is not None:
        structures: list[str] | None = list(m.structures)
    elif not present or any(s.kinds is None for s in present):
        structures = None
    else:
        structures = sorted(set().union(*(s.kinds or () for s in present)))
    return {
        "id": m.id, "label": m.label, "summary": m.summary, "tools": list(m.tools), "variants": list(m.variants),
        "library": list(m.library), "prompt": m.prompt, "structures": structures,
        "available": not missing_tools and not missing_modules, "missing": missing_tools + missing_modules,
    }


def methods_catalog() -> list[dict[str, Any]]:
    """The catalog as the UI reads it: areas with their methods, availability resolved against the registry."""
    return [{"id": a.id, "label": a.label, "summary": a.summary, "methods": [_method_entry(m) for m in a.methods]}
            for a in AREAS]
