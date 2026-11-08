"""
App de Streamlit: riesgo de fallecimiento por COVID-19 en Colombia.

Proyecto integrador de Aprendizaje de Máquinas (metodología CRISP-DM).
Modelo: Pipeline de scikit-learn (imputación + OneHotEncoder + StandardScaler + clasificador)
entrenado con casos positivos publicados por el INS en datos.gov.co.
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

COLUMNAS_ENTRADA = [
    "fecha reporte web", "Fecha de inicio de síntomas", "Fecha de diagnóstico", "Edad",
    "Unidad de medida de edad", "Sexo", "Tipo de contagio", "Nombre departamento",
    "Pertenencia étnica",
]
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]
UNIDADES = {"Años": 1, "Meses": 2, "Días": 3}


# --------------------------------------------------------------------------- #
# Carga de artefactos
# --------------------------------------------------------------------------- #
def _ruta(nombre: str) -> Path:
    """Busca el archivo junto a app.py o dentro de una subcarpeta 'despliegue'."""
    for carpeta in (BASE, BASE / "despliegue"):
        if (carpeta / nombre).exists():
            return carpeta / nombre
    raise FileNotFoundError(f"No se encontró '{nombre}' en el repositorio.")


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_artefactos():
    modelo = joblib.load(_ruta("modelo_final.joblib"))
    le_y = joblib.load(_ruta("label_encoder.joblib"))
    with open(_ruta("metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, le_y, meta


# --------------------------------------------------------------------------- #
# Preparación de datos (misma lógica del entrenamiento)
# --------------------------------------------------------------------------- #
def preparar_datos(df_nuevo: pd.DataFrame, meta: dict) -> pd.DataFrame:
    faltan = [c for c in COLUMNAS_ENTRADA if c not in df_nuevo.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el archivo: {faltan}")
    d = df_nuevo[COLUMNAS_ENTRADA].copy()

    # Edad en años (1 = años, 2 = meses, 3 = días)
    u = pd.to_numeric(d["Unidad de medida de edad"], errors="coerce")
    e = pd.to_numeric(d["Edad"], errors="coerce")
    d["edad_anios"] = np.select([u == 1, u == 2, u == 3], [e, e / 12, e / 365], default=np.nan)
    d.loc[(d["edad_anios"] < 0) | (d["edad_anios"] > 110), "edad_anios"] = np.nan

    # Fechas -> mes, año y días entre síntomas y diagnóstico
    for c in ["fecha reporte web", "Fecha de inicio de síntomas", "Fecha de diagnóstico"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d["dias_sintoma_diag"] = (d["Fecha de diagnóstico"] - d["Fecha de inicio de síntomas"]).dt.days
    d.loc[(d["dias_sintoma_diag"] < 0) | (d["dias_sintoma_diag"] > 60), "dias_sintoma_diag"] = np.nan
    d["mes_reporte"] = d["fecha reporte web"].dt.month
    d["anio_reporte"] = d["fecha reporte web"].dt.year

    # Texto normalizado contra las categorías con que se entrenó el modelo
    for c in ["Sexo", "Tipo de contagio", "Nombre departamento"]:
        mapa = {k.upper(): k for k in meta["categorias"][c]}
        t = d[c].astype(str).str.strip().str.upper().where(d[c].notna())
        d[c] = t.map(lambda v: mapa.get(v, v) if pd.notna(v) else np.nan)
    d["Pertenencia étnica"] = d["Pertenencia étnica"].map(
        lambda v: str(int(float(v))) if pd.notna(v) else np.nan)

    return d[meta["features"]]


def predecir(modelo, le_y, meta, X: pd.DataFrame):
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba >= meta["umbral"]).astype(int)
    return proba, le_y.inverse_transform(pred)


def fila_individual(edad, unidad, sexo, contagio, depto, etnia, dias, mes, anio) -> pd.DataFrame:
    rep = pd.Timestamp(year=anio, month=mes, day=15)
    diag = rep - pd.Timedelta(days=1)
    ini = diag - pd.Timedelta(days=int(dias)) if dias is not None else pd.NaT
    return pd.DataFrame([{
        "fecha reporte web": rep, "Fecha de inicio de síntomas": ini, "Fecha de diagnóstico": diag,
        "Edad": edad, "Unidad de medida de edad": unidad, "Sexo": sexo,
        "Tipo de contagio": contagio, "Nombre departamento": depto, "Pertenencia étnica": etnia,
    }])


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
def pestaña_individual(modelo, le_y, meta):
    cats = meta["categorias"]
    anios = meta["rango_numericas"]["anio_reporte"]
    st.subheader("Caso individual")
    st.caption("Ingresa los datos de un caso positivo de COVID-19 al momento del diagnóstico.")

    c1, c2 = st.columns(2)
    with c1:
        edad = st.number_input("Edad", min_value=0, max_value=110, value=60, step=1)
        unidad_txt = st.selectbox("Unidad de la edad", list(UNIDADES), index=0)
        sexo_txt = st.radio("Sexo", ["Femenino", "Masculino"], horizontal=True)
        contagio = st.selectbox("Tipo de contagio", cats["Tipo de contagio"],
                                format_func=str.title)
        depto = st.selectbox("Departamento", cats["Nombre departamento"], format_func=str.title)
    with c2:
        etnia = st.selectbox("Pertenencia étnica (código del INS)", cats["Pertenencia étnica"])
        sin_dato = st.checkbox("No se conoce la fecha de inicio de síntomas")
        dias = None if sin_dato else st.number_input(
            "Días entre el inicio de síntomas y el diagnóstico", min_value=0, max_value=60, value=5)
        anio = st.selectbox("Año de reporte", list(range(int(anios[0]), int(anios[1]) + 1)),
                            index=min(1, int(anios[1]) - int(anios[0])))
        mes = st.selectbox("Mes de reporte", list(range(1, 13)), format_func=lambda m: MESES[m - 1],
                           index=5)

    if st.button("Calcular riesgo", type="primary"):
        sexo = "F" if sexo_txt == "Femenino" else "M"
        fila = fila_individual(edad, UNIDADES[unidad_txt], sexo, contagio, depto, etnia, dias, mes, anio)
        X = preparar_datos(fila, meta)
        proba, etiqueta = predecir(modelo, le_y, meta, X)
        p, umbral = float(proba[0]), meta["umbral"]

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Índice de riesgo", f"{p:.3f}")
        m2.metric("Umbral de decisión", f"{umbral:.3f}")
        st.progress(min(max(p, 0.0), 1.0))
        if p >= umbral:
            st.error(f"**Predicción: {etiqueta[0]}** — riesgo ALTO. Se recomienda priorizar el seguimiento clínico.")
        else:
            st.success(f"**Predicción: {etiqueta[0]}** — riesgo BAJO. Seguimiento habitual.")
        st.caption("El índice es un valor relativo de riesgo (el modelo se entrenó ponderando la clase "
                   "minoritaria), no una probabilidad real de fallecer.")


def pestaña_lote(modelo, le_y, meta):
    st.subheader("Datos nuevos desde un archivo CSV")
    st.caption("Sube un CSV con los datos de varios casos. Se aplica la misma preparación del entrenamiento.")
    st.markdown("**Columnas requeridas:** " + ", ".join(f"`{c}`" for c in COLUMNAS_ENTRADA))

    ejemplo = _ruta("datos_nuevos.csv") if any((c / "datos_nuevos.csv").exists()
                                               for c in (BASE, BASE / "despliegue")) else None
    if ejemplo is not None:
        st.download_button("Descargar archivo de ejemplo", ejemplo.read_bytes(),
                           file_name="datos_nuevos.csv", mime="text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is None:
        return
    try:
        df = pd.read_csv(archivo)
        X = preparar_datos(df, meta)
    except Exception as exc:  # error legible para el usuario
        st.error(f"No se pudo procesar el archivo: {exc}")
        return

    proba, etiquetas = predecir(modelo, le_y, meta, X)
    out = df.copy()
    out["Índice de riesgo"] = np.round(proba, 3)
    out["Predicción"] = etiquetas

    st.success(f"Se procesaron {len(out):,} registros.")
    c1, c2 = st.columns(2)
    c1.metric("Riesgo ALTO (Fallecido)", int((etiquetas == "Fallecido").sum()))
    c2.metric("Riesgo BAJO (Recuperado)", int((etiquetas == "Recuperado").sum()))
    st.dataframe(out, use_container_width=True)

    buffer = io.StringIO()
    out.to_csv(buffer, index=False)
    st.download_button("Descargar predicciones", buffer.getvalue().encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")


def pestaña_modelo(meta):
    st.subheader("Acerca del modelo")
    st.markdown(
        f"- **Algoritmo:** {meta['modelo']}\n"
        f"- **Variables de entrada:** {', '.join(meta['features'])}\n"
        f"- **Umbral de decisión:** {meta['umbral']:.3f} (el que maximiza el F1 de la clase «Fallecido» "
        "en validación cruzada)"
    )
    met = {k.lower().replace("-", "_"): v for k, v in meta.get("metricas_test_umbral_optimo", {}).items()}
    st.markdown("**Desempeño en el conjunto de prueba (datos nunca vistos):**")
    etiquetas = [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
                 ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc")]
    columnas = st.columns(len(etiquetas))
    for col, (titulo, clave) in zip(columnas, etiquetas):
        valor = met.get(clave)
        col.metric(titulo, f"{valor:.3f}" if isinstance(valor, (int, float)) else "—")
    st.warning(
        "**Limitaciones.** Los fallecidos son cerca del 2,7 % de los casos y el modelo solo dispone de "
        "variables demográficas y epidemiológicas, sin datos clínicos (comorbilidades, saturación, etc.). "
        "Con la Precision indicada, una parte importante de las alertas son falsos positivos. "
        "Es una herramienta académica de apoyo; **no sustituye el criterio médico ni constituye un diagnóstico.**"
    )
    st.caption("Fuente de datos: Instituto Nacional de Salud (INS) · Datos Abiertos Colombia (datos.gov.co). "
               f"scikit-learn {meta['versiones']['scikit-learn']}.")


def main():
    st.set_page_config(page_title="Riesgo de mortalidad COVID-19", page_icon="🩺", layout="wide")
    st.title("🩺 Riesgo de fallecimiento por COVID-19 en Colombia")
    st.caption("Proyecto integrador · Aprendizaje de Máquinas · Metodología CRISP-DM")

    try:
        modelo, le_y, meta = cargar_artefactos()
    except Exception as exc:
        st.error(f"No fue posible cargar el modelo: {exc}")
        st.stop()

    t1, t2, t3 = st.tabs(["Caso individual", "Archivo de datos nuevos", "Acerca del modelo"])
    with t1:
        pestaña_individual(modelo, le_y, meta)
    with t2:
        pestaña_lote(modelo, le_y, meta)
    with t3:
        pestaña_modelo(meta)


if __name__ == "__main__":
    main()
