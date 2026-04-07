"""
===========================================================================
Predicción de Pobreza Multidimensional en Bolivia
Microdatos de la Encuesta de Hogares 2023 (BOL-INE-EH-2023)
===========================================================================

Script modular que implementa las fases de:
  A) Preparación de Datos (Data Preparation)
  B) Modelado y Explicabilidad (Modeling & XAI)

Autor : Proyecto ML - Encuesta de Hogares 2023
Fecha : Abril 2026
Python : 3.10+
Librerías principales: pandas, scikit-learn, xgboost, imblearn, shap
===========================================================================
"""

# ─────────────────────────────────────────────────────────────────────────
# 0. IMPORTACIONES
# ─────────────────────────────────────────────────────────────────────────
import os
import sys
import io
import warnings
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para guardar PNGs
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Forzar UTF-8 en stdout para evitar errores de codificacion en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────────
# Configuración de rutas
# ─────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BD_EH2023")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════
# FASE A: INGENIERÍA DE DATOS (DATA PREPARATION)
# ═════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# A.1  Carga y Fusión de Microdatos
# ─────────────────────────────────────────────────────────────────────────
def cargar_datos_sav(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los tres archivos .sav (SPSS) de la Encuesta de Hogares 2023:
      - EH2023_Vivienda.sav
      - EH2023_Equipamiento.sav
      - EH2023_Persona.sav

    Returns
    -------
    df_vivienda, df_equipamiento, df_persona : DataFrames crudos.
    """
    rutas = {
        "vivienda": os.path.join(data_dir, "EH2023_Vivienda.sav"),
        "equipamiento": os.path.join(data_dir, "EH2023_Equipamiento.sav"),
        "persona": os.path.join(data_dir, "EH2023_Persona.sav"),
    }
    for nombre, ruta in rutas.items():
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Archivo no encontrado: {ruta}")

    df_vivienda, _ = pyreadstat.read_sav(rutas["vivienda"])
    df_equipamiento, _ = pyreadstat.read_sav(rutas["equipamiento"])
    df_persona, _ = pyreadstat.read_sav(rutas["persona"])

    print(f"[OK] Vivienda     : {df_vivienda.shape[0]:>6,} registros, {df_vivienda.shape[1]:>3} columnas")
    print(f"[OK] Equipamiento : {df_equipamiento.shape[0]:>6,} registros, {df_equipamiento.shape[1]:>3} columnas")
    print(f"[OK] Persona      : {df_persona.shape[0]:>6,} registros, {df_persona.shape[1]:>3} columnas")
    return df_vivienda, df_equipamiento, df_persona


def construir_indice_equipamiento(df_equip: pd.DataFrame) -> pd.DataFrame:
    """
    Pivotea la tabla de equipamiento (formato largo) a nivel hogar,
    creando una columna binaria por cada bien duradero y calculando
    el **índice de equipamiento** como la suma normalizada (0-1).

    Bienes duraderos (17 ítems): cocina, refrigerador, TV, computadora,
    celular, lavadora, motocicleta, automóvil, etc.

    Parameters
    ----------
    df_equip : DataFrame con columnas [folio, item, s08b_1].

    Returns
    -------
    DataFrame a nivel hogar con columna `indice_equipamiento`.
    """
    # s08b_1: 1 = Sí posee, 2 = No posee → convertir a binario
    df = df_equip[["folio", "item", "s08b_1"]].copy()
    df["tiene_bien"] = (df["s08b_1"] == 1).astype(int)

    # Pivotar: una fila por hogar, una columna por ítem
    pivot = df.pivot_table(
        index="folio",
        columns="item",
        values="tiene_bien",
        aggfunc="max",
        fill_value=0,
    )
    pivot.columns = [f"equip_{int(c)}" for c in pivot.columns]

    # Índice de equipamiento = suma normalizada [0, 1]
    n_items = pivot.shape[1]
    pivot["indice_equipamiento"] = pivot.sum(axis=1) / n_items

    return pivot.reset_index()


def extraer_info_jefe_hogar(df_persona: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra al jefe/jefa del hogar (s01a_05 == 1) y extrae:
      - anios_educ_jefe  : años de escolaridad del jefe (variable `aestudio`).
      - afiliacion_afp   : 1 si aporta actualmente a la Gestora/AFP
                           (s04f_36 == 1), 0 en caso contrario.
      - p0 (target)      : indicador de pobreza por ingreso (0/1).

    Parameters
    ----------
    df_persona : DataFrame completo del archivo Persona.

    Returns
    -------
    DataFrame a nivel hogar (un registro por folio) con las columnas
    derivadas y la variable objetivo.
    """
    # Filtrar solo al jefe del hogar
    jefes = df_persona[df_persona["s01a_05"] == 1].copy()

    # Años de educación del jefe
    jefes["anios_educ_jefe"] = jefes["aestudio"].fillna(0).astype(float)

    # Afiliación AFP: 1 si actualmente aporta (s04f_36==1), 0 caso contrario
    jefes["afiliacion_afp"] = (jefes["s04f_36"] == 1).astype(int)

    # Variable objetivo: pobreza por ingreso (ya viene como 0/1)
    jefes["target_pobreza"] = jefes["p0"].astype(int)

    cols_salida = ["folio", "anios_educ_jefe", "afiliacion_afp", "target_pobreza"]
    return jefes[cols_salida].copy()


def fusionar_datasets(
    df_vivienda: pd.DataFrame,
    df_equipamiento_agg: pd.DataFrame,
    df_jefe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fusiona los tres datasets a nivel hogar mediante la llave `folio`.

    Parameters
    ----------
    df_vivienda       : Datos de vivienda (una fila por hogar).
    df_equipamiento_agg : Índice de equipamiento agregado por hogar.
    df_jefe           : Información extraída del jefe de hogar.

    Returns
    -------
    DataFrame fusionado con todas las variables de vivienda, equipamiento
    y persona (jefe) alineadas por `folio`.
    """
    df = df_vivienda.merge(df_equipamiento_agg, on="folio", how="inner")
    df = df.merge(df_jefe, on="folio", how="inner")
    print(f"\n[OK] Dataset fusionado: {df.shape[0]:>6,} hogares, {df.shape[1]:>3} columnas")
    return df


# ─────────────────────────────────────────────────────────────────────────
# A.2  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────
def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las variables predictoras clave:

    1. **hacinamiento**: ratio entre total de miembros del hogar (`totper`)
       y número de habitaciones para dormir (`s06a_17`).
    2. **material_vivienda**: material de construcción de las paredes
       (`s06a_03`), tratado como categórica.
    3. **area**: urbana (1) / rural (2), tratada como categórica.

    (Las variables `anios_educ_jefe`, `indice_equipamiento` y
     `afiliacion_afp` ya fueron creadas en pasos anteriores.)

    Parameters
    ----------
    df : DataFrame fusionado.

    Returns
    -------
    DataFrame con las nuevas features añadidas.
    """
    df = df.copy()

    # Hacinamiento: total de personas / habitaciones para dormir
    # Evitar división por cero: si s06a_17 == 0 → NaN → imputar con mediana
    dormitorios = df["s06a_17"].replace(0, np.nan)
    df["hacinamiento"] = df["totper"] / dormitorios
    mediana_hacinamiento = df["hacinamiento"].median()
    df["hacinamiento"] = df["hacinamiento"].fillna(mediana_hacinamiento)

    # Material de vivienda (paredes) como categórica con etiquetas legibles
    material_map = {
        1: "ladrillo_bloque_cemento_hormigon",
        2: "adobe_tapial",
        3: "tabique_quinche",
        4: "piedra",
        5: "madera",
        6: "cana_palma_tronco",
        7: "otro",
    }
    df["material_vivienda"] = (
        df["s06a_03"]
        .map(material_map)
        .fillna("otro")
        .astype("category")
    )

    # Área: 1=Urbana, 2=Rural → categórica legible
    area_map = {1: "urbana", 2: "rural"}
    df["area"] = (
        df["area"]
        .map(area_map)
        .fillna("urbana")
        .astype("category")
    )

    print(f"[OK] Features construidas: hacinamiento, material_vivienda, area")
    return df


# ─────────────────────────────────────────────────────────────────────────
# A.3  Selección de Variables y One-Hot Encoding
# ─────────────────────────────────────────────────────────────────────────
def seleccionar_y_codificar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Selecciona las variables predictoras finales y aplica
    One-Hot Encoding a las categóricas (drop_first=True).

    Variables numéricas:
      - hacinamiento
      - anios_educ_jefe
      - indice_equipamiento
      - afiliacion_afp

    Variables categóricas (One-Hot Encoding):
      - material_vivienda (7 categorías → 6 dummies)
      - area              (2 categorías → 1 dummy)

    Returns
    -------
    X : DataFrame de features codificadas.
    y : Series con la variable objetivo (target_pobreza).
    """
    features_num = [
        "hacinamiento",
        "anios_educ_jefe",
        "indice_equipamiento",
        "afiliacion_afp",
    ]
    features_cat = ["material_vivienda", "area"]
    target = "target_pobreza"

    # Verificar que no haya NaN residuales en numéricas
    for col in features_num:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"  [!] {col}: {n_nan} valores nulos imputados con mediana ({mediana:.2f})")

    X = df[features_num + features_cat].copy()
    y = df[target].copy()

    # One-Hot Encoding con drop_first=True
    X = pd.get_dummies(X, columns=features_cat, drop_first=True, dtype=int)

    print(f"\n[OK] Features finales ({X.shape[1]}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\n[OK] Variable objetivo: {target}")
    print(f"  Distribución:\n{y.value_counts().to_string()}")
    print(f"  Tasa de pobreza: {y.mean():.2%}")


    return X, y


# ─────────────────────────────────────────────────────────────────────────
# A.4  Separación y Balanceo de Clases (SMOTE)
# ─────────────────────────────────────────────────────────────────────────
def separar_y_balancear(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    smote_strategy: float = 0.8,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    1. Divide los datos en entrenamiento (80%) y prueba (20%) con
       estratificación sobre `target_pobreza`.
    2. Aplica SMOTE (sampling_strategy=0.8) **solo al conjunto de
       entrenamiento** para corregir el desequilibrio de clases.

    Parameters
    ----------
    X             : Features codificadas.
    y             : Variable objetivo.
    test_size     : Proporción del conjunto de prueba.
    smote_strategy: Ratio deseado de la clase minoritaria / mayoritaria.
    random_state  : Semilla de reproducibilidad.

    Returns
    -------
    X_train_res : Features de entrenamiento balanceadas (con SMOTE).
    X_test      : Features de prueba (datos originales, sin alterar).
    y_train_res : Objetivo de entrenamiento balanceado.
    y_test      : Objetivo de prueba original.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    print(f"\n{'-'*60}")
    print(f"SEPARACION TRAIN / TEST (estratificada)")
    print(f"{'-'*60}")
    print(f"  Entrenamiento : {X_train.shape[0]:>6,} muestras")
    print(f"  Prueba        : {X_test.shape[0]:>6,} muestras")
    print(f"  Train target  :\n{y_train.value_counts().to_string()}")

    # SMOTE solo sobre entrenamiento
    smote = SMOTE(
        sampling_strategy=smote_strategy,
        random_state=random_state,
        k_neighbors=5,
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"\n[OK] SMOTE aplicado (sampling_strategy={smote_strategy})")
    print(f"  Train original : {X_train.shape[0]:>6,} muestras")
    print(f"  Train SMOTE    : {X_train_res.shape[0]:>6,} muestras")
    print(f"  Distribucion post-SMOTE:\n{y_train_res.value_counts().to_string()}")

    return X_train_res, X_test, y_train_res, y_test


# ═════════════════════════════════════════════════════════════════════════
# FASE B: MODELADO Y EXPLICABILIDAD (MODELING & XAI)
# ═════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# B.1  Entrenamiento del Modelo XGBoost
# ─────────────────────────────────────────────────────────────────────────
def crear_modelo_xgb(random_state: int = SEED) -> XGBClassifier:
    """
    Instancia un clasificador XGBClassifier con hiperparámetros
    robustos diseñados para evitar sobreajuste en datos de encuestas:

    - objective       : binary:logistic (clasificación binaria)
    - n_estimators    : 300 árboles
    - max_depth       : 6 (controla complejidad del árbol)
    - learning_rate   : 0.05 (tasa de aprendizaje conservadora)
    - subsample       : 0.8 (muestreo de filas por árbol)
    - colsample_bytree: 0.8 (muestreo de columnas por árbol)
    - reg_alpha       : 0.1 (regularización L1)
    - reg_lambda      : 1.0 (regularización L2)
    - min_child_weight: 5 (peso mínimo de nodos hoja)
    - eval_metric     : logloss

    Returns
    -------
    XGBClassifier configurado (sin entrenar).
    """
    modelo = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )
    print(f"\n[OK] Modelo XGBClassifier instanciado")
    print(f"  objective       : binary:logistic")
    print(f"  n_estimators    : 300")
    print(f"  max_depth       : 6")
    print(f"  learning_rate   : 0.05")
    print(f"  subsample       : 0.8")
    print(f"  colsample_bytree: 0.8")
    return modelo


# ─────────────────────────────────────────────────────────────────────────
# B.2  Validación Cruzada Estratificada
# ─────────────────────────────────────────────────────────────────────────
def validacion_cruzada(
    modelo: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> float:
    """
    Ejecuta una validación cruzada estratificada (Stratified K-Fold)
    con 5 splits, calculando el AUC-ROC promedio.

    Parameters
    ----------
    modelo  : XGBClassifier (sin entrenar).
    X_train : Features de entrenamiento (post-SMOTE).
    y_train : Target de entrenamiento (post-SMOTE).
    n_splits: Número de folds.

    Returns
    -------
    auc_promedio : AUC-ROC promedio de la validación cruzada.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print(f"\n{'-'*60}")
    print(f"VALIDACION CRUZADA ESTRATIFICADA ({n_splits}-Fold)")
    print(f"{'-'*60}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: AUC-ROC = {s:.4f}")
    print(f"  {'-'*40}")
    print(f"  Promedio: AUC-ROC = {scores.mean():.4f} +/- {scores.std():.4f}")

    return scores.mean()


# ─────────────────────────────────────────────────────────────────────────
# B.3  Entrenamiento Final y Evaluación
# ─────────────────────────────────────────────────────────────────────────
def entrenar_y_evaluar(
    modelo: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[XGBClassifier, np.ndarray]:
    """
    Entrena el modelo con los datos sintéticos (post-SMOTE) y evalúa
    en el conjunto de prueba original.

    Imprime el reporte de clasificación priorizando el Recall, ya que
    el costo de un falso negativo (excluir a un hogar que SÍ es pobre
    de un programa social) es significativamente mayor que el de un
    falso positivo.

    Returns
    -------
    modelo         : XGBClassifier entrenado.
    y_pred_proba   : Probabilidades predichas para la clase positiva.
    """
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]

    auc_test = roc_auc_score(y_test, y_pred_proba)

    print(f"\n{'='*60}")
    print(f"EVALUACION EN CONJUNTO DE PRUEBA")
    print(f"{'='*60}")
    print(f"\n  AUC-ROC en Test: {auc_test:.4f}")
    print(f"\n  Reporte de Clasificacion:")
    print(f"  (Priorizando RECALL: el costo de excluir a un hogar")
    print(f"   pobre (Falso Negativo) es criticamente alto)\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Pobre (0)", "Pobre (1)"],
        digits=4,
    ))

    return modelo, y_pred_proba


# ─────────────────────────────────────────────────────────────────────────
# B.4  Gráficos: Curva ROC y SHAP Beeswarm
# ─────────────────────────────────────────────────────────────────────────
def graficar_curva_roc(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Genera y guarda la Curva ROC con el valor del AUC anotado.

    Parameters
    ----------
    y_test       : Valores reales del test.
    y_pred_proba : Probabilidades predichas (clase positiva).
    output_dir   : Directorio de salida para el PNG.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_val = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5,
            label=f"Modelo XGBoost (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5,
            linestyle="--", label="Azar (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2563eb")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=12)
    ax.set_title("Curva ROC - Prediccion de Pobreza Multidimensional\n"
                 "Encuesta de Hogares Bolivia 2023",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ruta = os.path.join(output_dir, "curva_roc.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Curva ROC guardada en: {ruta}")


def graficar_shap_beeswarm(
    modelo: XGBClassifier,
    X_test: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Genera el gráfico de explicabilidad global SHAP tipo beeswarm
    (shap.summary_plot) para interpretar el impacto causal de cada
    característica en las predicciones del modelo.

    Utiliza TreeExplainer (optimizado para modelos basados en árboles).

    Parameters
    ----------
    modelo     : XGBClassifier entrenado.
    X_test     : Features del conjunto de prueba.
    output_dir : Directorio de salida para el PNG.
    """
    print("\n[...] Calculando valores SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",    # beeswarm
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Beeswarm - Impacto Global de Features\n"
              "Prediccion de Pobreza - EH Bolivia 2023",
              fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    ruta = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[OK] SHAP beeswarm guardado en: {ruta}")


# ═════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════
def main():
    """Ejecuta el pipeline completo de predicción de pobreza."""

    print("=" * 60)
    print("  PREDICCION DE POBREZA MULTIDIMENSIONAL - BOLIVIA 2023")
    print("  Encuesta de Hogares (BOL-INE-EH-2023)")
    print("=" * 60)

    # -- FASE A -------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FASE A: INGENIERIA DE DATOS")
    print(f"{'='*60}")

    # A.1 Carga
    print("\n>> A.1 Cargando microdatos desde archivos .sav ...")
    df_vivienda, df_equipamiento, df_persona = cargar_datos_sav(DATA_DIR)

    # A.1 Procesamiento previo
    print("\n>> A.1 Construyendo indice de equipamiento ...")
    df_equip_agg = construir_indice_equipamiento(df_equipamiento)
    print(f"  Hogares con equipamiento: {df_equip_agg.shape[0]:,}")

    print("\n>> A.1 Extrayendo informacion del jefe de hogar ...")
    df_jefe = extraer_info_jefe_hogar(df_persona)
    print(f"  Jefes de hogar identificados: {df_jefe.shape[0]:,}")

    # A.1 Fusión
    print("\n>> A.1 Fusionando datasets por llave 'folio' ...")
    df = fusionar_datasets(df_vivienda, df_equip_agg, df_jefe)

    # A.2 Feature Engineering
    print("\n>> A.2 Construyendo features predictoras ...")
    df = construir_features(df)

    # A.3 Codificación
    print("\n>> A.3 Seleccionando variables y aplicando One-Hot Encoding ...")
    X, y = seleccionar_y_codificar(df)

    # A.4 Separación y SMOTE
    print("\n>> A.4 Separacion Train/Test + Balanceo SMOTE ...")
    X_train, X_test, y_train, y_test = separar_y_balancear(X, y)

    # -- FASE B -------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FASE B: MODELADO Y EXPLICABILIDAD")
    print(f"{'='*60}")

    # B.1 Modelo
    print("\n>> B.1 Instanciando modelo XGBoost ...")
    modelo = crear_modelo_xgb()

    # B.2 Validación cruzada
    print("\n>> B.2 Validacion cruzada estratificada ...")
    auc_cv = validacion_cruzada(modelo, X_train, y_train)

    # B.3 Entrenamiento final y evaluación
    print("\n>> B.3 Entrenamiento final y evaluacion ...")
    modelo, y_pred_proba = entrenar_y_evaluar(
        modelo, X_train, y_train, X_test, y_test
    )

    # B.4 Gráficos
    print("\n>> B.4 Generando graficos ...")
    graficar_curva_roc(y_test, y_pred_proba)
    graficar_shap_beeswarm(modelo, X_test)

    print(f"\n{'='*60}")
    print("  [OK] PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"  Resultados guardados en: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
