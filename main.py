"""
main.py - Orquestador del Pipeline
====================================
Prediccion de Pobreza Multidimensional en Bolivia
Microdatos de la Encuesta de Hogares 2023 (BOL-INE-EH-2023)

Este script orquesta la ejecucion secuencial de todas las fases:

  FASE A: Ingenieria de Datos (Data Preparation)
    A.1 - Carga y fusion de microdatos
    A.2 - Feature engineering
    A.3 - Seleccion de variables y One-Hot Encoding
    A.4 - Separacion train/test y balanceo SMOTE

  FASE B: Modelado y Explicabilidad (Modeling & XAI)
    B.1 - Instanciacion del modelo XGBoost
    B.2 - Validacion cruzada estratificada
    B.3 - Entrenamiento final y evaluacion
    B.4 - Graficos ROC y SHAP beeswarm
    B.5 - Serializacion del modelo entrenado (.pkl)

Uso:
    python main.py
"""

import sys
import io
import warnings
import joblib

# Forzar UTF-8 en stdout para evitar errores de codificacion en Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -- Importaciones del paquete src --
from src.config import DATA_DIR, OUTPUT_DIR, MODEL_PATH
from src.data_loading import (
    cargar_datos_sav,
    construir_indice_equipamiento,
    extraer_info_jefe_hogar,
    fusionar_datasets,
)
from src.feature_engineering import construir_features, seleccionar_y_codificar
from src.preprocessing import separar_y_balancear
from src.modeling import crear_modelo_xgb, validacion_cruzada, entrenar_y_evaluar
from src.visualization import graficar_curva_roc, graficar_shap_beeswarm


def main():
    """Orquesta la ejecucion completa del pipeline de prediccion de pobreza."""

    print("=" * 60)
    print("  PREDICCION DE POBREZA MULTIDIMENSIONAL - BOLIVIA 2023")
    print("  Encuesta de Hogares (BOL-INE-EH-2023)")
    print("=" * 60)

    # ==================================================================
    #  FASE A: INGENIERIA DE DATOS
    # ==================================================================
    print(f"\n{'='*60}")
    print("  FASE A: INGENIERIA DE DATOS")
    print(f"{'='*60}")

    # A.1 Carga de microdatos
    print("\n>> A.1 Cargando microdatos desde archivos .sav ...")
    df_vivienda, df_equipamiento, df_persona = cargar_datos_sav(DATA_DIR)

    # A.1 Indice de equipamiento
    print("\n>> A.1 Construyendo indice de equipamiento ...")
    df_equip_agg = construir_indice_equipamiento(df_equipamiento)
    print(f"  Hogares con equipamiento: {df_equip_agg.shape[0]:,}")

    # A.1 Informacion del jefe de hogar
    print("\n>> A.1 Extrayendo informacion del jefe de hogar ...")
    df_jefe = extraer_info_jefe_hogar(df_persona)
    print(f"  Jefes de hogar identificados: {df_jefe.shape[0]:,}")

    # A.1 Fusion
    print("\n>> A.1 Fusionando datasets por llave 'folio' ...")
    df = fusionar_datasets(df_vivienda, df_equip_agg, df_jefe)

    # A.2 Feature Engineering
    print("\n>> A.2 Construyendo features predictoras ...")
    df = construir_features(df)

    # A.3 Codificacion
    print("\n>> A.3 Seleccionando variables y aplicando One-Hot Encoding ...")
    X, y = seleccionar_y_codificar(df)

    # A.4 Separacion y SMOTE
    print("\n>> A.4 Separacion Train/Test + Balanceo SMOTE ...")
    X_train, X_test, y_train, y_test = separar_y_balancear(X, y)

    # ==================================================================
    #  FASE B: MODELADO Y EXPLICABILIDAD
    # ==================================================================
    print(f"\n{'='*60}")
    print("  FASE B: MODELADO Y EXPLICABILIDAD")
    print(f"{'='*60}")

    # B.1 Modelo
    print("\n>> B.1 Instanciando modelo XGBoost ...")
    modelo = crear_modelo_xgb()

    # B.2 Validacion cruzada
    print("\n>> B.2 Validacion cruzada estratificada ...")
    auc_cv = validacion_cruzada(modelo, X_train, y_train)

    # B.3 Entrenamiento final y evaluacion
    print("\n>> B.3 Entrenamiento final y evaluacion ...")
    modelo, y_pred_proba = entrenar_y_evaluar(
        modelo, X_train, y_train, X_test, y_test
    )

    # B.4 Graficos
    print("\n>> B.4 Generando graficos ...")
    graficar_curva_roc(y_test, y_pred_proba)
    graficar_shap_beeswarm(modelo, X_test)

    # B.5 Serializar modelo entrenado
    print("\n>> B.5 Guardando modelo entrenado ...")
    joblib.dump(modelo, MODEL_PATH)
    print(f"[OK] Modelo serializado en: {MODEL_PATH}")

    # ==================================================================
    #  RESUMEN FINAL
    # ==================================================================
    print(f"\n{'='*60}")
    print("  [OK] PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"  Resultados guardados en: {OUTPUT_DIR}")
    print(f"  Modelo listo para API:   {MODEL_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
