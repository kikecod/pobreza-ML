"""
config.py - Configuracion global del proyecto
==============================================

Centraliza todas las constantes, rutas de archivos y parametros
de reproducibilidad del pipeline de prediccion de pobreza.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
# Directorio raiz del proyecto (un nivel arriba de /src)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "BD_EH2023")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "modelo_xgb.pkl")

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Parametros del pipeline
# ---------------------------------------------------------------------------
# A.4 - Separacion y SMOTE
TEST_SIZE = 0.20
SMOTE_STRATEGY = 0.8
SMOTE_K_NEIGHBORS = 5

# B.1 - Hiperparametros del modelo XGBoost
XGB_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "random_state": SEED,
    "n_jobs": -1,
}

# B.2 - Validacion cruzada
CV_SPLITS = 5

# ---------------------------------------------------------------------------
# Variables del dataset EH-2023
# ---------------------------------------------------------------------------
# Features numericas finales
FEATURES_NUM = [
    "hacinamiento",
    "anios_educ_jefe",
    "indice_equipamiento",
    "afiliacion_afp",
]

# Features categoricas (se aplicara One-Hot Encoding)
FEATURES_CAT = [
    "material_vivienda",
    "area",
]

# Variable objetivo
TARGET = "target_pobreza"

# Mapeo de material de vivienda (s06a_03)
MATERIAL_MAP = {
    1: "ladrillo_bloque_cemento_hormigon",
    2: "adobe_tapial",
    3: "tabique_quinche",
    4: "piedra",
    5: "madera",
    6: "cana_palma_tronco",
    7: "otro",
}

# Mapeo de area (1=Urbana, 2=Rural)
AREA_MAP = {1: "urbana", 2: "rural"}
