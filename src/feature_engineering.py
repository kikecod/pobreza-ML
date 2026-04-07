"""
feature_engineering.py - Ingenieria de Features (Fase A.2 y A.3)
================================================================

Funciones para construir las variables predictoras y aplicar
la codificacion (One-Hot Encoding) necesaria para el modelo.
"""

import numpy as np
import pandas as pd

from .config import FEATURES_NUM, FEATURES_CAT, TARGET, MATERIAL_MAP, AREA_MAP


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las variables predictoras clave:

    1. **hacinamiento**: ratio entre total de miembros del hogar (`totper`)
       y numero de habitaciones para dormir (`s06a_17`).
    2. **material_vivienda**: material de construccion de las paredes
       (`s06a_03`), tratado como categorica.
    3. **area**: urbana (1) / rural (2), tratada como categorica.

    (Las variables `anios_educ_jefe`, `indice_equipamiento` y
     `afiliacion_afp` ya fueron creadas en la fase de carga.)

    Parameters
    ----------
    df : DataFrame fusionado.

    Returns
    -------
    DataFrame con las nuevas features aniadidas.
    """
    df = df.copy()

    # Hacinamiento: total de personas / habitaciones para dormir
    # Evitar division por cero: si s06a_17 == 0 -> NaN -> imputar con mediana
    dormitorios = df["s06a_17"].replace(0, np.nan)
    df["hacinamiento"] = df["totper"] / dormitorios
    mediana_hacinamiento = df["hacinamiento"].median()
    df["hacinamiento"] = df["hacinamiento"].fillna(mediana_hacinamiento)

    # Material de vivienda (paredes) como categorica con etiquetas legibles
    df["material_vivienda"] = (
        df["s06a_03"]
        .map(MATERIAL_MAP)
        .fillna("otro")
        .astype("category")
    )

    # Area: 1=Urbana, 2=Rural -> categorica legible
    df["area"] = (
        df["area"]
        .map(AREA_MAP)
        .fillna("urbana")
        .astype("category")
    )

    print(f"[OK] Features construidas: hacinamiento, material_vivienda, area")
    return df


def seleccionar_y_codificar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Selecciona las variables predictoras finales y aplica
    One-Hot Encoding a las categoricas (drop_first=True).

    Variables numericas:
      - hacinamiento
      - anios_educ_jefe
      - indice_equipamiento
      - afiliacion_afp

    Variables categoricas (One-Hot Encoding):
      - material_vivienda (7 categorias -> 6 dummies)
      - area              (2 categorias -> 1 dummy)

    Returns
    -------
    X : DataFrame de features codificadas.
    y : Series con la variable objetivo (target_pobreza).
    """
    # Verificar que no haya NaN residuales en numericas
    for col in FEATURES_NUM:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            print(f"  [!] {col}: {n_nan} valores nulos imputados con mediana ({mediana:.2f})")

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].copy()

    # One-Hot Encoding con drop_first=True
    X = pd.get_dummies(X, columns=FEATURES_CAT, drop_first=True, dtype=int)

    print(f"\n[OK] Features finales ({X.shape[1]}):")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\n[OK] Variable objetivo: {TARGET}")
    print(f"  Distribucion:\n{y.value_counts().to_string()}")
    print(f"  Tasa de pobreza: {y.mean():.2%}")

    return X, y
