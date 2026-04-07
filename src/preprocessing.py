"""
preprocessing.py - Separacion y Balanceo de Clases (Fase A.4)
=============================================================

Funcion para dividir los datos en train/test con estratificacion
y aplicar SMOTE al conjunto de entrenamiento.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from .config import TEST_SIZE, SMOTE_STRATEGY, SMOTE_K_NEIGHBORS, SEED


def separar_y_balancear(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    smote_strategy: float = SMOTE_STRATEGY,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    1. Divide los datos en entrenamiento (80%) y prueba (20%) con
       estratificacion sobre `target_pobreza`.
    2. Aplica SMOTE (sampling_strategy=0.8) **solo al conjunto de
       entrenamiento** para corregir el desequilibrio de clases.

    Parameters
    ----------
    X             : Features codificadas.
    y             : Variable objetivo.
    test_size     : Proporcion del conjunto de prueba.
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
        k_neighbors=SMOTE_K_NEIGHBORS,
    )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"\n[OK] SMOTE aplicado (sampling_strategy={smote_strategy})")
    print(f"  Train original : {X_train.shape[0]:>6,} muestras")
    print(f"  Train SMOTE    : {X_train_res.shape[0]:>6,} muestras")
    print(f"  Distribucion post-SMOTE:\n{y_train_res.value_counts().to_string()}")

    return X_train_res, X_test, y_train_res, y_test
