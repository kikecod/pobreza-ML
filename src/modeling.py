"""
modeling.py - Modelo XGBoost, Validacion y Evaluacion (Fase B.1, B.2, B.3)
==========================================================================

Funciones para instanciar el clasificador XGBoost, ejecutar la
validacion cruzada estratificada y evaluar en el conjunto de prueba.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from .config import XGB_PARAMS, CV_SPLITS, SEED


def crear_modelo_xgb() -> XGBClassifier:
    """
    Instancia un clasificador XGBClassifier con hiperparametros
    robustos disenados para evitar sobreajuste en datos de encuestas:

    - objective       : binary:logistic (clasificacion binaria)
    - n_estimators    : 300 arboles
    - max_depth       : 6 (controla complejidad del arbol)
    - learning_rate   : 0.05 (tasa de aprendizaje conservadora)
    - subsample       : 0.8 (muestreo de filas por arbol)
    - colsample_bytree: 0.8 (muestreo de columnas por arbol)
    - reg_alpha       : 0.1 (regularizacion L1)
    - reg_lambda      : 1.0 (regularizacion L2)
    - min_child_weight: 5 (peso minimo de nodos hoja)
    - eval_metric     : logloss

    Returns
    -------
    XGBClassifier configurado (sin entrenar).
    """
    modelo = XGBClassifier(**XGB_PARAMS)

    print(f"\n[OK] Modelo XGBClassifier instanciado")
    print(f"  objective       : {XGB_PARAMS['objective']}")
    print(f"  n_estimators    : {XGB_PARAMS['n_estimators']}")
    print(f"  max_depth       : {XGB_PARAMS['max_depth']}")
    print(f"  learning_rate   : {XGB_PARAMS['learning_rate']}")
    print(f"  subsample       : {XGB_PARAMS['subsample']}")
    print(f"  colsample_bytree: {XGB_PARAMS['colsample_bytree']}")
    return modelo


def validacion_cruzada(
    modelo: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_SPLITS,
) -> float:
    """
    Ejecuta una validacion cruzada estratificada (Stratified K-Fold)
    con 5 splits, calculando el AUC-ROC promedio.

    Parameters
    ----------
    modelo  : XGBClassifier (sin entrenar).
    X_train : Features de entrenamiento (post-SMOTE).
    y_train : Target de entrenamiento (post-SMOTE).
    n_splits: Numero de folds.

    Returns
    -------
    auc_promedio : AUC-ROC promedio de la validacion cruzada.
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


def entrenar_y_evaluar(
    modelo: XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[XGBClassifier, np.ndarray]:
    """
    Entrena el modelo con los datos sinteticos (post-SMOTE) y evalua
    en el conjunto de prueba original.

    Imprime el reporte de clasificacion priorizando el Recall, ya que
    el costo de un falso negativo (excluir a un hogar que SI es pobre
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
