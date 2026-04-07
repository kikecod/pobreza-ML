"""
visualization.py - Graficos ROC y SHAP Beeswarm (Fase B.4)
===========================================================

Funciones para generar y guardar los graficos de evaluacion
y explicabilidad del modelo.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para guardar PNGs
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
import shap

from .config import OUTPUT_DIR


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
    Genera el grafico de explicabilidad global SHAP tipo beeswarm
    (shap.summary_plot) para interpretar el impacto causal de cada
    caracteristica en las predicciones del modelo.

    Utiliza TreeExplainer (optimizado para modelos basados en arboles).

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
