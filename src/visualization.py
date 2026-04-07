"""
visualization.py - Graficos de Evaluacion y Explicabilidad (Fase B.4)
======================================================================

Funciones para generar y guardar los graficos de evaluacion del modelo,
explicabilidad (SHAP) e interpretacion de resultados.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para guardar PNGs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier
import shap

from .config import OUTPUT_DIR

# ── Estilos globales ─────────────────────────────────────────────────
COLOR_PRINCIPAL = "#2563eb"
COLOR_SECUNDARIO = "#f59e0b"
COLOR_POSITIVO = "#ef4444"
COLOR_NEGATIVO = "#10b981"
FONT_TITLE = 14
FONT_LABEL = 12
DPI = 150


# =====================================================================
# 1. Curva ROC
# =====================================================================
def graficar_curva_roc(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Genera y guarda la Curva ROC con el valor del AUC anotado."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_val = roc_auc_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLOR_PRINCIPAL, lw=2.5,
            label=f"Modelo XGBoost (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5,
            linestyle="--", label="Azar (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLOR_PRINCIPAL)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=FONT_LABEL)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)", fontsize=FONT_LABEL)
    ax.set_title("Curva ROC - Prediccion de Pobreza Multidimensional\n"
                 "Encuesta de Hogares Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ruta = os.path.join(output_dir, "curva_roc.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Curva ROC guardada en: {ruta}")


# =====================================================================
# 2. SHAP Beeswarm
# =====================================================================
def graficar_shap_beeswarm(
    modelo: XGBClassifier,
    X_test: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> shap.TreeExplainer:
    """
    Genera el grafico SHAP beeswarm y retorna el explainer
    para reutilizarlo en otros graficos SHAP.
    """
    print("\n[...] Calculando valores SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="dot",
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Beeswarm - Impacto Global de Features\n"
              "Prediccion de Pobreza - EH Bolivia 2023",
              fontsize=FONT_TITLE, fontweight="bold", pad=15)
    plt.tight_layout()

    ruta = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"[OK] SHAP beeswarm guardado en: {ruta}")

    return explainer


# =====================================================================
# 3. Matriz de Confusion
# =====================================================================
def graficar_matriz_confusion(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Genera un heatmap de la matriz de confusion con anotaciones."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["No Pobre (0)", "Pobre (1)"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Colores: verde para aciertos, rojo para errores
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Anotaciones en cada celda
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Prediccion del Modelo", fontsize=FONT_LABEL)
    ax.set_ylabel("Valor Real", fontsize=FONT_LABEL)
    ax.set_title("Matriz de Confusion\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")

    ruta = os.path.join(output_dir, "matriz_confusion.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Matriz de confusion guardada en: {ruta}")


# =====================================================================
# 4. Importancia de Features (XGBoost nativa)
# =====================================================================
def graficar_importancia_features(
    modelo: XGBClassifier,
    X_test: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Genera un grafico de barras horizontales con la importancia de features."""
    importances = modelo.feature_importances_
    feature_names = X_test.columns
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
    bars = ax.barh(
        range(len(sorted_idx)),
        importances[sorted_idx],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(feature_names[sorted_idx], fontsize=10)
    ax.set_xlabel("Importancia (Gain)", fontsize=FONT_LABEL)
    ax.set_title("Importancia de Features - XGBoost\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Anotar porcentaje en cada barra
    total_imp = importances.sum()
    for bar_idx, (bar, feat_idx) in enumerate(zip(bars, sorted_idx)):
        pct = importances[feat_idx] / total_imp * 100
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9, color="#374151")

    ruta = os.path.join(output_dir, "importancia_features.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Importancia de features guardada en: {ruta}")


# =====================================================================
# 5. Distribucion de Probabilidades Predichas
# =====================================================================
def graficar_distribucion_probabilidades(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Histograma de probabilidades predichas separado por clase real."""
    fig, ax = plt.subplots(figsize=(9, 6))

    mask_pobre = y_test == 1
    mask_no_pobre = y_test == 0

    ax.hist(y_pred_proba[mask_no_pobre], bins=50, alpha=0.6,
            color=COLOR_NEGATIVO, label="No Pobre (real)", edgecolor="white")
    ax.hist(y_pred_proba[mask_pobre], bins=50, alpha=0.6,
            color=COLOR_POSITIVO, label="Pobre (real)", edgecolor="white")

    # Linea vertical en umbral 0.5
    ax.axvline(x=0.5, color="#1e293b", linestyle="--", linewidth=2,
               label="Umbral = 0.5")

    ax.set_xlabel("Probabilidad Predicha (Clase Pobre)", fontsize=FONT_LABEL)
    ax.set_ylabel("Frecuencia", fontsize=FONT_LABEL)
    ax.set_title("Distribucion de Probabilidades Predichas por Clase\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=11, loc="upper center")
    ax.grid(axis="y", alpha=0.3)

    ruta = os.path.join(output_dir, "distribucion_probabilidades.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Distribucion de probabilidades guardada en: {ruta}")


# =====================================================================
# 6. Curva Precision-Recall
# =====================================================================
def graficar_curva_precision_recall(
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Genera la curva Precision-Recall con el Average Precision anotado."""
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=COLOR_SECUNDARIO, lw=2.5,
            label=f"XGBoost (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color=COLOR_SECUNDARIO)

    # Linea base (tasa de prevalencia)
    prevalencia = y_test.mean()
    ax.axhline(y=prevalencia, color="#94a3b8", linestyle="--", lw=1.5,
               label=f"Azar (Prevalencia = {prevalencia:.2f})")

    ax.set_xlabel("Recall (Sensibilidad)", fontsize=FONT_LABEL)
    ax.set_ylabel("Precision", fontsize=FONT_LABEL)
    ax.set_title("Curva Precision-Recall\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    ruta = os.path.join(output_dir, "curva_precision_recall.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Curva Precision-Recall guardada en: {ruta}")


# =====================================================================
# 7. SHAP Bar Plot (Importancia media absoluta SHAP)
# =====================================================================
def graficar_shap_bar(
    modelo: XGBClassifier,
    X_test: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Grafico de barras con la media de |SHAP values| por feature."""
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.magma(np.linspace(0.25, 0.85, len(sorted_idx)))
    ax.barh(
        range(len(sorted_idx)),
        mean_abs_shap[sorted_idx],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(X_test.columns[sorted_idx], fontsize=10)
    ax.set_xlabel("Media de |SHAP value|", fontsize=FONT_LABEL)
    ax.set_title("Importancia Global SHAP (Media Absoluta)\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    ruta = os.path.join(output_dir, "shap_importancia_barras.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] SHAP bar plot guardado en: {ruta}")


# =====================================================================
# 8. SHAP Waterfall (explicacion de una prediccion individual)
# =====================================================================
def graficar_shap_waterfall(
    modelo: XGBClassifier,
    X_test: pd.DataFrame,
    idx: int = 0,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """
    Genera un SHAP waterfall plot para una observacion individual
    del set de test (por defecto, la primera).
    """
    explainer = shap.TreeExplainer(modelo)
    shap_values_obj = explainer(X_test)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(shap_values_obj[idx], show=False)
    plt.title(f"SHAP Waterfall - Explicacion Prediccion Individual (obs #{idx})\n"
              "Prediccion de Pobreza - EH Bolivia 2023",
              fontsize=FONT_TITLE, fontweight="bold", pad=15)
    plt.tight_layout()

    ruta = os.path.join(output_dir, "shap_waterfall.png")
    plt.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close("all")
    print(f"[OK] SHAP waterfall guardado en: {ruta}")


# =====================================================================
# 9. Distribucion de Variable Objetivo (Pobre vs No Pobre)
# =====================================================================
def graficar_distribucion_target(
    y: pd.Series,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Grafico de barras con la distribucion de la variable objetivo."""
    counts = y.value_counts().sort_index()
    labels = ["No Pobre (0)", "Pobre (1)"]
    colors_bar = [COLOR_NEGATIVO, COLOR_POSITIVO]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors_bar,
                  edgecolor="white", linewidth=1.5, width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Cantidad de Hogares", fontsize=FONT_LABEL)
    ax.set_title("Distribucion de la Variable Objetivo\n"
                 "Hogares Pobres vs No Pobres - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, counts.max() * 1.2)

    ruta = os.path.join(output_dir, "distribucion_target.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Distribucion del target guardada en: {ruta}")


# =====================================================================
# 10. Comparacion Validacion Cruzada vs Test
# =====================================================================
def graficar_comparacion_cv_test(
    auc_cv: float,
    auc_test: float,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Compara el AUC de validacion cruzada vs. test en un bar chart."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Validacion Cruzada\n(5-Fold)", "Conjunto de\nPrueba"]
    values = [auc_cv, auc_test]
    colors_bar = [COLOR_PRINCIPAL, COLOR_SECUNDARIO]

    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white",
                  linewidth=1.5, width=0.45)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    ax.set_ylabel("AUC-ROC", fontsize=FONT_LABEL)
    ax.set_title("Comparacion AUC-ROC: Validacion Cruzada vs Test\n"
                 "Deteccion de Sobreajuste",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Zona de referencia (AUC = 0.5)
    ax.axhline(y=0.5, color="#94a3b8", linestyle="--", lw=1.5,
               label="Azar (AUC = 0.5)")
    ax.legend(fontsize=10)

    ruta = os.path.join(output_dir, "comparacion_cv_test.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Comparacion CV vs Test guardada en: {ruta}")


# =====================================================================
# 11. Correlacion entre Features Numericas
# =====================================================================
def graficar_correlacion_features(
    X: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Heatmap de correlacion de Pearson entre features numericas."""
    # Seleccionar solo columnas numericas continuas (no dummies)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    corr = X[num_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlacion de Pearson")

    n = len(num_cols)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(num_cols, fontsize=9)

    # Anotar valores
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("Matriz de Correlacion de Pearson entre Features\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold")

    ruta = os.path.join(output_dir, "correlacion_features.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Correlacion de features guardada en: {ruta}")


# =====================================================================
# 12. Boxplot de Features Numericas por Clase
# =====================================================================
def graficar_boxplot_por_clase(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Boxplots de features numericas continuas segmentados por clase."""
    num_cols = ["hacinamiento", "anios_educ_jefe", "indice_equipamiento"]
    available = [c for c in num_cols if c in X.columns]

    if not available:
        print("[!] No se encontraron features numericas para boxplot")
        return

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        data_0 = X.loc[y == 0, col]
        data_1 = X.loc[y == 1, col]

        bp = ax.boxplot(
            [data_0, data_1],
            labels=["No Pobre", "Pobre"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="#1e293b", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(COLOR_NEGATIVO)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(COLOR_POSITIVO)
        bp["boxes"][1].set_alpha(0.6)

        ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribucion de Features Numericas por Clase\n"
                 "Prediccion de Pobreza - EH Bolivia 2023",
                 fontsize=FONT_TITLE, fontweight="bold", y=1.02)

    ruta = os.path.join(output_dir, "boxplot_por_clase.png")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Boxplots por clase guardados en: {ruta}")
