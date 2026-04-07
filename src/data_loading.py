"""
data_loading.py - Carga y Fusion de Microdatos (Fase A.1)
=========================================================

Funciones para cargar los archivos .sav (SPSS) de la Encuesta
de Hogares 2023 y fusionarlos a nivel hogar usando la llave `folio`.
"""

import os
import numpy as np
import pandas as pd
import pyreadstat


def cargar_datos_sav(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los tres archivos .sav (SPSS) de la Encuesta de Hogares 2023:
      - EH2023_Vivienda.sav
      - EH2023_Equipamiento.sav
      - EH2023_Persona.sav

    Parameters
    ----------
    data_dir : Ruta al directorio que contiene los archivos .sav.

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
    el indice de equipamiento como la suma normalizada (0-1).

    Bienes duraderos (17 items): cocina, refrigerador, TV, computadora,
    celular, lavadora, motocicleta, automovil, etc.

    Parameters
    ----------
    df_equip : DataFrame con columnas [folio, item, s08b_1].

    Returns
    -------
    DataFrame a nivel hogar con columna `indice_equipamiento`.
    """
    # s08b_1: 1 = Si posee, 2 = No posee -> convertir a binario
    df = df_equip[["folio", "item", "s08b_1"]].copy()
    df["tiene_bien"] = (df["s08b_1"] == 1).astype(int)

    # Pivotar: una fila por hogar, una columna por item
    pivot = df.pivot_table(
        index="folio",
        columns="item",
        values="tiene_bien",
        aggfunc="max",
        fill_value=0,
    )
    pivot.columns = [f"equip_{int(c)}" for c in pivot.columns]

    # Indice de equipamiento = suma normalizada [0, 1]
    n_items = pivot.shape[1]
    pivot["indice_equipamiento"] = pivot.sum(axis=1) / n_items

    return pivot.reset_index()


def extraer_info_jefe_hogar(df_persona: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra al jefe/jefa del hogar (s01a_05 == 1) y extrae:
      - anios_educ_jefe  : anios de escolaridad del jefe (variable `aestudio`).
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

    # Anios de educacion del jefe
    jefes["anios_educ_jefe"] = jefes["aestudio"].fillna(0).astype(float)

    # Afiliacion AFP: 1 si actualmente aporta (s04f_36==1), 0 caso contrario
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
    df_vivienda         : Datos de vivienda (una fila por hogar).
    df_equipamiento_agg : Indice de equipamiento agregado por hogar.
    df_jefe             : Informacion extraida del jefe de hogar.

    Returns
    -------
    DataFrame fusionado con todas las variables alineadas por `folio`.
    """
    df = df_vivienda.merge(df_equipamiento_agg, on="folio", how="inner")
    df = df.merge(df_jefe, on="folio", how="inner")
    print(f"\n[OK] Dataset fusionado: {df.shape[0]:>6,} hogares, {df.shape[1]:>3} columnas")
    return df
