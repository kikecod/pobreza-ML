"""
main.py - API REST para Prediccion de Pobreza
===============================================
API de produccion construida con FastAPI para servir el modelo
XGBoost de prediccion de pobreza estructural en Bolivia.

Endpoints:
    GET  /            -> Health check e info de la API
    GET  /salud       -> Estado del modelo cargado
    POST /predecir    -> Prediccion de pobreza para un hogar

Uso:
    cd api
    uvicorn main:app --reload --port 8000

    O directamente:
    python main.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ruta al modelo serializado
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_xgb.pkl")

# Variable global para el modelo cargado
modelo = None

# ---------------------------------------------------------------------------
# Nombres de las features (en el MISMO orden que el entrenamiento)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "hacinamiento",
    "anios_educ_jefe",
    "indice_equipamiento",
    "afiliacion_afp",
    "material_vivienda_cana_palma_tronco",
    "material_vivienda_ladrillo_bloque_cemento_hormigon",
    "material_vivienda_madera",
    "material_vivienda_otro",
    "material_vivienda_piedra",
    "material_vivienda_tabique_quinche",
    "area_urbana",
]


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicacion: cargar modelo al iniciar
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo XGBoost al arrancar la API."""
    global modelo
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Modelo no encontrado en: {MODEL_PATH}")
        print("        Ejecuta primero: python main.py (pipeline de entrenamiento)")
        sys.exit(1)

    modelo = joblib.load(MODEL_PATH)
    print(f"[OK] Modelo XGBoost cargado desde: {MODEL_PATH}")
    print(f"     Features esperadas: {len(FEATURE_NAMES)}")
    yield
    # Cleanup al apagar
    modelo = None
    print("[OK] Modelo descargado de memoria")


# ---------------------------------------------------------------------------
# Inicializacion de FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API Prediccion de Pobreza - Bolivia EH-2023",
    description=(
        "API REST para prediccion de pobreza estructural en hogares bolivianos. "
        "Utiliza un modelo XGBoost entrenado con los microdatos de la "
        "Encuesta de Hogares 2023 (BOL-INE-EH-2023) del INE Bolivia."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware CORS - permitir todos los origenes para frontend de prueba
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# MODELOS DE DATOS (Pydantic)
# ===========================================================================

class DatosHogar(BaseModel):
    """
    Estructura del JSON que envia el frontend con los datos del hogar
    a evaluar. Cada campo corresponde a una feature del modelo XGBoost.
    """

    # --- Variables numericas principales ---
    hacinamiento: float = Field(
        ...,
        ge=0,
        description=(
            "Indice de hacinamiento: ratio entre el total de miembros "
            "del hogar y el numero de habitaciones exclusivas para dormir. "
            "Ejemplo: 5 personas / 2 dormitorios = 2.5"
        ),
        examples=[2.5],
    )

    anios_educ_jefe: int = Field(
        ...,
        ge=0,
        le=25,
        description=(
            "Anios de escolaridad del jefe/jefa del hogar. "
            "0 = sin educacion formal, 12 = bachillerato, 17 = licenciatura."
        ),
        examples=[11],
    )

    indice_equipamiento: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Indice normalizado (0-1) de bienes duraderos del hogar. "
            "Proporcion de los 17 bienes relevados que posee el hogar: "
            "cocina, refrigerador, TV, computadora, celular, lavadora, etc. "
            "0.0 = ningun bien, 1.0 = todos los bienes."
        ),
        examples=[0.47],
    )

    afiliacion_afp: int = Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Afiliacion a sistema de pensiones (Gestora/AFP). "
            "1 = el jefe de hogar aporta actualmente, "
            "0 = no aporta."
        ),
        examples=[0],
    )

    # --- Material de vivienda (One-Hot Encoded, drop_first=True) ---
    # La categoria base (drop_first) es: adobe_tapial
    material_vivienda_cana_palma_tronco: int = Field(
        default=0, ge=0, le=1,
        description="1 si las paredes son de cana, palma o tronco.",
    )
    material_vivienda_ladrillo_bloque_cemento_hormigon: int = Field(
        default=0, ge=0, le=1,
        description="1 si las paredes son de ladrillo, bloque, cemento u hormigon.",
    )
    material_vivienda_madera: int = Field(
        default=0, ge=0, le=1,
        description="1 si las paredes son de madera.",
    )
    material_vivienda_otro: int = Field(
        default=0, ge=0, le=1,
        description="1 si el material es otro no clasificado.",
    )
    material_vivienda_piedra: int = Field(
        default=0, ge=0, le=1,
        description="1 si las paredes son de piedra.",
    )
    material_vivienda_tabique_quinche: int = Field(
        default=0, ge=0, le=1,
        description="1 si las paredes son de tabique o quinche.",
    )

    # --- Area urbana/rural (One-Hot Encoded, drop_first=True) ---
    # La categoria base (drop_first) es: rural
    area_urbana: int = Field(
        default=0, ge=0, le=1,
        description="1 si el hogar esta en zona urbana, 0 si es rural.",
        examples=[1],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "hacinamiento": 3.5,
                    "anios_educ_jefe": 5,
                    "indice_equipamiento": 0.29,
                    "afiliacion_afp": 0,
                    "material_vivienda_cana_palma_tronco": 0,
                    "material_vivienda_ladrillo_bloque_cemento_hormigon": 0,
                    "material_vivienda_madera": 0,
                    "material_vivienda_otro": 0,
                    "material_vivienda_piedra": 0,
                    "material_vivienda_tabique_quinche": 0,
                    "area_urbana": 0,
                }
            ]
        }
    }


class ResultadoPrediccion(BaseModel):
    """Respuesta JSON devuelta por el endpoint /predecir."""
    prediccion: str = Field(
        ...,
        description="Clasificacion final: 'Pobre' o 'No Pobre'.",
    )
    probabilidad_pobreza: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad estimada de que el hogar sea pobre (0.0 a 1.0).",
    )
    probabilidad_no_pobreza: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad estimada de que el hogar NO sea pobre (0.0 a 1.0).",
    )
    confianza: str = Field(
        ...,
        description="Nivel de confianza cualitativo: Alta, Media o Baja.",
    )


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.get("/", tags=["General"])
async def raiz():
    """Health check e informacion general de la API."""
    return {
        "api": "Prediccion de Pobreza Estructural - Bolivia",
        "version": "1.0.0",
        "modelo": "XGBoost (binary:logistic)",
        "dataset": "Encuesta de Hogares 2023 (BOL-INE-EH-2023)",
        "estado": "activo" if modelo is not None else "sin modelo",
        "endpoints": {
            "GET /": "Informacion de la API",
            "GET /salud": "Estado del modelo",
            "POST /predecir": "Prediccion de pobreza para un hogar",
            "GET /docs": "Documentacion interactiva (Swagger UI)",
        },
    }


@app.get("/salud", tags=["General"])
async def salud():
    """Verifica que el modelo esta cargado y operativo."""
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Reinicie el servidor.",
        )
    return {
        "estado": "operativo",
        "modelo_cargado": True,
        "features_esperadas": len(FEATURE_NAMES),
        "tipo_modelo": type(modelo).__name__,
    }


@app.post("/predecir", response_model=ResultadoPrediccion, tags=["Prediccion"])
async def predecir(datos: DatosHogar):
    """
    Recibe los datos estructurales de un hogar boliviano y retorna
    la prediccion de pobreza junto con la probabilidad matematica.

    El modelo prioriza el **Recall** (sensibilidad), ya que el costo
    de un falso negativo (excluir a un hogar pobre de un programa
    social) es criticamente alto.
    """
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Reinicie el servidor.",
        )

    try:
        # Convertir los datos de entrada a un DataFrame con las columnas
        # en el MISMO orden que el modelo fue entrenado
        valores = [
            datos.hacinamiento,
            datos.anios_educ_jefe,
            datos.indice_equipamiento,
            datos.afiliacion_afp,
            datos.material_vivienda_cana_palma_tronco,
            datos.material_vivienda_ladrillo_bloque_cemento_hormigon,
            datos.material_vivienda_madera,
            datos.material_vivienda_otro,
            datos.material_vivienda_piedra,
            datos.material_vivienda_tabique_quinche,
            datos.area_urbana,
        ]

        df_input = pd.DataFrame([valores], columns=FEATURE_NAMES)

        # Prediccion binaria y probabilidades
        prediccion = modelo.predict(df_input)[0]
        probabilidades = modelo.predict_proba(df_input)[0]

        prob_no_pobre = float(probabilidades[0])
        prob_pobre = float(probabilidades[1])
        etiqueta = "Pobre" if prediccion == 1 else "No Pobre"

        # Nivel de confianza basado en la probabilidad maxima
        prob_max = max(prob_pobre, prob_no_pobre)
        if prob_max >= 0.80:
            confianza = "Alta"
        elif prob_max >= 0.60:
            confianza = "Media"
        else:
            confianza = "Baja"

        return ResultadoPrediccion(
            prediccion=etiqueta,
            probabilidad_pobreza=round(prob_pobre, 4),
            probabilidad_no_pobreza=round(prob_no_pobre, 4),
            confianza=confianza,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la prediccion: {str(e)}",
        )


# ===========================================================================
# EJECUCION CON UVICORN
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  API PREDICCION DE POBREZA - BOLIVIA EH-2023")
    print("  Iniciando servidor Uvicorn en http://127.0.0.1:8000")
    print("  Documentacion: http://127.0.0.1:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
