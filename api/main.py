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
import json
import hashlib
import subprocess
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "output")
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_xgb.pkl")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")
MLRUNS_DIR = os.path.join(PROJECT_DIR, "mlruns")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
TRAINING_LOG_PATH = os.path.join(MODEL_DIR, "training.log")

# Variable global para el modelo cargado
modelo = None
training_process: subprocess.Popen | None = None
training_started_at: str | None = None
training_finished_at: str | None = None
training_exit_code: int | None = None

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


def _safe_read_json(path: str) -> dict:
    """Lee un JSON de forma segura; retorna diccionario vacio si falla."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _iso_from_timestamp(timestamp: float | None) -> str | None:
    """Convierte timestamp unix a fecha ISO UTC."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _model_sha256(path: str) -> str | None:
    """Calcula SHA256 del artefacto del modelo para trazar versiones."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_meta_yaml(path: str) -> dict:
    """Parser simple para meta.yaml de MLflow sin dependencias extra."""
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip().strip("\"").strip("'")
    except Exception:
        return {}
    return data


def _read_run_metric(metric_file: str) -> float | None:
    """Lee el valor mas reciente de una metrica de MLflow."""
    try:
        with open(metric_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return None
        # Formato habitual: <timestamp> <value> <step>
        parts = lines[-1].split()
        if len(parts) >= 2:
            return float(parts[1])
    except Exception:
        return None
    return None


def _collect_mlruns(limit: int = 5) -> list[dict]:
    """Recolecta runs recientes del tracking local de MLflow (si existe)."""
    if not os.path.isdir(MLRUNS_DIR):
        return []

    runs: list[dict] = []
    for exp_id in os.listdir(MLRUNS_DIR):
        exp_dir = os.path.join(MLRUNS_DIR, exp_id)
        if not os.path.isdir(exp_dir):
            continue
        for run_id in os.listdir(exp_dir):
            run_dir = os.path.join(exp_dir, run_id)
            meta_path = os.path.join(run_dir, "meta.yaml")
            if not os.path.exists(meta_path):
                continue

            meta = _parse_meta_yaml(meta_path)
            metrics_dir = os.path.join(run_dir, "metrics")
            auc_test = None
            auc_cv = None
            if os.path.isdir(metrics_dir):
                auc_test = _read_run_metric(os.path.join(metrics_dir, "auc_test"))
                auc_cv = _read_run_metric(os.path.join(metrics_dir, "auc_cv"))

            start_ms_raw = meta.get("start_time")
            start_ms = int(start_ms_raw) if start_ms_raw and start_ms_raw.isdigit() else None
            start_ts = (start_ms / 1000) if start_ms is not None else None

            runs.append({
                "run_id": meta.get("run_id", run_id),
                "run_name": meta.get("run_name", "sin_nombre"),
                "experiment_id": meta.get("experiment_id", exp_id),
                "status": meta.get("status", "UNKNOWN"),
                "start_time": _iso_from_timestamp(start_ts),
                "auc_test": round(auc_test, 4) if auc_test is not None else None,
                "auc_cv": round(auc_cv, 4) if auc_cv is not None else None,
            })

    runs.sort(key=lambda r: r.get("start_time") or "", reverse=True)
    return runs[:limit]


def _build_mlops_status() -> dict:
    """Construye estado MLOps para la vista web."""
    metrics = _safe_read_json(METRICS_PATH)
    model_exists = os.path.exists(MODEL_PATH)
    model_stat = os.stat(MODEL_PATH) if model_exists else None
    model_hash = _model_sha256(MODEL_PATH) if model_exists else None
    runs = _collect_mlruns(limit=5)

    return {
        "api_version": app.version,
        "mlflow_tracking": {
            "tracking_uri": MLRUNS_DIR,
            "enabled": os.path.isdir(MLRUNS_DIR),
            "runs_detectados": len(runs),
            "runs": runs,
        },
        "modelo_activo": {
            "path": MODEL_PATH,
            "existe": model_exists,
            "tamano_bytes": model_stat.st_size if model_stat else None,
            "actualizado_utc": _iso_from_timestamp(model_stat.st_mtime) if model_stat else None,
            "sha256": model_hash,
            "version_corta": model_hash[:12] if model_hash else None,
        },
        "metricas": {
            "auc_test": metrics.get("auc_test"),
            "auc_cv": metrics.get("auc_cv"),
            "threshold_auc": metrics.get("threshold_auc", 0.84),
        },
    }


def _tail_file(path: str, max_lines: int = 60) -> list[str]:
    """Retorna ultimas lineas de un archivo de log sin fallar."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\n") for ln in lines[-max_lines:]]
    except Exception:
        return []


def _training_status_payload() -> dict:
    """Construye estado actual del proceso de entrenamiento en background."""
    global training_process, training_exit_code, training_finished_at

    running = False
    pid = None
    if training_process is not None:
        pid = training_process.pid
        poll_code = training_process.poll()
        if poll_code is None:
            running = True
        else:
            training_exit_code = poll_code
            if training_finished_at is None:
                training_finished_at = datetime.now(timezone.utc).isoformat()

    return {
        "running": running,
        "pid": pid,
        "started_at": training_started_at,
        "finished_at": training_finished_at,
        "exit_code": training_exit_code,
        "log_path": TRAINING_LOG_PATH,
        "log_tail": _tail_file(TRAINING_LOG_PATH, max_lines=60),
    }


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

# ---------------------------------------------------------------------------
# Servir archivos estaticos (graficos y frontend)
# ---------------------------------------------------------------------------
app.mount("/output", StaticFiles(directory=MODEL_DIR), name="output")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


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
# API ENDPOINTS
# ===========================================================================

@app.get("/api", tags=["General"])
async def raiz():
    """Health check e informacion general de la API."""
    return {
        "api": "Prediccion de Pobreza Estructural - Bolivia",
        "version": "1.0.0",
        "modelo": "XGBoost (binary:logistic)",
        "dataset": "Encuesta de Hogares 2023 (BOL-INE-EH-2023)",
        "estado": "activo" if modelo is not None else "sin modelo",
        "endpoints": {
            "GET /api": "Informacion de la API",
            "GET /api/salud": "Estado del modelo",
            "POST /api/predecir": "Prediccion de pobreza para un hogar",
            "GET /api/mlops/status": "Estado de versionado y metricas MLOps",
            "GET /api/mlops/runs": "Runs recientes de MLflow local",
            "POST /api/mlops/train": "Inicia entrenamiento/reentrenamiento",
            "GET /api/mlops/train/status": "Estado y logs de entrenamiento",
            "GET /mlops": "Panel web de versionado MLOps",
            "GET /docs": "Documentacion interactiva (Swagger UI)",
        },
    }


@app.get("/api/salud", tags=["General"])
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


@app.get("/api/mlops/status", tags=["MLOps"])
async def mlops_status():
    """Estado de versionado del modelo, metricas y runs locales de MLflow."""
    return _build_mlops_status()


@app.get("/api/mlops/runs", tags=["MLOps"])
async def mlops_runs(limit: int = 10):
    """Lista de runs detectados en tracking local MLflow (mlruns/)."""
    safe_limit = max(1, min(limit, 50))
    return {
        "count": safe_limit,
        "runs": _collect_mlruns(limit=safe_limit),
    }


@app.post("/api/mlops/train", tags=["MLOps"])
async def mlops_train():
    """Lanza entrenamiento/reentrenamiento en segundo plano."""
    global training_process, training_started_at, training_finished_at, training_exit_code

    # Evitar entrenamiento concurrente
    if training_process is not None and training_process.poll() is None:
        return {
            "ok": False,
            "message": "Ya existe un entrenamiento en ejecucion.",
            **_training_status_payload(),
        }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TRAINING_LOG_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("[INFO] Iniciando entrenamiento desde API...\n")

    cmd = [sys.executable, os.path.join(PROJECT_DIR, "main.py")]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_stream = open(TRAINING_LOG_PATH, "a", encoding="utf-8")
    training_process = subprocess.Popen(
        cmd,
        cwd=PROJECT_DIR,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        env=env,
    )

    training_started_at = datetime.now(timezone.utc).isoformat()
    training_finished_at = None
    training_exit_code = None

    return {
        "ok": True,
        "message": "Entrenamiento/reentrenamiento iniciado.",
        **_training_status_payload(),
    }


@app.get("/api/mlops/train/status", tags=["MLOps"])
async def mlops_train_status():
    """Estado de entrenamiento en background y cola de logs."""
    return _training_status_payload()


@app.post("/api/predecir", response_model=ResultadoPrediccion, tags=["Prediccion"])
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
# FRONTEND ROUTES - Serve HTML pages
# ===========================================================================

@app.get("/", tags=["Frontend"])
async def serve_home():
    """Sirve la pagina principal del frontend."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/informe", tags=["Frontend"])
async def serve_informe():
    """Sirve la pagina del informe."""
    return FileResponse(os.path.join(FRONTEND_DIR, "informe.html"))


@app.get("/graficos", tags=["Frontend"])
async def serve_graficos():
    """Sirve la pagina de graficos."""
    return FileResponse(os.path.join(FRONTEND_DIR, "graficos.html"))


@app.get("/predictor", tags=["Frontend"])
async def serve_predictor():
    """Sirve la pagina del predictor."""
    return FileResponse(os.path.join(FRONTEND_DIR, "predictor.html"))


@app.get("/mlops", tags=["Frontend"])
async def serve_mlops():
    """Sirve la pagina web de versionado MLOps."""
    return FileResponse(os.path.join(FRONTEND_DIR, "mlops.html"))


# ===========================================================================
# EJECUCION CON UVICORN
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  API PREDICCION DE POBREZA - BOLIVIA EH-2023")
    print("  Iniciando servidor Uvicorn en http://127.0.0.1:8000")
    print("  Frontend:       http://127.0.0.1:8000")
    print("  Documentacion:  http://127.0.0.1:8000/docs")
    print("=" * 60)

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
