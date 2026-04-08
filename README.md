# Pobreza-ML (EH 2023 Bolivia) — XGBoost + FastAPI + MLOps

Sistema de predicción de pobreza (etiqueta `target_pobreza`, derivada de `p0`) usando microdatos de la **Encuesta de Hogares 2023** (INE Bolivia). Incluye:

- **Pipeline de entrenamiento**: `main.py` (carga `.sav`, feature engineering, SMOTE, XGBoost, SHAP, gráficos, export de métricas y modelo)
- **API REST + Web**: FastAPI en `api/main.py` (sirve también el frontend estático)
- **Panel MLOps**: `/mlops` para ver métricas, versión (hash) y runs locales; botón para entrenar/reentrenar desde la web
- **Tracking**: MLflow local (carpeta `mlruns/`)
- **Contenerización**: `Dockerfile`
- **CI/CD (opcional)**: GitHub Actions para build/deploy (ver `docs/cicd_aws_ecs.md`)

---

## Requisitos

- Python 3.10+
- (Opcional) Docker
- Dataset `.sav` en `BD_EH2023/` (incluido en este repo)

---

## Quickstart (Local)

### 1) Crear entorno e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Entrenar / reentrenar (genera modelo, métricas y gráficos)

```bash
python main.py
```

Artefactos esperados:
- `output/modelo_xgb.pkl`
- `output/metrics.json`
- `output/*.png` (gráficos)
- `mlruns/` (si se ejecuta MLflow tracking)

### 3) Levantar la web + API

Desde la raíz del proyecto:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Abrir en el navegador:
- Home: http://127.0.0.1:8000/
- Predictor: http://127.0.0.1:8000/predictor
- Gráficos: http://127.0.0.1:8000/graficos
- MLOps: http://127.0.0.1:8000/mlops
- Swagger: http://127.0.0.1:8000/docs

---

## Entrenar desde la Web (botón)

En http://127.0.0.1:8000/mlops puedes lanzar **Entrenar / Reentrenar**.

Notas:
- El entrenamiento ejecuta `python main.py` en segundo plano y escribe logs en `output/training.log`.
- Requiere que exista `BD_EH2023/` (los `.sav`) en el filesystem donde corre la app.

---

## Docker (demo rápida)

### Build

```bash
docker build -t pobreza-ml .
```

### Run (serving + frontend)

```bash
docker run --rm -p 8000:8000 pobreza-ml
```

Abrir: http://127.0.0.1:8000/

### Run con reentrenamiento habilitado (montando dataset y persistiendo artefactos)

> Recomendado para demo: persiste `output/` y `mlruns/` y permite que el botón de entrenar funcione.

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/BD_EH2023:/app/BD_EH2023" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/mlruns:/app/mlruns" \
  pobreza-ml
```

---

## CI/CD (AWS ECS)

- Workflow: `.github/workflows/mlops.yml`
- Guía: `docs/cicd_aws_ecs.md`

El workflow valida un **umbral de AUC** (quality gate). Si tu `output/metrics.json` no cumple el umbral, el deploy se bloquea.

---

## Estructura del repo (resumen)

- `main.py`: pipeline completo de entrenamiento/evaluación
- `src/`: carga de datos, features, preprocessing, modelado, visualización
- `api/main.py`: FastAPI + endpoints de predicción y MLOps + serving del frontend
- `frontend/`: HTML/CSS/JS
- `output/`: artefactos del modelo, métricas y gráficos
- `mlruns/`: tracking local MLflow

---

## Presentación

Para una guía de demo y checklist, ver `docs/CHECKLIST_PRESENTACION.md` y `docs/COMO_CORRER.md`.
