# Cómo ejecutar el proyecto (Local y Docker)

Esta guía sirve para correr el proyecto de forma reproducible y preparar una demo.

## 0) Requisitos

- Python 3.10+ (recomendado: 3.10)
- Dataset de EH 2023 en `BD_EH2023/` (archivos `.sav`)
- (Opcional) Docker

---

## 1) Ejecutar local (recomendado para demo completa)

### 1.1 Crear entorno e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.2 Entrenar / reentrenar (pipeline completo)

Ejecuta el pipeline desde la raíz del repo:

```bash
python main.py
```

Se generan artefactos y gráficos en `output/` y (si corresponde) tracking en `mlruns/`.

### 1.3 Levantar la web + API

Desde la raíz del repo:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

URLs útiles:
- Home: http://127.0.0.1:8000/
- Predictor: http://127.0.0.1:8000/predictor
- Gráficos: http://127.0.0.1:8000/graficos
- Panel MLOps: http://127.0.0.1:8000/mlops
- Swagger: http://127.0.0.1:8000/docs

### 1.4 Entrenar desde la web

En http://127.0.0.1:8000/mlops:
- Botón **Entrenar / Reentrenar** inicia el proceso en segundo plano.
- Logs: `output/training.log` (la UI muestra un “tail”).

Notas:
- Requiere que `BD_EH2023/` exista y contenga los `.sav`.
- El proceso puede tardar varios minutos dependiendo de la máquina.

---

## 2) Ejecutar con Docker

### 2.1 Build

Desde la raíz del repo:

```bash
docker build -t pobreza-ml .
```

### 2.2 Run (serving + frontend)

```bash
docker run --rm -p 8000:8000 pobreza-ml
```

Abrir: http://127.0.0.1:8000/

> Nota: para que el panel de gráficos y el panel MLOps muestren datos, la imagen incluye los artefactos actuales de `output/`.

### 2.3 Run con reentrenamiento habilitado (recomendado)

Para que el botón de entrenar funcione dentro del contenedor:
- monta el dataset (`BD_EH2023/`)
- persiste `output/` y `mlruns/` para ver cambios de versión/métricas

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/BD_EH2023:/app/BD_EH2023" \
  -v "$PWD/output:/app/output" \
  -v "$PWD/mlruns:/app/mlruns" \
  pobreza-ml
```

---

## 3) (Opcional) Ver MLflow UI

Si quieres abrir la interfaz de MLflow en local:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Abrir: http://127.0.0.1:5000/

---

## 4) Troubleshooting rápido

### Error: “Modelo no encontrado” al iniciar la API
- Solución: ejecuta primero `python main.py` para generar `output/modelo_xgb.pkl`.

### Error al entrenar desde la web
- Verifica que exista `BD_EH2023/` con los `.sav`.
- Revisa `output/training.log`.

### El puerto 8000 está ocupado
- Cambia el puerto en uvicorn, por ejemplo `--port 8001`, o detén el proceso que usa el 8000.
