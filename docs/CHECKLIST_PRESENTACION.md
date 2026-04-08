# Checklist de presentación / entrega

## 1) Antes de presentar (5–10 min)

- [ ] Verifica que el entrenamiento corre: `python main.py` (al menos una vez)
- [ ] Confirma que existen:
  - [ ] `output/modelo_xgb.pkl`
  - [ ] `output/metrics.json`
  - [ ] `output/curva_roc.png` (y otros gráficos)
- [ ] Levanta el servidor: `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`
- [ ] Abre y revisa que cargan sin errores:
  - [ ] `/predictor`
  - [ ] `/graficos`
  - [ ] `/mlops`
  - [ ] `/docs`

Si vas a presentar con Docker:
- [ ] `docker build -t pobreza-ml .`
- [ ] `docker run --rm -p 8000:8000 pobreza-ml`

---

## 2) Guión sugerido de demo (3–6 min)

1) **Home** (`/`) y **Swagger** (`/docs`)
   - Menciona que el backend es FastAPI y expone endpoints de predicción y MLOps.

2) **Predictor** (`/predictor`)
   - Ingresa un ejemplo de hogar y muestra la predicción + probabilidades.

3) **Gráficos** (`/graficos`)
   - Enseña curva ROC y matriz de confusión (y SHAP si lo deseas).

4) **Panel MLOps** (`/mlops`)
   - Señala:
     - métricas (`auc_cv`, `auc_test`)
     - versión del modelo (hash SHA256 del artefacto)
     - runs detectados (MLflow local si existe `mlruns/`)

5) (Opcional) **Reentrenamiento desde la web**
   - Pulsa “Entrenar / Reentrenar” y muestra:
     - el estado “running”
     - el tail de logs
     - al finalizar, que cambia la versión (hash) / métricas

> Tip: si el entrenamiento demora, deja el proceso ya ejecutado antes y solo muestras el panel con logs/métricas existentes.

---

## 3) Qué entregar junto al informe

Recomendado:
- `docs/proyecto.md` (informe)
- `README.md` (quickstart)
- `docs/COMO_CORRER.md` (guía detallada)
- (Opcional) `docs/cicd_aws_ecs.md` si vas a explicar el deploy a AWS ECS

Si tu entrega NO puede incluir el dataset:
- agrega un apartado en el informe indicando cómo obtener EH 2023 y dónde colocar los `.sav` (en `BD_EH2023/`).
- incluye `output/modelo_xgb.pkl` + `output/metrics.json` para poder correr la app sin reentrenar.

---

## 4) Nota sobre CI/CD y quality gate

El workflow de GitHub Actions valida un umbral de AUC (por defecto 0.84). Si las métricas actuales (en `output/metrics.json`) están por debajo, el deploy se bloqueará.

Para presentación puedes:
- mantener el umbral como “objetivo” y explicar que bloquea despliegues si el modelo no alcanza la calidad, o
- ajustar el umbral a un valor acorde a tu resultado actual (si tu docente lo permite).
