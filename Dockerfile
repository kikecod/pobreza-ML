# Usa una imagen oficial y ligera de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Dependencias del sistema (necesarias para XGBoost/OpenMP en slim)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

# Copia los archivos de requerimientos e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el codigo de la API y utilidades del modelo
COPY api/ /app/api/
COPY src/ /app/src/
COPY main.py /app/main.py
COPY output/ /app/output/
COPY frontend/ /app/frontend/

# Exponer el puerto
EXPOSE 8000

# Comando para arrancar FastAPI con uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
