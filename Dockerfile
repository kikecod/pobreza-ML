# Usa una imagen oficial y ligera de Python
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el codigo de la API y utilidades del modelo
COPY api/ /app/api/
COPY src/ /app/src/
COPY output/ /app/output/
COPY frontend/ /app/frontend/

# Exponer el puerto
EXPOSE 8000

# Comando para arrancar FastAPI con uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
