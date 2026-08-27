FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

EXPOSE 8000

# Limit BLAS threads to prevent CPU oversubscription with multiple workers.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# `exec` forwards SIGTERM directly to uvicorn.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${APP_WORKERS:-4}"]