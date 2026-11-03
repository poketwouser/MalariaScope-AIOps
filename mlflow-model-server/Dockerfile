# Serves the registered model with `mlflow models serve` (Lecture 10: MLflow serving).
# The model is not baked in: serve.py resolves the registry alias and restarts the server
# whenever the alias moves, so a promotion or rollback takes effect on its own.

# Stage 1: builder
FROM python:3.10 AS builder
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: runtime
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=malariascope \
    MODEL_ALIAS=champion
COPY --from=builder /opt/venv /opt/venv
RUN useradd --create-home --uid 10001 appuser
USER appuser
WORKDIR /home/appuser
COPY serve.py .
EXPOSE 5001
# serve.py runs `mlflow models serve --env-manager local`: the image already holds
# the model's pinned requirements.
CMD ["python", "serve.py"]
