# MLflow tracking server + model registry. Multi-stage build (Mod3 Lecture 1):
# dependencies go into a venv in the builder, and only that venv ships.

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
    HOME=/tmp \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY --from=builder /opt/venv /opt/venv
# The store (mlflow.db + mlartifacts/) is bind-mounted at /project by docker-compose,
# so the container runs as the host user (compose `user:`) to keep file ownership.
WORKDIR /project
EXPOSE 5000
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", \
     "--artifacts-destination", "./mlartifacts", "--host", "0.0.0.0", "--port", "5000"]
