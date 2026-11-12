# the model is not baked into the image, serve.py looks up @champion at start
# and again every 15 s

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
CMD ["python", "serve.py"]
