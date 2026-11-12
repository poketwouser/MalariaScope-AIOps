# build context is the repo root: the backend reuses image.py and labels.py from src/

# Stage 1: builder
FROM python:3.10 AS builder
COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: runtime
FROM python:3.10-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src:/app/backend \
    HOME=/tmp \
    FEEDBACK_DIR=/data/feedback
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY src/malariascope/__init__.py src/malariascope/image.py src/malariascope/labels.py \
     ./src/malariascope/
COPY backend/app ./backend/app
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /data/feedback && chown -R appuser:appuser /data
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
