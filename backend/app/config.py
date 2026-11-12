import os
from pathlib import Path

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:5001").rstrip("/")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
FEEDBACK_DIR = Path(os.getenv("FEEDBACK_DIR", "feedback"))
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
MODEL_INFO_TTL = float(os.getenv("MODEL_INFO_TTL", "60"))
MODEL_NAME = os.getenv("MODEL_NAME", "malariascope")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
REQUEST_TIMEOUT = float(os.getenv("MODEL_REQUEST_TIMEOUT", "10"))
