import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_server_url: str
    tracking_uri: str
    feedback_dir: Path
    max_upload_bytes: int
    model_info_ttl: float
    model_name: str
    model_alias: str
    request_timeout: float


def load_settings():
    return Settings(
        model_server_url=os.getenv("MODEL_SERVER_URL", "http://localhost:5001").rstrip(
            "/"
        ),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        feedback_dir=Path(os.getenv("FEEDBACK_DIR", "feedback")),
        max_upload_bytes=int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024),
        model_info_ttl=float(os.getenv("MODEL_INFO_TTL", "60")),
        model_name=os.getenv("MODEL_NAME", "malariascope"),
        model_alias=os.getenv("MODEL_ALIAS", "champion"),
        request_timeout=float(os.getenv("MODEL_REQUEST_TIMEOUT", "10")),
    )
