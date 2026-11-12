"""MalariaScope backend: the REST API the Streamlit UI calls."""

import io
import logging
import os
import time
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app import config
from app.feedback_store import FeedbackStore
from app.model_client import ModelClient, ModelUnavailable
from app.schemas import (
    ErrorResponse,
    FeedbackResponse,
    FeedbackStats,
    Health,
    ModelInfo,
    PredictResponse,
    Ready,
)
from malariascope.image import to_model_input
from malariascope.labels import CLASS_NAMES

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("backend")

ALLOWED_FORMATS = {"PNG", "JPEG"}
ERROR_RESPONSES = {code: {"model": ErrorResponse} for code in (400, 413, 422, 500, 503)}


class APIError(Exception):
    def __init__(self, status, error, detail):
        super().__init__(detail)
        self.status, self.error, self.detail = status, error, detail


def error_json(request, status, error, detail):
    metrics.ERRORS.labels(error=error).inc()
    body = {"error": error, "detail": detail, "request_id": request.state.request_id}
    return JSONResponse(status_code=status, content=body)


def read_image(upload, max_bytes):
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise APIError(
            413,
            "file_too_large",
            f"file is larger than {max_bytes // (1024 * 1024)} MB",
        )
    if not data:
        raise APIError(400, "invalid_image", "the uploaded file is empty")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise APIError(400, "invalid_image", "the file is not a readable image")
    if img.format not in ALLOWED_FORMATS:
        raise APIError(
            400, "invalid_image", f"only PNG and JPEG are accepted, got {img.format}"
        )
    return img, len(data)


app = FastAPI(
    title="MalariaScope API",
    version="1.0",
    description="Malaria parasite detection for single blood cell images.",
)
# created once at start-up, not per request
model = ModelClient()
store = FeedbackStore(config.FEEDBACK_DIR)


@app.middleware("http")
async def timing(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    metrics.IN_FLIGHT.inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        metrics.IN_FLIGHT.dec()
    elapsed = time.perf_counter() - start
    # label by route template, not raw URL, to keep metric cardinality low
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    if path != "/metrics":
        metrics.HTTP_REQUESTS.labels(
            request.method, path, str(response.status_code)
        ).inc()
        metrics.HTTP_LATENCY.labels(request.method, path).observe(elapsed)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed * 1000:.2f}"
    return response


@app.exception_handler(APIError)
async def api_error(request: Request, exc: APIError):
    log.warning("%s %s: %s", request.url.path, exc.error, exc.detail)
    return error_json(request, exc.status, exc.error, exc.detail)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(request: Request, exc: ModelUnavailable):
    log.error("model unavailable: %s", exc)
    return error_json(
        request,
        503,
        "model_unavailable",
        "the model service is not available, try again shortly",
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
    return error_json(
        request, 422, "validation_error", f"missing or invalid fields: {fields}"
    )


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return error_json(request, 500, "internal_error", "unexpected server error")


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    metrics.MODEL_SERVER_UP.set(1 if model.ready() else 0)
    try:
        info = model.info()  # cached, so cheap on every scrape
        metrics.MODEL_VERSION.labels(alias=info["alias"]).set(float(info["version"]))
        metrics.MODEL_THRESHOLD.set(info["threshold"])
    except ModelUnavailable:
        pass  # model_server_up and the error counters show the outage
    try:
        metrics.FEEDBACK_PENDING.set(store.stats()["since_last_export"])
    except Exception:
        log.exception("could not read feedback stats for /metrics")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=Health)
def health():
    return {"status": "ok"}


@app.get("/ready", response_model=Ready, responses={503: {"model": Ready}})
def ready():
    if model.ready():
        return {"status": "ready", "model_server": "up"}
    return JSONResponse(
        status_code=503, content={"status": "not_ready", "model_server": "down"}
    )


@app.get("/model/info", response_model=ModelInfo, responses=ERROR_RESPONSES)
def model_info():
    info = model.info()
    metrics.MODEL_VERSION.labels(alias=info["alias"]).set(float(info["version"]))
    metrics.MODEL_THRESHOLD.set(info["threshold"])
    return info


# plain def, so fastapi runs it in a thread pool and the model call
# doesn't block the event loop
@app.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
def predict(request: Request, file: UploadFile = File(...)):
    start = time.perf_counter()
    img, size = read_image(file, config.MAX_UPLOAD_BYTES)
    info = model.info()
    x = to_model_input(img, info["image_size"])
    metrics.UPLOAD_BYTES.observe(size)
    metrics.INPUT_INTENSITY.observe(float(x.mean()))

    with metrics.INFERENCE_LATENCY.time():
        p = float(model.p_parasitized(x[None])[0])
    is_parasitized = p >= info["threshold"]
    label = CLASS_NAMES[0] if is_parasitized else CLASS_NAMES[1]
    confidence = p if is_parasitized else 1 - p
    metrics.PREDICTIONS.labels(label=label).inc()
    metrics.CONFIDENCE.labels(label=label).observe(confidence)
    log.info(
        "predict %s: %s p=%.4f v%s",
        request.state.request_id,
        label,
        p,
        info["version"],
    )
    return {
        "request_id": request.state.request_id,
        "label": label,
        "is_parasitized": is_parasitized,
        "p_parasitized": p,
        # with a threshold below 0.5 a "parasitized" call can have p < 0.5;
        # report it as is so the UI can flag it for a manual re-check
        "confidence": confidence,
        "threshold": info["threshold"],
        "model_version": info["version"],
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    responses=ERROR_RESPONSES,
)
def feedback(
    file: UploadFile = File(...),
    request_id: str = Form(...),
    predicted_label: str = Form(...),
    correct_label: str = Form(...),
    p_parasitized: float = Form(..., ge=0, le=1),
    model_version: str = Form(...),
):
    for name, value in (
        ("predicted_label", predicted_label),
        ("correct_label", correct_label),
    ):
        if value not in CLASS_NAMES:
            raise APIError(
                400, "invalid_label", f"{name} must be one of {list(CLASS_NAMES)}"
            )
    img, _ = read_image(file, config.MAX_UPLOAD_BYTES)
    try:
        fid, total = store.add(
            img,
            request_id,
            predicted_label,
            correct_label,
            p_parasitized,
            model_version,
        )
    except Exception as e:
        log.exception("could not store feedback")
        raise APIError(500, "storage_error", "could not store the feedback") from e
    agreed = predicted_label == correct_label
    metrics.FEEDBACK.labels(
        agreed=str(agreed).lower(), correct_label=correct_label
    ).inc()
    log.info(
        "feedback %d for %s: %s -> %s",
        fid,
        request_id,
        predicted_label,
        correct_label,
    )
    return {
        "stored": True,
        "feedback_id": fid,
        "agreed": agreed,
        "feedback_count": total,
    }


@app.get("/feedback/stats", response_model=FeedbackStats)
def feedback_stats():
    return store.stats()
