"""Prometheus metrics for /metrics."""

from prometheus_client import Counter, Gauge, Histogram, Summary

PREFIX = "malariascope_"

HTTP_REQUESTS = Counter(
    PREFIX + "http_requests_total", "HTTP requests", ["method", "path", "status"]
)
HTTP_LATENCY = Histogram(
    PREFIX + "http_request_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
IN_FLIGHT = Gauge(PREFIX + "requests_in_flight", "Requests being handled right now")

PREDICTIONS = Counter(PREFIX + "predictions_total", "Predictions by label", ["label"])
CONFIDENCE = Histogram(
    PREFIX + "prediction_confidence",
    "Probability of the returned label",
    ["label"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
INFERENCE_LATENCY = Histogram(
    PREFIX + "inference_seconds",
    "Model server round trip",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
# drift signal: compare with the training distribution (docs/figures/intensity.png)
INPUT_INTENSITY = Histogram(
    PREFIX + "input_mean_intensity",
    "Mean pixel intensity (0-255) of uploaded images",
    buckets=(60, 80, 90, 100, 110, 115, 120, 125, 130, 140, 160, 200),
)
UPLOAD_BYTES = Summary(PREFIX + "upload_bytes", "Size of uploaded images")

FEEDBACK = Counter(
    PREFIX + "feedback_total", "Feedback received", ["agreed", "correct_label"]
)
ERRORS = Counter(PREFIX + "errors_total", "Handled errors by code", ["error"])
MODEL_VERSION = Gauge(
    PREFIX + "model_version", "Champion model version currently used", ["alias"]
)
MODEL_THRESHOLD = Gauge(PREFIX + "model_threshold", "Decision threshold in use")

# refreshed on every scrape, so they stay current without traffic
MODEL_SERVER_UP = Gauge(
    PREFIX + "model_server_up", "1 if the MLflow model server answers /ping"
)
FEEDBACK_PENDING = Gauge(
    PREFIX + "feedback_pending", "Feedback items not yet exported for retraining"
)
