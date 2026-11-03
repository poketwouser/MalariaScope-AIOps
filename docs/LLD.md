# Low-Level Design: MalariaScope

> Draft (written ahead of the Day 5 backend build). The backend must match this spec exactly. If the code needs to differ, change this document first.

## 1. Source tree

```
src/malariascope/
  common.py          params, logging, Timer, write_json
  image.py           to_model_input(): resize shared by pipeline and backend
  model.py           build_model(), CLASS_NAMES
  tracking.py        MLflow URI, run tags (git commit, DVC data hash)
  data/ingest.py     TFDS -> PNG + manifest
  data/validate.py   Ray Data image checks + dedup
  data/split.py      stratified seeded split
  data/preprocess.py Ray Data resize -> .npz
  eda.py             dataset figures
  train.py           CNN training, MLflow logging
  evaluate.py        test metrics, val-tuned threshold
  promote.py         registry: challenger vs champion gate
  rollback.py        champion <- previous_champion
backend/             FastAPI service (section 3)
frontend/            Streamlit UI (section 5)
airflow/dags/        training and feedback-retrain DAGs (section 6)
monitoring/          Prometheus and Grafana configuration (section 7)
```

## 2. Pipeline stages (dvc.yaml)

| Stage | Command (`PYTHONPATH=src python -m ...`) | Inputs | Outputs |
|---|---|---|---|
| ingest | `malariascope.data.ingest` | TFDS `malaria`, `params.data.dataset` | `data/raw/`, `results/ingest.json` |
| validate | `malariascope.data.validate` | `data/raw`, `params.validate` | `data/validated/`, `results/validate.json` |
| split | `malariascope.data.split` | `data/validated`, `params.data.seed/split` | `data/split/`, `results/split.json` |
| preprocess | `malariascope.data.preprocess` | `data/raw`, `data/split`, `params.preprocess` | `data/processed/{train,val,test}.npz`, `results/preprocess.json` |
| eda | `malariascope.eda` | `data/raw`, `data/validated` | `docs/figures/*.png` |
| train | `malariascope.train` | `data/processed`, `params.train` | `models/model.keras`, `results/train.json`, MLflow run |
| evaluate | `malariascope.evaluate` | model, `data/processed`, `params.evaluate` | `results/evaluate.json`, `results/plots/*.png` |
| promote | `malariascope.promote` | `results/evaluate.json`, `params.promote` | registry version + aliases, `results/promote.json` |

## 3. Backend REST API (FastAPI, port 8000)

Run: `PYTHONPATH=src:backend uvicorn app.main:create_app --factory --port 8000`. Every response carries `X-Request-ID` and `X-Process-Time-ms` headers.

All JSON responses use UTF-8. Every error response has the same shape:

```json
{"error": "<machine_code>", "detail": "<human readable message>", "request_id": "<uuid4>"}
```

### 3.1 `POST /predict`

Classify one cell image.

| | |
|---|---|
| Request | `multipart/form-data`, field `file`: PNG or JPEG, ≤ 5 MB |
| 200 | `PredictResponse` |
| 400 `invalid_image` | file isn't a decodable PNG/JPEG |
| 413 `file_too_large` | file > 5 MB |
| 422 `validation_error` | `file` field missing |
| 503 `model_unavailable` | MLflow model server unreachable or erroring |

```json
PredictResponse {
  "request_id": "string (uuid4)",
  "label": "parasitized | uninfected",
  "is_parasitized": "bool",
  "p_parasitized": "float 0-1",
  "confidence": "float 0-1, probability of the returned label (can be < 0.5 when the tuned threshold is below 0.5; the UI flags these for a manual re-check)",
  "threshold": "float, decision threshold of the serving model version",
  "model_version": "string",
  "latency_ms": "float"
}
```

Steps:
1. Validate the content type and size.
2. Decode with PIL and run `to_model_input(img, image_size)`.
3. POST `{"instances": [[...uint8 64x64x3...]]}` to `MODEL_SERVER_URL/invocations`.
4. Apply the threshold: the label is `parasitized` if `p_parasitized >= threshold`.
5. Update the Prometheus metrics (section 7).

### 3.2 `POST /feedback`

Store a user's correction or confirmation for retraining.

| | |
|---|---|
| Request | `multipart/form-data`: `file` (the same image), `request_id` (str), `predicted_label`, `correct_label` (both `parasitized`/`uninfected`), `p_parasitized` (float), `model_version` (str) |
| 201 | `{"stored": true, "feedback_id": int, "agreed": bool, "feedback_count": int}` |
| 400 `invalid_image` / `invalid_label` | |
| 413 `file_too_large` | |
| 500 `storage_error` | writing the image or the DB row failed |

### 3.3 `GET /feedback/stats`

Returns 200 `{"total": int, "agreed": int, "disagreed": int, "since_last_export": int}`. Airflow uses this to decide on retraining.

### 3.4 `GET /model/info`

Returns 200 with the following body, or 503 `model_unavailable`:

```json
{"name": "malariascope", "alias": "champion", "version": "string",
 "threshold": float, "image_size": int, "test_accuracy": float, "test_recall_parasitized": float,
 "git_commit": "string", "data_md5": "string", "class_names": ["parasitized", "uninfected"]}
```

Read from the MLflow registry (the tags on the version behind `@champion`; `image_size` comes from the model signature) and cached for `MODEL_INFO_TTL` seconds. If the registry is down, the last known info is served.

### 3.5 `GET /health` and `GET /ready`

- `/health`: always 200 `{"status": "ok"}` while the process is up (liveness).
- `/ready`: 200 `{"status": "ready", "model_server": "up"}`, or 503 `{"status": "not_ready", "model_server": "down"}` (readiness, used by the Docker healthcheck and the UI banner).

### 3.6 `GET /metrics`

Prometheus text exposition format (section 7).

### 3.7 Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_SERVER_URL` | `http://model-server:5001` | MLflow model server |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | registry lookup for `/model/info` |
| `FEEDBACK_DIR` | `/app/feedback` | feedback images and DB |
| `MAX_UPLOAD_MB` | `5` | upload limit |
| `MODEL_INFO_TTL` | `60` | seconds to cache registry info |
| `LOG_LEVEL` | `INFO` | logging level |

## 4. Feedback store

```
$FEEDBACK_DIR/
  feedback.db                         SQLite metadata
  images/<correct_label>/<id>.png     image files (easy to `dvc add` and to diff)
```

```sql
CREATE TABLE feedback (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  ts              REAL    NOT NULL,      -- unix time
  request_id      TEXT    NOT NULL,
  predicted_label TEXT    NOT NULL,
  correct_label   TEXT    NOT NULL,
  p_parasitized   REAL    NOT NULL,
  model_version   TEXT    NOT NULL,
  image_path      TEXT    NOT NULL,      -- relative to FEEDBACK_DIR
  exported        INTEGER NOT NULL DEFAULT 0   -- 1 once copied into a training snapshot
);
```

## 5. Frontend (Streamlit, port 8501)

- Reads `BACKEND_URL` from the environment. It has no other link to the backend (loose coupling).
- **Predict page:** upload → preview → Predict → label, confidence bar and threshold → feedback buttons ("Correct" / "Wrong, it is …").
- **Pipeline page:** latest Airflow DAG runs, MLflow champion version and metrics, feedback stats, and links to the Airflow, MLflow, Grafana and Prometheus UIs.
- Error states: backend unreachable, model not ready, invalid file, file too large.

## 6. Airflow DAGs

Airflow 2.10.5 (`apache/airflow:2.10.5-python3.10`, as in the reference repo) runs as three
compose services: `airflow-init` (DB migration and admin user), `airflow-webserver` (UI and
REST API on :8080) and `airflow-scheduler`. The executor is `SequentialExecutor` on SQLite.
Tasks run inside the scheduler container on the repo bind-mounted at `/opt/project`. They use
a separate venv (`/opt/pipeline`, from `airflow/requirements-pipeline.txt`), so the pipeline's
TensorFlow, Ray, DVC and MLflow can't clash with Airflow's own pinned packages.

### 6.1 `training_pipeline` (manual, REST API, or started by `feedback_retrain`)

```
ingest → validate → split → preprocess → train → evaluate → promote → dvc_push
              └──→ eda ─────────────────────────────────────────────────┘
```

- **Tasks:** each task is `dvc repro <stage>` and then prints the stage's metrics JSON. The
  printed line becomes the task's XCom, which the Pipeline page reads (for example Ray
  throughput).
- **Skipping:** DVC skips stages whose inputs are unchanged, so a run with no new data takes
  under a minute.
- **One run at a time:** `max_active_runs=1` keeps DVC runs on the shared repo from
  overlapping. `retries=0` keeps failures visible.
- **Promotion:** `promote` only moves the `@champion` alias. The model server watches the
  alias (`mlflow-model-server/serve.py`, polling every 15 s) and restarts itself on the new
  version, so Airflow needs no access to other containers. Rollback works the same way.

### 6.2 `feedback_retrain` (every 30 minutes)

```
enough_feedback → export_feedback → start_training (triggers training_pipeline)
```

- **`enough_feedback`** (ShortCircuitOperator): `GET {BACKEND_URL}/feedback/stats`. It
  continues only if `since_last_export ≥ retrain.min_feedback` (params.yaml, 50). A trigger
  config `{"min_feedback": N}` overrides the threshold for one run. Otherwise the downstream
  tasks show as skipped.
- **`export_feedback`**: `python -m malariascope.data.feedback`, then `dvc add data/feedback`.
  - It copies unexported items into `data/feedback/images/<label>/fb_<id>.png` and appends them
    to the cumulative `data/feedback/manifest.csv` (with the pixel SHA-1).
  - It sets `exported=1` in `feedback.db`, and only after the snapshot is written.
- **`start_training`**: TriggerDagRunOperator. It doesn't wait for the training run, because
  under the SequentialExecutor a waiting task would hold the only slot.

### 6.3 How feedback enters training (split stage)

- **Train only:** feedback rows join the **train** split only (`source=feedback`). Val and test
  never change, so challenger and champion are compared on the same images.
- **Leakage guard:** feedback whose decoded pixels match a val or test image is dropped. The
  UI's sample images come from the test set.
- **Duplicates:** repeated feedback on the same cell keeps the latest label.
- **Order:** feedback rows are appended after the raw rows. With an empty snapshot,
  `train.npz` is byte-identical to before (verified on Day 8), and training is skipped.

## 7. Prometheus metrics (backend)

To be detailed on Day 9:
- `malariascope_predictions_total{label}`
- `malariascope_prediction_confidence` (histogram)
- `malariascope_inference_latency_seconds` (histogram)
- `malariascope_feedback_total{agreed}`
- `malariascope_input_mean_intensity` (histogram, drift signal)
- `malariascope_model_info{version}`
- `malariascope_errors_total{error}`
