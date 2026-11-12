# Instructions for TAs: MalariaScope

Kiran Kumar P

How to check each part quickly. If you don't want to run anything, every check has screenshots in
[`screenshots/`](screenshots/) (one folder per area; numbers like "03" below are the file prefixes).
The design documents are in [`docs/`](docs/).

## 1. Running it

Needs Linux, Docker with Compose v2, about 15 GB of free disk, and Miniforge/Mamba (for `dvc` and the pipeline).
Everything runs on CPU. Run all commands from the repository root.

```bash
mamba env create -f environment.yml && mamba activate malariascope
```

The DVC remote in `.dvc/config` is a local folder on my laptop (`/home/kiran/tmp/dvc_storage`), so
it doesn't exist on your machine. Point DVC at `~/tmp/dvc_storage` instead; Airflow mounts that same path.
Then pick **A** if you have a copy of my DVC remote (about 1 GB), or **B** to rebuild everything from TFDS.

```bash
mkdir -p ~/tmp/dvc_storage
dvc remote modify --local loc_remote url ~/tmp/dvc_storage
```

**A. With a copy of the DVC remote (about 5 minutes plus the image build)**

```bash
cp -r /path/to/dvc_storage/. ~/tmp/dvc_storage/
dvc pull                                 # data, model and the MLflow store snapshot
python scripts/mlflow_store.py restore   # registry: 3 versions, malariascope@champion = v1
docker compose up -d --build
```

**B. From scratch, without the remote (about 9 minutes plus the image build)**

```bash
docker compose up -d mlflow              # tracking server and registry, empty
mkdir -p data/feedback                   # no feedback yet
dvc repro                                # downloads TFDS malaria, Ray stages, trains, registers v1 as @champion
docker compose up -d --build
```

The first `docker compose up --build` takes about 35 minutes (the Airflow image installs TensorFlow and Ray).
Wait until `docker compose ps` shows every service `(healthy)` (12 services, plus `airflow-init`, which exits).

| URL | What you should see |
|---|---|
| http://localhost:8501 | the app: Predict page and Pipeline page |
| http://localhost:8000/docs | FastAPI Swagger: `/predict`, `/feedback`, `/feedback/stats`, `/model/info`, `/health`, `/ready` |
| http://localhost:5000 | MLflow: experiment `malariascope`, model `malariascope` with `@champion` |
| http://localhost:5001/ping | MLflow model server for `malariascope@champion` |
| http://localhost:8080 | Airflow (admin / admin): `training_pipeline` and `feedback_retrain` |
| http://localhost:9090/targets | Prometheus: `backend`, `pushgateway`, `node`, `alertmanager`, `prometheus`, all UP |
| http://localhost:9093 | Alertmanager |
| http://localhost:3000 | Grafana, dashboards "MalariaScope - Service" and "MalariaScope - Pipeline" (no login) |

To stop: `docker compose down` (images and volumes are kept, so the next `up -d` takes about a minute).

**Quickest check (no stack needed):** `pytest -m "not e2e"` gives **65 passed** (40 unit, 25 integration).
With the stack up, `pytest` gives **89 passed** (adds 24 end-to-end tests against the containers).

## 2. Checklist

### Front-end UI/UX

Screenshots: [`screenshots/01_ui`](screenshots/01_ui). User manual for non-technical users: [`docs/user_manual.md`](docs/user_manual.md).

| Check | How | Expected |
|---|---|---|
| Main flow | Predict page, pick a sample cell (or upload one), **Analyse** | red "Parasite detected" or green "No parasite detected", confidence bar (03, 04) |
| Feedback | 👍 / 👎 under the result | "Thanks" message, sent once per prediction (05) |
| Not an image | upload a `.txt` renamed to `.png` | plain message, no traceback (06) |
| Too large | file over 5 MB | "too large" message before any upload (07) |
| Backend down | `docker compose stop backend`, reload | "Analysis service unavailable", Analyse disabled (08) |
| Low confidence | any result under 70% | yellow warning to get the slide checked (UI-04 in the test plan) |
| Narrow window | resize the browser to phone width | columns stack (09) |

### ML pipeline visualisation

Screenshots: [`screenshots/02_pipeline`](screenshots/02_pipeline).

| Check | Where | Expected |
|---|---|---|
| Separate pipeline screen | app → **Pipeline** page | champion model, Airflow runs per DAG with **Run now**, per-stage state and time, Ray throughput, model versions, recent MLflow runs, feedback counts, links to every tool UI (01, 11–14) |
| Management console | Airflow UI | DAG graph, grid with success and failed runs, task logs (02–05, 15–17) |
| Throughput | Pipeline page "Stages" table, or `dvc metrics show` | validate about 5,400 images/s, preprocess about 5,100 images/s with Ray (06, 12) |
| Errors and failures | Grafana "MalariaScope - Pipeline" | failed runs, run duration, stage last success ([05_monitoring/04](screenshots/05_monitoring)) |

### Design documents

| Document | File |
|---|---|
| Architecture diagram with each block explained | [`docs/architecture.pdf`](docs/architecture.pdf) |
| HLD with design choices, paradigm and trade-offs | [`docs/HLD.pdf`](docs/HLD.pdf) |
| LLD: every endpoint with request/response schemas, status codes, error format | [`docs/LLD.pdf`](docs/LLD.pdf) (matches Swagger at `/docs`, [06_packaging/03, 05](screenshots/06_packaging)) |

Loose coupling: the frontend reaches the backend only through `BACKEND_URL` ([`frontend/api.py`](frontend/api.py)),
and the backend reaches the model only through the MLflow model server's REST `/invocations`.
Paradigm: functional DVC stages for the pipeline, small classes for the service layer (`ModelClient`, `FeedbackStore`).

### Implementation and testing

Screenshots: [`screenshots/07_testing`](screenshots/07_testing).

| Check | How | Expected |
|---|---|---|
| Style | `black --check .` and `flake8` | clean (03, 04) |
| Logging and errors | `docker compose logs backend` | one line per request with request id and latency; one JSON error shape ([LLD](docs/LLD.pdf) §3) |
| Tests | `pytest` | 89 passed (01); coverage 51% (02, explained in the report) |
| Alert rules | `promtool test rules monitoring/prometheus/rules_test.yml` (inside the Prometheus container) | 6/6 (05) |
| Test plan, cases, acceptance criteria | [`docs/test_plan.md`](docs/test_plan.md) | AC-1…AC-7, cases UT/IT/UI/E2E/RT/ST |
| Test report | [`docs/test_report.md`](docs/test_report.md) | counts, results per case, 6 of 7 criteria met (see §4) |

### Data engineering

`src/malariascope/data/`: `ingest` (TFDS → PNG + manifest) → `validate` (Ray Data `map_batches`: unreadable,
non-RGB, size, blank, SHA-1 duplicates) → `split` (seeded, stratified 80/10/10) → `preprocess` (Ray Data resize to
64 px, `.npz`). Throughput and time are in `results/validate.json` and `results/preprocess.json`.
Ray dashboard during preprocessing: [02_pipeline/07, 08](screenshots/02_pipeline).

### Source control and CI

Screenshots: [`screenshots/03_dvc_git`](screenshots/03_dvc_git).

| Check | Command | Expected |
|---|---|---|
| DAG | `dvc dag` | 8 stages: ingest → validate → split → preprocess → train → evaluate → promote, plus eda (01) |
| Reproducible | `dvc repro` on a pulled clone | "Data and pipelines are up to date" (02) |
| Metrics and params | `dvc metrics show`, `dvc params diff`, `dvc exp show -A` | (03–06) |
| Data, feedback and models in DVC | `data/feedback.dvc`, `data/mlflow_store.dvc`, stage outputs in `dvc.lock` | (07, 09, 10) |

### Experiment tracking

Screenshots: [`screenshots/04_mlflow`](screenshots/04_mlflow). MLflow at http://localhost:5000.

| Check | Where | Expected |
|---|---|---|
| Params, metrics, artifacts | any run in experiment `malariascope` | per-epoch curves, confusion matrix, ROC, classification report, model (04–06, 08–10) |
| Beyond autolog | run tags and params | tags `git_commit`, `git_dirty`, `data_md5` (DVC hash), `dvc_exp`, python, keras, CPU; params `n_train` (22,046, or 22,106 with feedback), `n_val`; metrics `train_seconds`, `test_tuned_recall_parasitized`; the version's `test_recall_parasitized` tag |
| Experiments | runs table / compare | baseline, lr5e-4, wider, img96, no-augment (02, 03) |
| Registry with a promotion rule | Models → malariascope | v1 `@champion`; v2 and v3 rejected (accuracy 0.9561 < 0.9626 + 0.005), v3 `@challenger` (07, 11–13) |

### Prometheus and Grafana

Screenshots: [`screenshots/05_monitoring`](screenshots/05_monitoring).

| Check | Where | Expected |
|---|---|---|
| Custom metrics | http://localhost:8000/metrics | `malariascope_predictions_total{label}`, `malariascope_prediction_confidence`, `malariascope_inference_seconds`, `malariascope_input_mean_intensity` (drift), `malariascope_feedback_total{agreed}`, `malariascope_model_version` (02; full list in [LLD](docs/LLD.pdf) §7.1) |
| Components monitored | Prometheus targets | backend, model server (`malariascope_model_server_up`), host (node-exporter), pipeline tasks and DAG runs (Pushgateway), Alertmanager (01) |
| Dashboards | Grafana | Service (15 panels) and Pipeline (23 panels), provisioned as JSON in `monitoring/grafana/dashboards/` (03, 04, 07, 12–17) |
| Alerts | http://localhost:9090/alerts | 10 rules; alerts go Alertmanager → `alert-receiver` → `results/alerts.log` (05, 06, 08) |
| Drift demo | `python scripts/simulate_traffic.py -n 80 --brightness 1.3` | `InputDrift` fires after about 5 minutes, resolves when traffic stops (05, 07) |

### Software packaging

Screenshots: [`screenshots/06_packaging`](screenshots/06_packaging).

| Check | How | Expected |
|---|---|---|
| MLflow for APIfication | `python scripts/smoke_test_serving.py` | 4/4 sample cells right via `/invocations` (02) |
| MLproject | `export MLFLOW_TRACKING_URI=http://localhost:5000`, then `mlflow run . -e train --experiment-name malariascope --env-manager local` | one entry point per DVC stage, plus `main` and `rollback` (04) |
| FastAPI | http://localhost:8000/docs | (03, 05) |
| Docker | `docker compose ps` | frontend and backend as separate services, pinned images, healthchecks, `depends_on: service_healthy` (01, 06) |

### Feedback loop and retraining

Screenshots: [`screenshots/08_feedback_loop`](screenshots/08_feedback_loop).

1. `python scripts/simulate_traffic.py -n 60 --feedback` sends 60 train-split cells with their true label.
2. In Airflow, trigger `feedback_retrain` with conf `{"min_feedback": 50}` (or wait for the 30-minute schedule).
3. It exports the feedback into `data/feedback` (DVC), then starts `training_pipeline`.
4. The split stage adds feedback to **train only** (val/test stay fixed) and drops anything matching a val/test image.
5. `promote` registers a new version and decides; `results/promote.json` and the version's tags give the reason.
6. If it is promoted, the model server follows the `@champion` alias within about 15 s, with no restart.
   Roll back with `python -m malariascope.rollback` (run with `PYTHONPATH=src`).

## 3. What the numbers should look like

Training is seeded, so `dvc repro` on another machine should give the same split and very close metrics.

| | Value |
|---|---|
| Data | 27,558 images, 13,779 per class; split 22,046 / 2,756 / 2,756 |
| v1 test accuracy at threshold 0.5 | 0.9648 (69 infections missed, 28 false alarms) |
| v1 at the val-tuned threshold 0.2417 | accuracy 0.9626, Parasitized recall 0.9673 (45 missed), specificity 0.958, ROC-AUC 0.993 |
| Training time | about 7 minutes on a 16-core CPU (11 epochs, early stopping) |
| `/predict` latency | p50 about 65 ms, p95 about 200 ms on an idle laptop |

## 4. Known limitations

- **AC-2 is not met:** Parasitized recall is 96.73%, my own target was 97% (4 cells short). The project statement
  asks for ≥95% accuracy and high recall, which is met. Tuning the threshold on the test split would pass it but
  leak test data, so I didn't ([`docs/test_report.md`](docs/test_report.md) §5).
- Stored feedback images aren't encrypted, and Grafana and Airflow use local default passwords. Ports are bound
  to 127.0.0.1 and containers run as non-root.
- No Grad-CAM; the UI explains a result through the confidence score and a low-confidence warning.
- `models/model.keras` and `results/evaluate.json` in the repo are from the latest pipeline run (the rejected v3).
  What is served is decided by the registry alias (`@champion` = v1), not by the latest file.
- v1 and v2 carry `git_dirty=True` from a bug fixed on 7 Nov (DVC deletes stage outputs before running); v3 shows
  `git_dirty=False` and matches v2 exactly.
