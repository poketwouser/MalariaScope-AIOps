# MalariaScope Architecture

![MalariaScope architecture](architecture.png)

Solid arrows are the main data path; dotted arrows are read-only lookups and pipeline controls from the UI. Blocks talk only over REST or files they own, using addresses set in the environment.

## Blocks

| Block | Responsibility | Course source |
|---|---|---|
| Streamlit frontend | Upload, show prediction and confidence, collect feedback, Pipeline page | Reference repo `frontend_streamlit/` |
| FastAPI backend | Validates input, resizes, calls the model server, applies the tuned threshold, stores feedback, exposes `/metrics` | Lecture 10 REST API serving; reference `backend/` |
| MLflow model server | Serves the version behind `@champion` over REST and switches by itself when the alias moves | Lecture 10 MLflow serving; reference `mlflow-model-server/` |
| Feedback store | Keeps user corrections; versioned with DVC for retraining | Reference `backend/main.py` (SQLite); Lecture 2 DVC |
| Airflow | Schedules and runs the pipeline and the feedback retraining | Reference `airflow/dags/` and `Dockerfile.airflow` (init + webserver + scheduler, SequentialExecutor) |
| DVC pipeline | Reproducible stages with params and outputs hashed in `dvc.lock` | Lecture 11 DVC for CI (`dvc stage add`, `params.yaml`, `dvc exp`); Lecture 2 (local remote) |
| Ray Data (validate and preprocess stages) | Parallel image checks and resizing, with throughput logged | Lecture 8 Ray; Assignment 3 (`map_batches`) |
| MLflow tracking and registry | Runs, metrics, artifacts, model versions and the `@champion`, `@challenger` and `@previous_champion` aliases | Lecture 3 (tracking server with SQLite), Lecture 10 (registry and aliases) |
| Prometheus, node-exporter, Pushgateway | Collect app, host and pipeline metrics | Lecture 9 (`Monitoring/prom_client.py`, `sample_prometheus.yml`); reference `monitoring/`. Pushgateway is not in the labs: Lecture 9 only has the "pull vs push" slide |
| Alertmanager + alert receiver | Routes alerts from the Prometheus rules to a local log | Lecture 9 (`sample_rules.yml`, `sample_alertmanager.yml`); Assignment 4 (`alert_receiver.py`, `rules.yml`) |
| Grafana | Near-real-time dashboards (Service, Pipeline) | Lecture 9; Assignment 4 |

## Where the code patterns come from

- Conda/Mamba `environment.yml` with Python 3.10: DA3408 Mod1 lab.
- Multi-stage `python:3.10` → `python:3.10-slim` Dockerfiles: Lecture 4 and the Lecture 5 Docker exercise. Copying a venv between stages, the non-root user, healthchecks and `depends_on: service_healthy` follow my Assignment 2.
- MLflow tracking server with a SQLite backend and proxied artifacts: Lecture 3. The registry needs a database-backed store: Lecture 10.
- Per-epoch params and metrics logged by hand rather than only autolog: MLflow Exercise 1. Model logged with `infer_signature` and an input example: MLflow Exercise 3.
- `mlflow run . -e train --experiment-name …` without `set_experiment` inside the run: MLflow Exercise 2. `MLproject` entry points: Lecture 3.
- Registering a version, `@champion` and `@challenger` aliases, and a gate that promotes only on a minimum accuracy gain: Lecture 10 MLflow serving exercise and MLflow Exercise 4. Rollback is moving the alias back.
- `mlflow models serve` resolves an alias only at start-up: Lecture 10. `mlflow-model-server/serve.py` restarts it when the alias moves; stopping the whole process group (SIGTERM, then SIGKILL) is the same as `m4util.stop` in the Lecture 10 notebooks.
- Plain `def` endpoints so FastAPI runs them in a thread pool, and the `X-Process-Time-ms` header from a timing middleware: Lecture 10 REST API serving exercise.
- `prometheus_client` counters, gauges, histograms and a summary: Lecture 9 (`Monitoring/prom_client.py`). Recording rules, the alert receiver and node-exporter with `hwmon` and `powersupplyclass` turned off: Assignment 4.
- Run tags for the git commit and the DVC data hash: the development guideline's "reproducible via a specific Git commit hash and a corresponding MLflow run ID".
- EDA figures with colour-blind-safe colours, labelled axes and a one-line message under each title: the plotting rules in `Mod 6_Assignments.pdf` (Assignment 4).
- `tensorflow-cpu` in the model server and Airflow images: reference repo (no GPU).
