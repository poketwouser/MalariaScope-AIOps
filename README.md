# MalariaScope

Automated malaria parasite detection from thin blood smear cell images, with the ML pipeline, serving and monitoring around it.

- Problem and objectives: [PROJECT_STATEMENT.md](PROJECT_STATEMENT.md)
- Documents: [architecture](docs/architecture.pdf), [HLD](docs/HLD.pdf), [LLD](docs/LLD.pdf), [test plan](docs/test_plan.md), [test report](docs/test_report.md), [user manual](docs/user_manual.md)
- Evidence: [screenshots/](screenshots/), one folder per area
- [Instructions for TAs](Instructions_for_TAs.md): how to run and check each part
- [AI disclosure](AI_DISCLOSURE.md)

## Development environment

```bash
mamba env create -f environment.yml
mamba activate malariascope
```

## Run the stack (Docker Compose)

Needs Docker with Compose v2 and the Conda environment above (for `dvc`). The DVC remote is a local
folder; on another machine, point it at `~/tmp/dvc_storage`, which Airflow mounts:

```bash
mkdir -p ~/tmp/dvc_storage
dvc remote modify --local loc_remote url ~/tmp/dvc_storage
```

With a copy of the remote in that folder:

```bash
dvc pull                               # data, model and the MLflow store snapshot
python scripts/mlflow_store.py restore # registry with the malariascope@champion model
docker compose up -d --build           # first build takes about 35 minutes
docker compose ps                      # wait until every service shows (healthy)
```

Without it, rebuild everything from TFDS (about 9 minutes on a 16-core CPU, including the download), then start the stack:

```bash
docker compose up -d mlflow            # empty tracking server and registry
mkdir -p data/feedback                 # no feedback yet
dvc repro                              # ingest → … → train → evaluate → promote (registers v1 as @champion)
docker compose up -d --build
```

| Service | URL | What it is |
|---|---|---|
| frontend | http://localhost:8501 | Streamlit UI (Predict and Pipeline pages) |
| backend | http://localhost:8000/docs | FastAPI inference API (Swagger) |
| mlflow | http://localhost:5000 | MLflow tracking server and model registry |
| model-server | http://localhost:5001/ping | `mlflow models serve` for `malariascope@champion` |
| prometheus | http://localhost:9090/targets | metrics scraping and alert rules |
| pushgateway | http://localhost:9091 | metrics pushed by pipeline stages and DAG runs |
| alertmanager | http://localhost:9093 | alert routing; notifications go to `results/alerts.log` |
| grafana | http://localhost:3000 | Service and Pipeline dashboards (admin / admin, or anonymous view) |
| airflow-webserver | http://localhost:8080 | Airflow UI and REST API (admin / admin) |

Ports are bound to 127.0.0.1 only. The model server follows the `@champion` alias by itself,
so a promotion or a rollback takes effect within about 15 seconds, with no restart.

Airflow runs two DAGs: `training_pipeline` (the DVC stages; start it from the Airflow UI or the
Pipeline page) and `feedback_retrain` (every 30 minutes; retrains once 50 new feedback items have
arrived). Each task pushes its stage metrics to the Pushgateway for the Grafana Pipeline
dashboard. `scripts/simulate_traffic.py` sends train-split cells through the API (optionally
with feedback, or brightened to trigger the drift alert). The first build of the Airflow image downloads about 1.5 GB. Rebuild one service with
`docker compose up -d --build --no-deps <service>`. Stop everything with `docker compose down`.

## Tests

```bash
pytest --cov=src/malariascope --cov=backend/app   # unit, integration and e2e
pytest -m "not e2e"                               # without the stack running
```

The e2e tests in `tests/e2e/` run against the compose stack and skip themselves when it is down.
Test plan, test cases and acceptance criteria: `docs/test_plan.md`. Latest results:
`docs/test_report.md`.

## Repository layout

```
src/malariascope/   pipeline code (data, training, evaluation, promotion)
backend/            FastAPI inference service
frontend/           Streamlit UI
airflow/dags/       Airflow DAGs
monitoring/         Prometheus and Grafana configuration
mlflow-server/      MLflow tracking server image
mlflow-model-server/ MLflow model server image
tests/              unit, integration and end-to-end tests
docs/               architecture, HLD, LLD, test plan and report, user manual
scripts/            traffic simulator, serving smoke test, MLflow store snapshot
screenshots/        evidence, one folder per area
results/            pipeline metrics and plots tracked by DVC
```
