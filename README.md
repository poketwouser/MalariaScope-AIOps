# MalariaScope

Automated malaria parasite detection from thin blood smear cell images, built as an AIOps (DA3408) course project.

- Problem and objectives: [PROJECT_STATEMENT.md](PROJECT_STATEMENT.md)
- Architecture (draft): [docs/architecture.md](docs/architecture.md)
- High-level design (in progress): [docs/HLD.md](docs/HLD.md)

> Work in progress. Setup and run instructions will be completed as the components are built.

## Development environment

```bash
mamba env create -f environment.yml
mamba activate malariascope
```

## Run the stack (Docker Compose)

Needs Docker with Compose v2, the data and model pulled with `dvc pull`, and a registered
`malariascope@champion` model in `mlflow.db` (created by `dvc repro`).

```bash
docker compose up -d --build     # first build takes about 9 minutes
docker compose ps                # wait until every service shows (healthy)
```

| Service | URL | What it is |
|---|---|---|
| frontend | http://localhost:8501 | Streamlit UI (Predict and Pipeline pages) |
| backend | http://localhost:8000/docs | FastAPI inference API (Swagger) |
| mlflow | http://localhost:5000 | MLflow tracking server and model registry |
| model-server | http://localhost:5001/ping | `mlflow models serve` for `malariascope@champion` |
| prometheus | http://localhost:9090/targets | metrics scraping |
| grafana | http://localhost:3000 | dashboards (admin / admin, or anonymous view) |
| airflow-webserver | http://localhost:8080 | Airflow UI and REST API (admin / admin) |

Ports are bound to 127.0.0.1 only. The model server follows the `@champion` alias by itself,
so a promotion or a rollback takes effect within about 15 seconds, with no restart.

Airflow runs two DAGs: `training_pipeline` (the DVC stages; start it from the Airflow UI or the
Pipeline page) and `feedback_retrain` (every 30 minutes; retrains once 50 new feedback items have
arrived). The first build of the Airflow image downloads about 1.5 GB. Rebuild one service with
`docker compose up -d --build --no-deps <service>`. Stop everything with `docker compose down`.

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
docs/               architecture, HLD, LLD, test plan, user manual
screenshots/        evidence, one folder per rubric section
results/            raw logs behind every reported number
```
