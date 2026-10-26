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

## Repository layout

```
src/malariascope/   pipeline code (data, training, evaluation, promotion)
backend/            FastAPI inference service
frontend/           Streamlit UI
airflow/dags/       Airflow DAGs
monitoring/         Prometheus and Grafana configuration
tests/              unit, integration and end-to-end tests
docs/               architecture, HLD, LLD, test plan, user manual
screenshots/        evidence, one folder per rubric section
results/            raw logs behind every reported number
```
