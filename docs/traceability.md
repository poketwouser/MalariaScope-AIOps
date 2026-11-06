# Traceability: where each choice comes from

Every tool and pattern in MalariaScope is listed here with its source. Sources were checked by searching the text of the DA3408 lectures, notebooks and handouts, the reference repo (`reference_ALL`, the ALL classifier by another team) and Assignments 1–5.

## 1. Pipeline, versioning and tracking

| Choice | DA3408 | Reference repo | My assignments |
|---|---|---|---|
| Conda/Mamba `environment.yml` | Mod1 lab | Different (`conda.yaml`) | A1 |
| `dvc init`, `dvc remote add`, `dvc add`, `dvc push`, `dvc pull` | Lecture 2 | Not used | A1 |
| `dvc stage add`, `dvc repro`, `dvc dag`, `params.yaml`, `dvc exp`, `dvc metrics` | Lecture 11 | Not used | |
| MLflow tracking server with SQLite | Lecture 3, MLflow Exercise 1 | Same idea | A1, A5 |
| `MLproject`, `mlflow run --experiment-name --env-manager local` | Lecture 3, MLflow Exercise 2 | `MLproject` | A1 |
| Registry with `@champion` and `@challenger`, gate with a minimum gain | Lecture 10 MLflow Serving | Registers without a gate | A5 |
| `mlflow models serve` | Lecture 3, Lecture 10 | `mlflow-model-server/` | A5 |
| Ray Data | Lecture 8 (Ray AIR notebook) | Not used | A3 |
| `map_batches` batch functions | | Not used | A3 |

## 2. Serving and packaging

| Choice | DA3408 | Reference repo | My assignments |
|---|---|---|---|
| FastAPI with Pydantic schemas | Lecture 10 REST API | FastAPI, plain dicts | A2, A5 |
| FastAPI `TestClient`, pytest | Testing required by the Development Guidelines | No tests | A5 |
| Streamlit front end | Project statement | `frontend_streamlit/` | A4 |
| Multi-stage `python:3.10-slim` images, `.dockerignore` | Lecture 4, Lecture 5 exercise | Single stage | A2 |
| `healthcheck`, `depends_on: service_healthy` | | Not used | A2 |
| Service split: MLflow, model server, backend, frontend | | Same split | |
| Airflow init + webserver + scheduler, `SequentialExecutor` | Named in the AIOps course outline only | Same layout | |
| SQLite feedback store | | `backend/feedback.py` | |

## 3. Monitoring

| Choice | DA3408 | Reference repo | My assignments |
|---|---|---|---|
| `prometheus_client` custom metrics | Lecture 9, `Monitoring/prom_client.py` | Default instrumentator only | A4, A5 |
| Prometheus config, alert rules, Alertmanager | Lecture 9, `Monitoring/sample_*.yml` | Prometheus only | A4 |
| Recording rules, alert receiver | | | A4 |
| node-exporter | Lecture 9 | Yes | A4 |
| Grafana provisioning | Lecture 9 | Datasource only | A4 |

## 4. Not in the course material, the reference repo or the assignments

| Item | What it is used for | Why it is here |
|---|---|---|
| **Pushgateway** | Part of the running system: pipeline metrics | Lecture 9 covers pull vs push but doesn't name it. Batch tasks end before Prometheus can scrape them |
| flake8 | Dev tool: lint | The rubric asks for a standard coding style. Approved on 27 Oct |
| promtool | Dev tool: tests the alert rules | Ships with Prometheus |
| Keras `EarlyStopping` | Training callback | Standard Keras |
| Mermaid | Docs only: diagram source | Renders in GitHub; not part of the system |

Playwright was used only to capture screenshots, as in A5. It isn't a project dependency.

## 5. Deliberate differences from the reference repo

| Reference repo | MalariaScope | Reason |
|---|---|---|
| No DVC | DVC for data, models and CI | Rubric requires it |
| Hard-coded backend URL | `BACKEND_URL` from the environment | Rubric requires loose coupling |
| Registers every model | Gate on the same test split | Project statement: promote only if better |
| Backend loads the model | Backend calls MLflow serving over REST | Rubric: MLflow for APIfication |
| Feedback threshold 100 | `retrain.min_feedback: 50` | Configurable in `params.yaml` |
| `:latest` images, no healthchecks | Pinned images, healthchecks | Reproducible start-up |
| Python 3.9 | Python 3.10 | Mod1 and Mod3 |
| No logging, no tests | Both | Rubric requires them |

Lecture 10's aliases are used instead of the `Production` stage from MLflow Exercises 4 and 5, because Lecture 10 is the newer pattern in the course.

## 6. Document structure

| Document | Follows |
|---|---|
| `HLD.md` | Reference HLD sections, plus design choices and trade-offs (rubric) |
| `LLD.md` | Reference LLD sections: source tree, API spec, SQLite schema, DAGs, metrics, Docker images, environment variables |
| `user_manual.md` | Reference manual sections: audience, requirements, quick start, interpreting the result, feedback, common issues |
| `test_plan.md`, `test_report.md` | Rubric testing items |
| `AI_DISCLOSURE.md` | Code of conduct section 4, same layout as A4 and A5 |
