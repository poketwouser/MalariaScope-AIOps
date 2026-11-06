# Test Report: MalariaScope

Test cases and acceptance criteria are in [test_plan.md](test_plan.md).

- **Run date:** 2026-11-07
- **Stack:** Docker Compose, 12 services, all healthy
- **Model under test:** `malariascope@champion`, version 1 (MLflow run `e7c15723…`)

## 1. Summary

| Level | Run | Passed | Failed | Skipped |
|---|---|---|---|---|
| Unit | 40 | 40 | 0 | 0 |
| Integration, backend | 16 | 16 | 0 | 0 |
| Integration, UI | 9 | 9 | 0 | 0 |
| End to end | 24 | 24 | 0 | 0 |
| **pytest total** | **89** | **89** | **0** | **0** |
| Alert rules (promtool) | 6 | 6 | 0 | 0 |
| System (manual) | 9 | 9 | 0 | 0 |

Static checks: `black --check .` clean, `flake8` no issues.

Evidence: `screenshots/07_testing/`.

## 2. Coverage

Line coverage of `src/malariascope` and `backend/app` from the pytest process: **51%** (1,072 statements, 522 missed).

| Area | Coverage | Why |
|---|---|---|
| Backend `main.py`, `schemas.py`, `config.py`, `metrics.py` | 95–100% | Covered by the integration tests |
| Backend `model_client.py` | 26% | Replaced by a fake in integration tests. The real one runs inside the backend container, where E2E-02, E2E-04 and E2E-07 exercise it, but in-process coverage can't see that |
| `image.py`, `model.py`, `labels.py`, `pushmetrics.py` | 95–100% | Unit tests |
| `train.py`, `ingest.py`, `eda.py`, `rollback.py` | 0% | Long-running stage scripts. They are run by `dvc repro` and the Airflow DAG (ST-01, ST-05), not by pytest |

## 3. Results per test case

All cases in test plan sections 6.1–6.5 passed. Notable measurements:

| Case | Result |
|---|---|
| E2E-02 | Backend serves version 1, threshold 0.2417, same git commit as the registry tag |
| E2E-04 | All 4 sample cells labelled correctly through the containers |
| E2E-05 | 400 / 400 / 413 / 422 with the LLD error codes |
| E2E-10 | 50 `/predict` calls on an idle machine: **p50 65.6 ms, p95 196.0 ms** (`07_testing/06_latency_p95.png`) |

System tests (section 6.6):

| Case | Result |
|---|---|
| ST-01 | Clean-commit training in a fresh clone and in Airflow gave identical test results to v2: accuracy 0.9560957910014514, 42 missed, 79 false alarms, `git_dirty=False` |
| ST-02 | Ray validate 15.4 s → 5.1 s, preprocess 14.1 s → 5.4 s, output byte-identical |
| ST-03 | Alias move noticed within 3 s; serving the new version about 5 s later |
| ST-04 | Feedback received 62, leakage guard dropped 2 |
| ST-05 | Retrain in Airflow. v2 and v3 accuracy 0.9561 vs champion 0.9626, so the gate rejected both and v1 kept serving |
| ST-06 | `InputDrift` fired during brightened traffic and resolved when it stopped |
| ST-07 | `ingest` failed with `PermissionError`, 8 tasks `upstream_failed`, `PipelineRunFailed` fired, and resolved after the next good run |
| ST-08 | Fresh clone: 12 healthy services and all tests passing. Images were built from Docker's layer cache, so this doesn't time a build from nothing |
| ST-09 | Fake PNG, oversize file and stopped backend each gave a plain message |

## 4. Defects found and fixed in the final run

The full run-through on 7 Nov found problems the automated tests had missed. Each is fixed.

| Defect | Effect | Fix |
|---|---|---|
| A fake or corrupt `.png` crashed the Predict page | Users saw a Python traceback. AC-5 had been reported as met on 5 Nov, which was wrong | Decode the file before previewing it; UT-18 added |
| `git_dirty` was always true under `dvc repro` | DVC deletes a stage's outputs first, so the flag said nothing. v1 and v2 still carry `git_dirty=True` from this bug | Ignore DVC outputs in the check; UT-17 added |
| `mlflow run . -e train` failed | The `MLproject` entry point couldn't run | Follow MLflow Exercise 2: set no experiment under `mlflow run` |
| A fresh clone had no model to serve | The MLflow store wasn't versioned, so the model server couldn't start | DVC snapshot of the store, `scripts/mlflow_store.py` |
| The scheduler could start before the backend | A catch-up `feedback_retrain` run failed at cold start and raised a false alert | `depends_on: backend: service_healthy` |
| Grafana "Stage last success" showed `NaN` | Text columns were formatted as dates | Apply the date format to the time column only |

## 5. Acceptance criteria

| ID | Criterion | Measured | Met? |
|---|---|---|---|
| AC-1 | Test accuracy ≥ 95% | 96.26% | ✅ Yes |
| AC-2 | Parasitized recall ≥ 97% | 96.73% (45 of 1,378 infected cells missed) | ❌ **No**, short by 0.27 points (4 cells) |
| AC-3 | p95 `/predict` < 500 ms | 196.0 ms | ✅ Yes |
| AC-4 | All automated tests pass | 89/89 pytest, 6/6 promtool | ✅ Yes |
| AC-5 | UI handles every error case | UI-02, UI-06, UI-08, UI-09, UT-18, E2E-05, ST-09 pass | ✅ Yes, after the fix in section 4 |
| AC-6 | Lint clean | black and flake8 clean | ✅ Yes |
| AC-7 | Reproducible training | ST-01 identical metrics | ✅ Yes |

**Verdict:** 6 of 7 criteria met. The software is **not yet fully accepted**: AC-2 fails.

About AC-2: the threshold is tuned on the validation split to reach 97% recall there (`evaluate.target_recall`). On the test split the same threshold gives 96.73%. Tuning the threshold on test would pass AC-2, but it would leak test data, so it isn't done. Variants with higher test recall (wider 97.1%, img96 97.1%, no-augment 97.2%) got it only with much lower thresholds and lower accuracy (HLD section 5.4).
