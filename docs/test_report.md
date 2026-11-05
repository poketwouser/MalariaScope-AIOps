# Test Report: MalariaScope

Test cases and acceptance criteria are in [test_plan.md](test_plan.md).

- **Run date:** 2026-11-05
- **Stack:** Docker Compose, 12 services, all healthy
- **Model under test:** `malariascope@champion`, version 1 (MLflow run `e7c15723…`)

## 1. Summary

| Level | Run | Passed | Failed | Skipped |
|---|---|---|---|---|
| Unit | 32 | 32 | 0 | 0 |
| Integration, backend | 16 | 16 | 0 | 0 |
| Integration, UI | 9 | 9 | 0 | 0 |
| End to end | 24 | 24 | 0 | 0 |
| **pytest total** | **81** | **81** | **0** | **0** |
| Alert rules (promtool) | 6 | 6 | 0 | 0 |
| System (manual) | 6 | 6 | 0 | 0 |

Static checks: `black --check .` clean (44 files), `flake8` no issues.

Logs: `results/logs/day10_pytest.log`, `day10_e2e_latency.log`, `day10_promtool.log`, `day10_lint.log`.

## 2. Coverage

Line coverage of `src/malariascope` and `backend/app` from the pytest process: **51%** (1,068 statements, 523 missed).

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
| E2E-10 | 50 `/predict` calls: **p50 59.2 ms, p95 247.0 ms** |

System tests (section 6.6), from their logs:

| Case | Result |
|---|---|
| ST-01 | Clean-commit `dvc repro` gave accuracy 0.9626 and recall 0.9673, identical to the baseline experiment |
| ST-02 | Ray validate 15.4 s → 5.1 s, preprocess 14.1 s → 5.4 s, output byte-identical |
| ST-03 | Alias move noticed within 3 s; serving the new version about 5 s later |
| ST-04 | `received 2, leaked_dropped 2` |
| ST-05 | Retrain in Airflow (259 s). v2 accuracy 0.9561 vs v1 0.9626, so the gate rejected it and v1 kept serving |
| ST-06 | `InputDrift` fired at 09:48 and resolved at 10:02 |

## 4. Acceptance criteria

| ID | Criterion | Measured | Met? |
|---|---|---|---|
| AC-1 | Test accuracy ≥ 95% | 96.26% | ✅ Yes |
| AC-2 | Parasitized recall ≥ 97% | 96.73% (45 of 1,378 infected cells missed) | ❌ **No**, short by 0.27 points (4 cells) |
| AC-3 | p95 `/predict` < 500 ms | 247.0 ms | ✅ Yes |
| AC-4 | All automated tests pass | 81/81 pytest, 6/6 promtool | ✅ Yes |
| AC-5 | UI handles every error case | UI-02, UI-06, UI-08, UI-09, E2E-05 pass | ✅ Yes |
| AC-6 | Lint clean | black and flake8 clean | ✅ Yes |
| AC-7 | Reproducible training | ST-01 identical metrics | ✅ Yes |

**Verdict:** 6 of 7 criteria met. The software is **not yet fully accepted**: AC-2 fails.

About AC-2: the threshold is tuned on the validation split to reach 97% recall there (`evaluate.target_recall`). On the test split the same threshold gives 96.73%. Tuning the threshold on test would pass AC-2, but it would leak test data, so it isn't done. Variants with higher test recall (wider 97.1%, img96 97.1%, no-augment 97.2%) got it only with much lower thresholds and lower accuracy (`results/logs/day3_experiments.log`).
