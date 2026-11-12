# Test Plan: MalariaScope

Results of the latest run are in [test_report.md](test_report.md).

## 1. Scope

**In scope**
- Data pipeline stages: image checks, the split (including feedback and the leakage guard), the resize shared by training and serving.
- Model contract and metrics: input/output shapes, threshold tuning, false-negative counting.
- Registry promotion gate.
- Backend REST API (LLD section 3): every endpoint, validation, error shapes, metrics.
- Streamlit UI: every path on the Predict page and the Pipeline page, including every error path.
- The running Docker Compose stack: health of all services, champion serving, Prometheus scraping, alert rules, Grafana dashboards, Airflow DAG loading, latency.
- Alert rules (promtool).

**Out of scope**
- Model quality beyond the acceptance criteria below (covered by the experiments in HLD section 5.4).
- Load and stress testing beyond the 50-request latency check.
- Browsers other than the one used for screenshots; Streamlit renders the same everywhere.
- Security testing (the stack binds only to 127.0.0.1).

## 2. Strategy

| Level | Tool | Where | What it isolates |
|---|---|---|---|
| Unit | pytest | `tests/unit/` | Pure functions of the pipeline, metrics, promotion gate and Pushgateway client. No services. |
| Integration | pytest + FastAPI `TestClient` | `tests/integration/test_backend_api.py` | The backend app with a fake model client, so each failure (model down, bad upload) can be forced. |
| Integration (UI) | pytest + Streamlit `AppTest` | `tests/integration/test_frontend.py` | Each UI page with the `api` module stubbed, so every error path can be forced without services. |
| End to end | pytest + requests | `tests/e2e/test_stack.py` | The real containers, images, env vars and network. Skipped automatically when the stack is down. |
| Rules | promtool | `monitoring/prometheus/rules_test.yml` | Alert rules fire (and stay silent) on synthetic series. |
| System (manual) | DVC, Airflow, scripts | `screenshots/` | Long-running flows: full pipeline, retrain loop, drift alert. Run once each with the log kept as evidence. |
| Static | black, flake8 | whole repo | Formatting and lint. |

The end-to-end level exists because the Day 7 bug (the backend image lacked pandas, so `/model/info` failed in the container) passed every unit and integration test.

End-to-end tests never store feedback, so running them doesn't change the retraining data.

## 3. Environment

| Item | Version |
|---|---|
| OS | Linux, 16 CPU cores, 14 GB RAM, no GPU |
| Python (tests) | 3.10.22, conda env `malariascope` (`environment.yml`) |
| pytest / pytest-cov | 9.1.1 / 7.1.0 |
| Streamlit (AppTest) | 1.65 |
| Docker / Compose | 29.1.3 / 2.40.3 |
| Stack under test | `docker compose up -d`: 12 services, all images pinned in `docker-compose.yml` |
| Model under test | registry `malariascope@champion` (version 1) |

## 4. How to run

```bash
mamba activate malariascope
docker compose up -d                       # needed only for tests/e2e
pytest --cov=src/malariascope --cov=backend/app
pytest -m "not e2e"                        # without the stack
docker compose exec -T -w /etc/prometheus prometheus promtool test rules rules_test.yml
black --check . && flake8
```

## 5. Acceptance criteria

These were fixed in `PLAN.md` before testing began. The software is accepted when all of them hold.

| ID | Criterion | Measured by |
|---|---|---|
| AC-1 | Test accuracy ≥ 95% on the fixed test split (2,756 images) | champion's registry tag `test_accuracy` (promote stage) |
| AC-2 | Parasitized recall ≥ 97% on the same split | champion's registry tag `test_recall_parasitized` |
| AC-3 | p95 `/predict` latency < 500 ms, end to end through the backend and model server | E2E-10 (50 calls); same limit as the `PredictLatencyHigh` alert |
| AC-4 | Every automated test passes | pytest, promtool |
| AC-5 | The UI handles every error case with a message and no crash | UI-02, UI-06, UI-08, UI-09, UT-18, E2E-05, ST-09 |
| AC-6 | Code is lint clean | black, flake8 |
| AC-7 | Training is reproducible: same commit and data give the same metrics | ST-01 |

## 6. Test cases

Steps for the automated cases are the test functions themselves; the expected result is what they assert.

### 6.1 Unit (`tests/unit/`)

| ID | Test | Expected result |
|---|---|---|
| UT-01 | `test_data_stages::test_check_image_accepts_normal_cell` | A normal RGB cell passes the quality checks |
| UT-02 | `test_check_image_rejects_blank` / `_tiny` / `_grayscale` / `_corrupt_file` | Each bad image is rejected with its reason |
| UT-03 | `test_split_sizes_and_no_overlap` | 80/10/10 sizes; no image in two splits |
| UT-04 | `test_split_is_stratified` / `_is_reproducible` / `_rejects_bad_fractions` | Class balance kept per split; same seed gives the same split; bad fractions raise |
| UT-05 | `test_to_model_input_shape_and_dtype` / `_converts_rgba_to_rgb` | Any upload becomes uint8 64×64×3 |
| UT-06 | `test_intensity_stats_match_backend_per_image_mean` | Drift baseline uses the same statistic the backend reports |
| UT-07 | `test_feedback::test_pixel_hash_ignores_file_encoding` | Same pixels give the same hash across PNG re-encoding |
| UT-08 | `test_export_copies_new_items_once` | Feedback is exported once and marked exported |
| UT-09 | `test_feedback_goes_to_train_only` / `test_no_feedback_leaves_split_unchanged` | Feedback enters train only; val/test never change |
| UT-10 | `test_feedback_matching_val_or_test_is_dropped` | Leakage guard drops feedback that matches val/test pixels |
| UT-11 | `test_duplicate_feedback_keeps_latest_label` | Latest correction wins |
| UT-12 | `test_model_and_metrics::test_perfect_predictions` / `test_missed_infection_counts_as_false_negative` | Metrics count FN/FP with parasitized as positive |
| UT-13 | `test_lower_threshold_flags_more_cells_as_parasitized` / `test_pick_threshold_reaches_target_recall` | Threshold tuning on val reaches the target recall |
| UT-14 | `test_model_input_output_contract` | Model takes uint8 (n,64,64,3), returns (n,2) probabilities |
| UT-15 | `test_promote::*` (4 tests) | First model promoted; low recall never promoted; promotion needs `min_gain`; worse challenger rejected |
| UT-16 | `test_pushmetrics::*` (5 tests) | Stage JSON becomes gauges; push is a no-op without a gateway and never fails a task |
| UT-17 | `test_tracking::*` (3 tests) | `git_dirty` is false for DVC's own outputs, true for changed source |
| UT-18 | `test_uploads::*` (5 tests) | Good image accepted; empty, fake and truncated files and oversize files get a message |

### 6.2 Backend integration (`tests/integration/test_backend_api.py`)

| ID | Test | Expected result |
|---|---|---|
| IT-01 | `test_health`, `test_ready_reflects_model_server` | `/health` 200; `/ready` 200 or 503 following the model server |
| IT-02 | `test_predict_parasitized` / `_uninfected` | Correct label, confidence, version in the response |
| IT-03 | `test_predict_uses_tuned_threshold_not_half` | Decision uses the champion's tuned threshold |
| IT-04 | `test_predict_accepts_jpeg` | JPEG accepted |
| IT-05 | `test_predict_rejects_non_image` / `_other_image_formats` / `_large_file` / `_missing_file` | 400 / 400 / 413 / 422 with the LLD error shape |
| IT-06 | `test_predict_model_down` | 503 `model_unavailable` |
| IT-07 | `test_feedback_is_stored_and_counted` / `test_feedback_rejects_unknown_label` | 201 and counted; unknown label 400 |
| IT-08 | `test_model_info` | Champion metadata returned |
| IT-09 | `test_metrics_count_predictions` / `test_metrics_report_serving_state_without_traffic` | Prometheus counters and gauges update |

### 6.3 UI (`tests/integration/test_frontend.py`)

| ID | Steps | Expected result |
|---|---|---|
| UI-01 | Open Predict page, backend up | Sidebar shows "Ready" and model version 1 |
| UI-02 | Open Predict page, backend down | Sidebar shows "Analysis service unavailable"; no crash |
| UI-03 | Open page; then pick a sample | Analyse is disabled until an image is chosen |
| UI-04 | Pick sample → Analyse | "Parasite detected" card; no low-confidence warning at 95% |
| UI-05 | Backend returns confidence 0.55 | Result shown with "Low confidence" warning |
| UI-06 | Backend times out on predict | Error message shown; no result or feedback buttons |
| UI-07 | Predict → "No, it is …" | Feedback sent once with the corrected label; thank-you shown; buttons gone |
| UI-08 | Feedback call fails | "Feedback could not be saved" shown; no crash |
| UI-09 | Open Pipeline page with Airflow, MLflow and backend all down | Warnings for each; no crash |

### 6.4 End to end (`tests/e2e/test_stack.py`, stack running)

| ID | Steps | Expected result |
|---|---|---|
| E2E-01 | Call each service's health endpoint (9 services) | All healthy |
| E2E-02 | `GET /model/info`; read `@champion` from the MLflow REST API | Same version, threshold and git commit |
| E2E-03 | POST a sample to the model server `/invocations` | Shape (1, 2), rows sum to 1 |
| E2E-04 | POST each of the 4 sample cells to `/predict` | Label matches the file name; decision matches threshold; request ID header matches body |
| E2E-05 | POST a text file, empty file, 5 MB + 1 byte, no file | 400, 400, 413, 422 with the right error codes |
| E2E-06 | POST feedback with label "sick" | 400 `invalid_label`; feedback total unchanged |
| E2E-07 | Predict once, poll Prometheus | Backend target up; `malariascope_predictions_total` rises within 20 s |
| E2E-08 | Query Prometheus rules and Grafana search | Alert rules loaded; both dashboards provisioned |
| E2E-09 | Query Airflow REST API | No import errors; both DAGs present |
| E2E-10 | 50 sequential `/predict` calls | p95 < 500 ms (AC-3) |

### 6.5 Alert rules (`monitoring/prometheus/rules_test.yml`)

| ID | Expected result |
|---|---|
| RT-01 … RT-06 | The 6 promtool cases pass (alerts fire on bad series, stay silent on good ones) |

### 6.6 System tests (manual, run once)

| ID | Steps | Expected result | Evidence |
|---|---|---|---|
| ST-01 | `dvc repro` on a clean commit in a fresh clone | Same metrics as the same data trained in Airflow; `git_dirty=False` | `04_mlflow/04_run_params_metrics_tags.png` |
| ST-02 | Re-run validate and preprocess with 32 Ray blocks | Byte-identical `.npz`; DVC skips train | `02_pipeline/06_airflow_preprocess_log_throughput.png` |
| ST-03 | Move the `@champion` alias | Model server restarts on the new version by itself | Registry page, `04_mlflow/07_registry_versions_aliases.png` |
| ST-04 | Send test-set images as feedback, run `feedback_retrain` | Leakage guard drops them | `05_monitoring/04_grafana_pipeline_dashboard.png` |
| ST-05 | Train-split feedback; `feedback_retrain` starts training | Gate rejects a worse model; v1 keeps serving | `08_feedback_loop/` |
| ST-06 | `simulate_traffic.py --brightness 1.3` | `InputDrift` fires and resolves | `05_monitoring/05_prometheus_alerts_firing.png`, `07_grafana_drift_panel.png` |
| ST-07 | Fault injection: make one metrics file unreadable, run `training_pipeline`, restore it, run again | Task fails, downstream tasks are `upstream_failed`, `PipelineRunFailed` fires, then resolves after the good run | `02_pipeline/04_airflow_grid_success_and_failed.png`, `05_airflow_failed_task_log.png` |
| ST-08 | Fresh clone, `dvc pull`, restore the MLflow store, `docker compose up -d --build`, `pytest` | 12 healthy services; all tests pass | `06_packaging/01_docker_compose_ps.png` |
| ST-09 | In the browser: upload a fake PNG, an oversize file; stop the backend and press Analyse | A plain message each time, never a traceback | `01_ui/06…08` |
| ST-10 | Fresh clone with an empty DVC remote: start `mlflow`, `dvc repro` from TFDS, `docker compose up -d`, `pytest` | Same v1 metrics as the original run; v1 registered as `@champion`; 12 healthy services; all tests pass | `docs/test_report.md` |
