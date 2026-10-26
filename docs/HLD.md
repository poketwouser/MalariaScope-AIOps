# High-Level Design: MalariaScope

> Outline (Day 1). Section structure follows the reference HLD, plus the "design choices and reasons" the rubric asks for. Fill each section in on the day you build that part. The *Your reasoning* prompts are for your own words: the viva tests whether you can defend them.

## 1. Purpose

One paragraph: what the system does and who uses it. Source: `PROJECT_STATEMENT.md` sections a to c.

## 2. Architecture overview

Embed `architecture.png` and give a short walkthrough of each block. Source: `docs/architecture.md`.

## 3. Key components and responsibilities

| Component | Technology | Responsibility |
|---|---|---|
| UI | Streamlit | |
| API | FastAPI | |
| Model serving | MLflow model server | |
| Model | TensorFlow Keras CNN | |
| Data engineering | Ray Data | |
| Pipeline / CI | DVC | |
| Orchestration | Airflow | |
| Tracking and registry | MLflow | |
| Monitoring | Prometheus, Grafana, node-exporter | |

## 4. Data flow

### 4.1 Inference
Numbered steps from upload to the rendered result.

### 4.2 Training pipeline
Ingest → validate → preprocess → train → evaluate → promote.

### 4.3 Feedback and retraining loop
Covers what triggers retraining.

## 5. Design choices and reasons

For each choice, write what you chose, the alternatives, and why.

1. **Loose coupling of UI and backend through configurable REST.** *Your reasoning:*
2. **Ray rather than Spark for preprocessing.** *Your reasoning (image files, per-file Python work, laptop CPU):*
3. **DVC stages run by Airflow, rather than logic duplicated in both.** *Your reasoning:*
4. **Keras CNN at 64×64.** *Your reasoning (CPU only, cell image size, accuracy target):*
5. **Backend calls the MLflow model server, or loads the model itself.** *Your reasoning:*
6. **Feedback storage format.** *Your reasoning:*
7. **Promotion rule and rollback.** *Your reasoning:*
8. **SQLite / file storage rather than a database server.** *Your reasoning:*

## 6. Non-functional requirements

Scalability, portability, observability, reproducibility (git commit plus MLflow run ID), security (encryption for stored feedback), explainability.

## 7. Assumptions and constraints

CPU only, no cloud (course rule), single machine, dataset size, single-hospital source data (bias).

## 8. Trade-offs and risks

Notes to revise from before the viva.
