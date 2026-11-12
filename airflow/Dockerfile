# tasks run `dvc repro <stage>` with the pipeline venv below, not airflow's own python
FROM apache/airflow:2.10.5-python3.10

USER root
# git: tracking.py tags every MLflow run with the commit hash
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN python -m venv /opt/pipeline && \
    /opt/pipeline/bin/pip install --no-cache-dir -r /tmp/requirements-pipeline.txt

USER airflow
ENV PIPELINE_BIN=/opt/pipeline/bin
