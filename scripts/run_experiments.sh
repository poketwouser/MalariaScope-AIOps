#!/usr/bin/env bash
# Run the model-comparison experiments one after another with `dvc exp run --temp`
# (Lecture 11). Each starts from HEAD with only its own -S overrides.
# Usage: scripts/run_experiments.sh [name ...]   (default: all)
# Needs the MLflow server on MLFLOW_TRACKING_URI (default http://127.0.0.1:5000).
set -u
cd "$(dirname "$0")/.."
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"
export MLFLOW_DISABLE_AGENT_HINT=1 TF_CPP_MIN_LOG_LEVEL=2
LOG=results/logs/day3_experiments.log

declare -A OPTS=(
  [lr5e-4]="-S train.learning_rate=0.0005 -S train.patience=6"
  [wider]="-S train.filters=[32,64,128,256] -S train.dense_units=256"
  [no-augment]="-S train.augment=false"
  [img96]="-S preprocess.image_size=96"
)
ORDER=(lr5e-4 wider no-augment img96)   # img96 last: about 2.5x slower per epoch

for name in "${@:-${ORDER[@]}}"; do
  echo "=== exp $name (${OPTS[$name]}) start $(date +%T)" >> "$LOG"
  # shellcheck disable=SC2086
  dvc exp run --temp -n "$name" ${OPTS[$name]} >> "$LOG" 2>&1
  echo "=== exp $name exit=$? end $(date +%T)" >> "$LOG"
done
echo "=== all experiments done $(date +%T)" >> "$LOG"
