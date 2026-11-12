"""Snapshot or restore the MLflow store, so a fresh clone has a model to serve.

mlflow.db and mlartifacts/ are not in git, a copy is kept with dvc in data/mlflow_store.

  python scripts/mlflow_store.py snapshot && dvc add data/mlflow_store && dvc push
  dvc pull && python scripts/mlflow_store.py restore     # after a fresh clone
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "mlflow.db"
ARTIFACTS = ROOT / "mlartifacts"
SNAPSHOT = ROOT / "data" / "mlflow_store"


def snapshot():
    # sqlite backup api, safe while the server is running
    if not DB.exists() or not ARTIFACTS.exists():
        sys.exit("No MLflow store here to snapshot.")
    shutil.rmtree(SNAPSHOT, ignore_errors=True)
    SNAPSHOT.mkdir(parents=True)
    src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    dst = sqlite3.connect(SNAPSHOT / DB.name)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    shutil.copytree(ARTIFACTS, SNAPSHOT / ARTIFACTS.name)
    print(f"Snapshot written to {SNAPSHOT}")


def restore(force):
    """Copy the snapshot back; never replaces a live store unless forced."""
    if not (SNAPSHOT / DB.name).exists():
        sys.exit("No snapshot found. Run `dvc pull data/mlflow_store` first.")
    if (DB.exists() or ARTIFACTS.exists()) and not force:
        sys.exit("An MLflow store already exists. Use --force to replace it.")
    shutil.rmtree(ARTIFACTS, ignore_errors=True)
    shutil.copy2(SNAPSHOT / DB.name, DB)
    shutil.copytree(SNAPSHOT / ARTIFACTS.name, ARTIFACTS)
    print("MLflow store restored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["snapshot", "restore"])
    parser.add_argument("--force", action="store_true", help="replace a live store")
    args = parser.parse_args()
    snapshot() if args.action == "snapshot" else restore(args.force)


if __name__ == "__main__":
    main()
