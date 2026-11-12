"""Roll @champion back to @previous_champion.

only the aliases move. the model server notices within about 15 s (serve.py).
"""

from mlflow import MlflowClient

from malariascope.common import get_logger
from malariascope.promote import CHAMPION, PREVIOUS, version_by_alias
from malariascope.tracking import MODEL_NAME, tracking_uri

log = get_logger("rollback")


def rollback(client):
    current = version_by_alias(client, CHAMPION)
    previous = version_by_alias(client, PREVIOUS)
    if previous is None:
        raise SystemExit("no previous_champion alias set, nothing to roll back to")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous.version)
    if current is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS, current.version)
    return (current.version if current else None), previous.version


def main():
    client = MlflowClient(tracking_uri=tracking_uri())
    old, new = rollback(client)
    log.info("champion rolled back from version %s to version %s", old, new)


if __name__ == "__main__":
    main()
