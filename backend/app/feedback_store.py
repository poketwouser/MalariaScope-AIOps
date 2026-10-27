"""Feedback store: SQLite metadata plus one PNG per feedback (LLD section 4).

Images are plain files under images/<correct_label>/ so the retraining DAG can
snapshot the folder with `dvc add` and train on it like the original data.
"""

import io
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  ts              REAL    NOT NULL,
  request_id      TEXT    NOT NULL,
  predicted_label TEXT    NOT NULL,
  correct_label   TEXT    NOT NULL,
  p_parasitized   REAL    NOT NULL,
  model_version   TEXT    NOT NULL,
  image_path      TEXT    NOT NULL,
  exported        INTEGER NOT NULL DEFAULT 0
)
"""


class FeedbackStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.root / "feedback.db", check_same_thread=False)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def add(
        self,
        image,
        request_id,
        predicted_label,
        correct_label,
        p_parasitized,
        model_version,
    ):
        """Store one feedback item; return (feedback_id, total_count)."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO feedback (ts, request_id, predicted_label, correct_label,"
                " p_parasitized, model_version, image_path) VALUES (?,?,?,?,?,?,'')",
                (
                    time.time(),
                    request_id,
                    predicted_label,
                    correct_label,
                    p_parasitized,
                    model_version,
                ),
            )
            fid = cur.lastrowid
            rel = Path("images") / correct_label / f"{fid:06d}.png"
            try:
                (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.root / rel).write_bytes(buf.getvalue())
            except OSError:
                self._conn.rollback()
                raise
            self._conn.execute(
                "UPDATE feedback SET image_path=? WHERE id=?", (str(rel), fid)
            )
            self._conn.commit()
            total = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        return fid, total

    def stats(self):
        with self._lock:
            total, agreed, pending = self._conn.execute(
                "SELECT COUNT(*),"
                " COALESCE(SUM(predicted_label = correct_label), 0),"
                " COALESCE(SUM(exported = 0), 0) FROM feedback"
            ).fetchone()
        return {
            "total": total,
            "agreed": agreed,
            "disagreed": total - agreed,
            "since_last_export": pending,
        }
