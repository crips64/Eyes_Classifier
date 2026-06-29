"""SQLite persistence for predictions, drift alerts, and retraining events."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(os.getenv("DATABASE_PATH", "data/predictions.db"))


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column(conn: sqlite3.Connection, table: str, definition: str) -> None:
    name = definition.split()[0]
    if name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def init_db(db_path: Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                stored_path TEXT,
                score REAL NOT NULL,
                label TEXT NOT NULL,
                true_label TEXT,
                is_anomaly INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                mode TEXT NOT NULL,
                trigger_type TEXT NOT NULL DEFAULT 'manual',
                epochs INTEGER,
                model_version TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                json_path TEXT NOT NULL,
                html_path TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        # Forward-compatible migration from the original course project database.
        _add_column(conn, "predictions", "stored_path TEXT")
        _add_column(conn, "predictions", "true_label TEXT")
        _add_column(conn, "retrain_events", "job_id TEXT")
        _add_column(conn, "retrain_events", "trigger_type TEXT NOT NULL DEFAULT 'manual'")
        _add_column(conn, "retrain_events", "epochs INTEGER")
        _add_column(conn, "retrain_events", "model_version TEXT")
        conn.commit()


def save_prediction(
    *,
    filename: str,
    stored_path: str,
    score: float,
    label: str,
    true_label: str | None,
    is_anomaly: bool,
    created_at: str,
    model_version: str,
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (filename, stored_path, score, label, true_label, is_anomaly, created_at,
                 model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                stored_path,
                score,
                label,
                true_label,
                int(is_anomaly),
                created_at,
                model_version,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_predictions(
    *,
    limit: int = 50,
    offset: int = 0,
    label: str | None = None,
    anomaly: bool | None = None,
    db_path: Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    clauses: list[str] = []
    values: list[Any] = []
    if label:
        clauses.append("label = ?")
        values.append(label)
    if anomaly is not None:
        clauses.append("is_anomaly = ?")
        values.append(int(anomaly))
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as conn:
        total = int(
            conn.execute(f"SELECT COUNT(*) FROM predictions {where}", values).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT id, filename, stored_path, score, label, true_label, is_anomaly,
                   created_at, model_version
            FROM predictions
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            [*values, limit, offset],
        ).fetchall()
    return (
        [
            {
                **dict(row),
                "prediction_id": row["id"],
                "is_anomaly": bool(row["is_anomaly"]),
                "is_correct": (
                    row["label"] == row["true_label"] if row["true_label"] is not None else None
                ),
            }
            for row in rows
        ],
        total,
    )


def labeled_prediction_stats(db_path: Path | None = None) -> tuple[int, int]:
    """Return persisted labeled/correct counts used to restore quality metrics."""
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS labeled,
                COALESCE(SUM(CASE WHEN label = true_label THEN 1 ELSE 0 END), 0) AS correct
            FROM predictions
            WHERE true_label IS NOT NULL
            """
        ).fetchone()
    return int(row["labeled"]), int(row["correct"])


def save_retrain_event(
    *,
    status: str,
    message: str,
    mode: str,
    created_at: str,
    job_id: str | None = None,
    trigger_type: str = "manual",
    epochs: int | None = None,
    model_version: str | None = None,
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO retrain_events
                (job_id, status, message, mode, trigger_type, epochs, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                status,
                message,
                mode,
                trigger_type,
                epochs,
                model_version,
                created_at,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_retrain_events(
    limit: int = 20,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, job_id, status, message, mode, trigger_type, epochs, model_version,
                   created_at
            FROM retrain_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def latest_retrain_at(
    trigger_type: str = "auto",
    db_path: Path | None = None,
) -> datetime | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT created_at FROM retrain_events
            WHERE trigger_type = ? AND status IN ('started', 'completed')
            ORDER BY id DESC LIMIT 1
            """,
            (trigger_type,),
        ).fetchone()
    if row is None:
        return None
    return datetime.fromisoformat(str(row["created_at"]))


def save_drift_report_record(
    *,
    report_id: str,
    status: str,
    json_path: str,
    html_path: str,
    summary: dict[str, Any],
    created_at: str,
    db_path: Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO drift_reports
                (report_id, status, json_path, html_path, summary_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                status,
                json_path,
                html_path,
                json.dumps(summary, ensure_ascii=False),
                created_at,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_drift_reports(
    limit: int = 20,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT report_id, status, json_path, html_path, summary_json, created_at
            FROM drift_reports
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "report_id": row["report_id"],
            "status": row["status"],
            "json_path": row["json_path"],
            "html_path": row["html_path"],
            "summary": json.loads(row["summary_json"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def save_alert(
    *,
    level: str,
    category: str,
    message: str,
    details: dict[str, Any] | None = None,
    created_at: str | None = None,
    db_path: Path | None = None,
) -> int:
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO alerts (level, category, message, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                level,
                category,
                message,
                json.dumps(details or {}, ensure_ascii=False),
                timestamp,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_alerts(
    *,
    limit: int = 50,
    unacknowledged_only: bool = False,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    where = "WHERE acknowledged = 0" if unacknowledged_only else ""
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT id, level, category, message, details_json, acknowledged, created_at
            FROM alerts {where}
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "level": row["level"],
            "category": row["category"],
            "message": row["message"],
            "details": json.loads(row["details_json"] or "{}"),
            "acknowledged": bool(row["acknowledged"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def acknowledge_alert(alert_id: int, db_path: Path | None = None) -> bool:
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
            (alert_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
