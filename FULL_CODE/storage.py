from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "storepulse.sqlite"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    store_id TEXT,
    region TEXT,
    store_size TEXT,
    timestamp TEXT,
    channel TEXT,
    feedback_text TEXT,
    issue_category TEXT,
    severity TEXT,
    actionable INTEGER,
    department TEXT,
    dataset_mode TEXT,
    incident_tag TEXT,
    known_incident_flag INTEGER,
    language_hint TEXT,
    feedback_text_normalized TEXT,
    week_start TEXT,
    store_week_id TEXT,
    week_label TEXT
);
"""


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL;")
    return connection



def initialise_database(df: pd.DataFrame | None = None, db_path: str | Path = DB_PATH) -> Path:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        if df is not None and not df.empty:
            upsert_feedback_frame(df, connection=connection)
    return Path(db_path)



def upsert_feedback_frame(df: pd.DataFrame, connection: sqlite3.Connection | None = None, db_path: str | Path = DB_PATH) -> None:
    frame = df.copy()
    if frame.empty:
        return
    close_after = False
    if connection is None:
        connection = connect(db_path)
        close_after = True
    try:
        frame = frame.copy()
        for column in ["timestamp", "week_start"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").astype(str)
        for column in ["actionable", "known_incident_flag"]:
            if column in frame.columns:
                frame[column] = frame[column].astype(int)
        frame.to_sql("feedback", connection, if_exists="replace", index=False)
    finally:
        if close_after:
            connection.close()



def load_feedback_frame(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    with connect(db_path) as connection:
        try:
            frame = pd.read_sql_query("SELECT * FROM feedback", connection)
        except Exception:
            return pd.DataFrame()
    if not frame.empty:
        for column in ["timestamp", "week_start"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        for column in ["actionable", "known_incident_flag"]:
            if column in frame.columns:
                frame[column] = frame[column].astype(bool)
    return frame
