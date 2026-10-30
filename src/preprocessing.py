from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

COMMON_REPLACEMENTS = {
    "avvatledu": "not working",
    "avatledu": "not working",
    "work avvatledu": "not working",
    "work avvale": "not working",
    "spoil avtunnayi": "spoiling",
    "spoil avvutondi": "spoiling",
    "chala": "very",
    "baaga": "very",
    "illai": "not",
    "irukku": "is",
    "wrong label": "incorrect label",
    "kharab": "bad",
    "thevayillai": "not needed",
    "matta": "dirty",
    "saaf": "clean",
    "coupon not work": "promotion mismatch",
    "payment not going": "payment failure",
    "payment keeps failing": "payment failure",
    "no stock": "stock shortage",
    "empty rack": "stock shortage",
    "packet bad": "damaged package",
    "freezer work avvatledu": "freezer not working",
    "customer complaint": "feedback"
}

NORMALIZATION_REPLACEMENTS = {
    "&": " and ",
    "@": " ",
    "#": " ",
    "\n": " ",
    "\t": " ",
}


def normalize_text(value: object) -> str:
    """Return a lowercase, punctuation-light representation for rules and duplicates."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    for src, dst in NORMALIZATION_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    for src, dst in COMMON_REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_parse_date(value: object, default: pd.Timestamp | None = None) -> pd.Timestamp | pd.NaT:
    default = default or pd.NaT
    if value is None or value == "":
        return default
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=False)
        if pd.isna(parsed):
            return default
        return parsed.tz_localize(None) if getattr(parsed, "tzinfo", None) else parsed
    except Exception:
        return default


REQUIRED_UPLOAD_COLUMNS = {
    "text": ["feedback_text", "feedback", "comment", "text", "message"],
    "date": ["timestamp", "date", "created_at", "feedback_date"],
    "store_id": ["store_id", "store", "store code", "location_id"],
    "region": ["region", "zone", "cluster"],
    "channel": ["channel", "source", "platform"],
}


def infer_required_columns(columns: Iterable[str]) -> dict[str, str | None]:
    lookup = {normalize_text(col): col for col in columns}
    mapping: dict[str, str | None] = {}
    for target, candidates in REQUIRED_UPLOAD_COLUMNS.items():
        found = None
        for candidate in candidates:
            normalized = normalize_text(candidate)
            if normalized in lookup:
                found = lookup[normalized]
                break
        mapping[target] = found
    return mapping


def standardize_feedback_frame(
    df: pd.DataFrame,
    *,
    text_column: str,
    date_column: str,
    store_column: str,
    region_column: str,
    channel_column: str,
) -> pd.DataFrame:
    """Clean uploaded or synthetic feedback into a consistent schema."""
    frame = df.copy()
    rename_map = {
        text_column: "feedback_text",
        date_column: "timestamp",
        store_column: "store_id",
        region_column: "region",
        channel_column: "channel",
    }
    frame = frame.rename(columns=rename_map)

    for column in ["feedback_text", "store_id", "region", "channel"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    frame["timestamp"] = frame["timestamp"].apply(safe_parse_date)
    frame = frame.loc[~frame["timestamp"].isna()].copy()
    frame["week_start"] = frame["timestamp"].dt.to_period("W-MON").dt.start_time
    frame["feedback_text_normalized"] = frame["feedback_text"].apply(normalize_text)
    frame["store_id"] = frame["store_id"].fillna("UNKNOWN").astype(str).str.strip()
    frame["region"] = frame["region"].fillna("Unknown").astype(str).str.strip()
    frame["channel"] = frame["channel"].fillna("Unknown").astype(str).str.strip()
    frame = frame.loc[frame["feedback_text"].str.strip().ne("")].copy()
    return frame


def week_label(value: object) -> str:
    parsed = safe_parse_date(value)
    if pd.isna(parsed):
        return "Unknown"
    return pd.Timestamp(parsed).to_period("W-MON").start_time.strftime("%Y-%m-%d")


def compact_text(value: object, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"
