from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from preprocessing import compact_text


@dataclass(slots=True)
class EmergingAlert:
    store_id: str
    region: str
    issue_category: str
    severity: str
    department: str
    current_count: int
    normalized_rate: float
    previous_baseline: float
    growth_indicator: float
    spike_score: float
    alert_date: pd.Timestamp
    explanation: str
    supporting_quotes: list[str]
    total_store_feedback: int
    current_week: pd.Timestamp

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "region": self.region,
            "issue_category": self.issue_category,
            "severity": self.severity,
            "department": self.department,
            "current_count": self.current_count,
            "normalized_rate": self.normalized_rate,
            "previous_baseline": self.previous_baseline,
            "growth_indicator": self.growth_indicator,
            "spike_score": self.spike_score,
            "alert_date": self.alert_date,
            "explanation": self.explanation,
            "supporting_quotes": self.supporting_quotes,
            "total_store_feedback": self.total_store_feedback,
            "current_week": self.current_week,
        }


def ensure_week_start(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "week_start" not in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.loc[~frame["timestamp"].isna()].copy()
        frame["week_start"] = frame["timestamp"].dt.to_period("W-MON").dt.start_time
    frame["week_start"] = pd.to_datetime(frame["week_start"], errors="coerce")
    frame = frame.loc[~frame["week_start"].isna()].copy()
    return frame


def build_weekly_store_issue_metrics(df: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_week_start(df)
    frame = frame.copy()
    frame["feedback_text"] = frame["feedback_text"].fillna("").astype(str)
    group_cols = ["store_id", "region", "issue_category", "week_start"]
    weekly = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            issue_count=("feedback_text", "size"),
            total_store_feedback=("feedback_text", lambda s: len(s)),
        )
        .reset_index()
    )
    total_by_store_week = (
        frame.groupby(["store_id", "region", "week_start"])
        .size()
        .reset_index(name="store_week_total")
    )
    weekly = weekly.merge(
        total_by_store_week, on=["store_id", "region", "week_start"], how="left"
    )
    weekly["normalized_rate"] = weekly["issue_count"] / weekly["store_week_total"].clip(
        lower=1
    )
    return weekly.sort_values(["store_id", "issue_category", "week_start"]).reset_index(
        drop=True
    )


def _robust_baseline(history: np.ndarray) -> tuple[float, float]:
    if history.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(history))
    mad = float(np.median(np.abs(history - baseline)))
    return baseline, mad


def _safe_growth(current_rate: float, baseline: float) -> float:
    if baseline <= 0:
        return float(current_rate / max(0.01, 1e-6))
    return float(current_rate / max(baseline, 1e-6))


def _safe_spike_score(current_rate: float, baseline: float, mad: float) -> float:
    if baseline <= 0 and mad == 0:
        return float(current_rate * 100.0)
    scale = max(1.4826 * mad, 1e-6)
    return float((current_rate - baseline) / scale)


def _supporting_quotes(group: pd.DataFrame, limit: int = 3) -> list[str]:
    quotes: list[str] = []
    sorted_group = group.sort_values("timestamp", ascending=False)
    for text in sorted_group["feedback_text"].fillna("").astype(str).tolist():
        if not text.strip():
            continue
        quotes.append(compact_text(text, 140))
        if len(quotes) >= limit:
            break
    return quotes


def detect_emerging_incidents(
    df: pd.DataFrame,
    *,
    minimum_support: int = 2,
    growth_threshold: float = 2.5,
    spike_threshold: float = 3.0,
    lookback_weeks: int = 6,
) -> pd.DataFrame:
    """Store-level, normalized-rate early warning detector.

    Rules:
    - Compare each store/category week against the previous lookback window.
    - Use median and MAD for a robust baseline.
    - Handle zero baselines safely using smoothed rate comparisons.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "store_id",
                "region",
                "issue_category",
                "severity",
                "department",
                "current_count",
                "normalized_rate",
                "previous_baseline",
                "growth_indicator",
                "spike_score",
                "alert_date",
                "explanation",
                "supporting_quotes",
                "total_store_feedback",
                "current_week",
            ]
        )

    frame = ensure_week_start(df)
    weekly = build_weekly_store_issue_metrics(frame)
    alerts: list[dict] = []

    for (store_id, issue_category), group in weekly.groupby(
        ["store_id", "issue_category"]
    ):
        group = group.sort_values("week_start")
        if len(group) < 2:
            continue
        for idx in range(len(group)):
            current_row = group.iloc[idx]
            history = group.iloc[max(0, idx - lookback_weeks) : idx]
            history_rates = history["normalized_rate"].astype(float).to_numpy()
            baseline, mad = _robust_baseline(history_rates)
            growth_indicator = _safe_growth(
                float(current_row["normalized_rate"]), baseline
            )
            spike_score = _safe_spike_score(
                float(current_row["normalized_rate"]), baseline, mad
            )
            current_count = int(current_row["issue_count"])
            current_rate = float(current_row["normalized_rate"])
            total_store_feedback = int(current_row["store_week_total"])
            alert_date = pd.Timestamp(current_row["week_start"])

            if current_count < minimum_support:
                continue

            zero_baseline = baseline <= 0 and history_rates.size > 0
            absolute_jump = current_rate - baseline
            exceeds_growth = growth_indicator >= growth_threshold
            exceeds_spike = spike_score >= spike_threshold
            early_low_volume = (
                zero_baseline
                and current_count >= minimum_support
                and absolute_jump > 0.015
            )

            if not (exceeds_growth or exceeds_spike or early_low_volume):
                continue

            current_group = frame.loc[
                (frame["store_id"] == store_id)
                & (frame["issue_category"] == issue_category)
                & (frame["week_start"] == current_row["week_start"])
            ]
            quotes = _supporting_quotes(current_group, limit=3)
            region = str(current_row.get("region", "Unknown"))
            severity = (
                str(current_group["severity"].mode().iloc[0])
                if "severity" in current_group.columns
                and not current_group["severity"].dropna().empty
                else "MEDIUM"
            )
            department = (
                str(current_group["department"].mode().iloc[0])
                if "department" in current_group.columns
                and not current_group["department"].dropna().empty
                else "Store Operations"
            )
            explanation = (
                f"{issue_category} complaints at {store_id} increased relative to the previous {len(history)} weeks. "
                f"Current rate {current_rate:.3f} vs baseline {baseline:.3f}; growth {growth_indicator:.2f}x."
            )
            if baseline <= 0:
                explanation += " Historical baseline was near zero, so a smoothed low-volume spike rule was used."
            alerts.append(
                EmergingAlert(
                    store_id=store_id,
                    region=region,
                    issue_category=issue_category,
                    severity=severity,
                    department=department,
                    current_count=current_count,
                    normalized_rate=current_rate,
                    previous_baseline=float(baseline),
                    growth_indicator=float(growth_indicator),
                    spike_score=float(spike_score),
                    alert_date=alert_date,
                    explanation=explanation,
                    supporting_quotes=quotes,
                    total_store_feedback=total_store_feedback,
                    current_week=alert_date,
                ).to_dict()
            )

    if not alerts:
        return pd.DataFrame(
            columns=[
                "store_id",
                "region",
                "issue_category",
                "severity",
                "department",
                "current_count",
                "normalized_rate",
                "previous_baseline",
                "growth_indicator",
                "spike_score",
                "alert_date",
                "explanation",
                "supporting_quotes",
                "total_store_feedback",
                "current_week",
            ]
        )

    alerts_df = (
        pd.DataFrame(alerts)
        .sort_values(
            ["alert_date", "growth_indicator", "current_count"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    return alerts_df


def build_trend_window(
    df: pd.DataFrame, store_id: str, issue_category: str, weeks: int = 8
) -> pd.DataFrame:
    frame = ensure_week_start(df)
    weekly = build_weekly_store_issue_metrics(frame)
    subset = weekly[
        (weekly["store_id"] == store_id) & (weekly["issue_category"] == issue_category)
    ].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values("week_start")
    return subset.tail(weeks).reset_index(drop=True)


def summarize_alerts(alerts_df: pd.DataFrame) -> pd.DataFrame:
    if alerts_df.empty:
        return alerts_df
    result = alerts_df.copy()
    result["alert_label"] = np.where(
        result["severity"].astype(str).str.upper().eq("HIGH")
        | (result["growth_indicator"] >= 3),
        "HIGH PRIORITY",
        "WATCHLIST",
    )
    return result
