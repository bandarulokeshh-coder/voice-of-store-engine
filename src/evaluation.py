from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .classifier import FeedbackAnalyzer
from .emerging_detector import detect_emerging_incidents
from .synthetic_data import incident_profiles_dataframe


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision: float
    recall: float
    confusion: pd.DataFrame
    n_samples: int
    demo_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "precision": self.precision,
            "recall": self.recall,
            "confusion": self.confusion,
            "n_samples": self.n_samples,
            "demo_only": self.demo_only,
        }



def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> ClassificationMetrics:
    labels = sorted(set(y_true) | set(y_pred))
    accuracy = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0
    precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0
    recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0
    matrix = confusion_matrix(y_true, y_pred, labels=labels) if y_true else np.zeros((0, 0), dtype=int)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        precision=precision,
        recall=recall,
        confusion=confusion,
        n_samples=len(y_true),
        demo_only=True,
    )



def evaluate_feedback_classification(df: pd.DataFrame, analyzer: FeedbackAnalyzer, max_rows: int | None = 600) -> dict[str, ClassificationMetrics]:
    frame = df.copy()
    if frame.empty:
        empty = compute_classification_metrics([], [])
        return {"issue_type": empty, "severity": empty, "actionability": empty}

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=42)

    predictions = frame["feedback_text"].astype(str).apply(analyzer.analyze)
    predicted_issue = [result.issue_category for result in predictions]
    predicted_severity = [result.severity for result in predictions]
    predicted_actionable = [bool(result.actionable) for result in predictions]

    issue_metrics = compute_classification_metrics(frame["issue_category"].astype(str).tolist(), predicted_issue)
    severity_metrics = compute_classification_metrics(frame["severity"].astype(str).tolist(), predicted_severity)
    action_metrics = compute_classification_metrics(frame["actionable"].astype(bool).astype(str).tolist(), [str(value) for value in predicted_actionable])
    return {"issue_type": issue_metrics, "severity": severity_metrics, "actionability": action_metrics}



def evaluate_emerging_detection(df: pd.DataFrame, alerts_df: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.loc[~frame["timestamp"].isna()].copy()
    if alerts_df is None:
        alerts_df = detect_emerging_incidents(frame)

    incident_profiles = incident_profiles_dataframe()
    detected_incidents = 0
    lead_times: list[float] = []
    matched_alerts = set()

    for _, incident in incident_profiles.iterrows():
        store_id = incident["store_id"]
        category = incident["issue_category"]
        start_date = pd.Timestamp(incident["incident_start_date"])
        matching_alerts = alerts_df[(alerts_df["store_id"] == store_id) & (alerts_df["issue_category"] == category)]
        matching_alerts = matching_alerts[pd.to_datetime(matching_alerts["alert_date"]) >= start_date]
        if not matching_alerts.empty:
            first_alert = pd.to_datetime(matching_alerts["alert_date"]).min()
            detected_incidents += 1
            lead_times.append(float((first_alert - start_date).days))
            matched_alerts.add((store_id, category))

    false_alerts = 0
    for _, alert in alerts_df.iterrows():
        key = (alert["store_id"], alert["issue_category"])
        if key not in {(row.store_id, row.issue_category) for row in incident_profiles.itertuples(index=False)}:
            false_alerts += 1
        elif key not in matched_alerts:
            false_alerts += 1

    total_escalations = len(alerts_df)
    true_actionable_escalations = int(alerts_df[alerts_df["severity"].astype(str).str.upper() == "HIGH"].shape[0])
    escalation_precision = None
    if total_escalations > 0:
        escalation_precision = true_actionable_escalations / total_escalations

    return {
        "true_incidents_detected": detected_incidents,
        "total_known_incidents": int(len(incident_profiles)),
        "median_detection_lead_time_days": float(np.median(lead_times)) if lead_times else None,
        "mean_detection_lead_time_days": float(np.mean(lead_times)) if lead_times else None,
        "false_alert_count": false_alerts,
        "total_alerts": total_escalations,
        "escalation_precision": escalation_precision,
        "note": "Demonstration-only evaluation using seeded synthetic incidents." if not alerts_df.empty else "No alerts generated.",
    }



def model_insights_summary(analyzer: FeedbackAnalyzer) -> dict[str, Any]:
    return {
        "classifier_mode": analyzer.model_bundle.notes,
        "trained_on_demo_data": analyzer.model_bundle.trained_on_demo_data,
        "supports_ml": analyzer.model_bundle.issue_model is not None,
        "fallback_mode": analyzer.model_bundle.issue_model is None,
        "limitations": [
            "Rule-based policy engine is intentionally used for safety escalation decisions.",
            "Synthetic demo data is labeled for demonstration purposes only.",
            "Uploaded official datasets may not include ground-truth labels, so some metrics may be unavailable.",
        ],
    }
