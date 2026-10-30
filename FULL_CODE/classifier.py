from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from policy_engine import evaluate_policy
from preprocessing import normalize_text
from routing import category_keywords, get_department_for_issue, get_suggested_action

GENERAL_FEEDBACK_CATEGORY = "General Feedback"
PRAISE_CATEGORY = "Customer Praise"

POSITIVE_HINTS = [
    "good",
    "great",
    "excellent",
    "nice",
    "happy",
    "helpful",
    "love",
    "thanks",
    "thank you",
]
NEGATIVE_HINTS = [
    "bad",
    "issue",
    "problem",
    "broken",
    "failed",
    "delay",
    "dirty",
    "wrong",
    "not working",
]


@dataclass(slots=True)
class AnalysisResult:
    issue_category: str
    severity: str
    actionable: bool
    department: str
    suggested_action: str
    explanation: str
    method: str
    confidence: float | None = None
    rule_matches: list[str] | None = None
    human_review_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_category": self.issue_category,
            "severity": self.severity,
            "actionable": self.actionable,
            "department": self.department,
            "suggested_action": self.suggested_action,
            "explanation": self.explanation,
            "method": self.method,
            "confidence": self.confidence,
            "rule_matches": self.rule_matches or [],
            "human_review_required": self.human_review_required,
        }


@dataclass(slots=True)
class ModelBundle:
    issue_model: Pipeline | None = None
    issue_labels: list[str] | None = None
    trained_on_demo_data: bool = False
    notes: str = "rule_based"


class FeedbackAnalyzer:
    def __init__(self, taxonomy_path: str | None = None):
        self.taxonomy_path = taxonomy_path
        self.keywords = (
            category_keywords(taxonomy_path) if taxonomy_path else category_keywords()
        )
        self.model_bundle = ModelBundle()

    def fit_from_dataframe(self, df: pd.DataFrame) -> None:
        if "issue_category" not in df.columns:
            return
        training_df = df.loc[
            df["feedback_text"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        if training_df.empty:
            return
        if training_df["issue_category"].nunique() < 2:
            return
        try:
            model = Pipeline(
                steps=[
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            ngram_range=(1, 2), min_df=2, max_features=8000
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=500, class_weight="balanced", multi_class="auto"
                        ),
                    ),
                ]
            )
            model.fit(
                training_df["feedback_text"].astype(str),
                training_df["issue_category"].astype(str),
            )
            self.model_bundle = ModelBundle(
                issue_model=model,
                issue_labels=sorted(
                    training_df["issue_category"].astype(str).unique().tolist()
                ),
                trained_on_demo_data=True,
                notes="trained_tfidf_logreg",
            )
        except Exception:  # noqa: BLE001
            self.model_bundle = ModelBundle(notes="rule_based_fallback")

    def _score_rules(self, text: str) -> tuple[str, float, list[str]]:
        normalized = normalize_text(text)
        best_category = GENERAL_FEEDBACK_CATEGORY
        best_score = 0.0
        matches: list[str] = []
        for category, keywords in self.keywords.items():
            hits = [
                keyword for keyword in keywords if keyword and keyword in normalized
            ]
            score = float(len(hits))
            if score > best_score:
                best_score = score
                best_category = category
                matches = hits
            elif score == best_score and score > 0 and category != best_category:
                # Prefer more specific categories with safety relevance.
                if any(
                    token in category.lower()
                    for token in [
                        "safety",
                        "contamination",
                        "allergen",
                        "hygiene",
                        "misconduct",
                    ]
                ):
                    best_category = category
                    matches = hits
        if best_score == 0:
            if any(token in normalized for token in POSITIVE_HINTS) and not any(
                token in normalized for token in NEGATIVE_HINTS
            ):
                return PRAISE_CATEGORY, 0.15, []
            return GENERAL_FEEDBACK_CATEGORY, 0.0, []
        confidence = min(0.95, 0.35 + (best_score * 0.2))
        return best_category, confidence, matches

    def _predict_with_model(self, text: str) -> tuple[str, float]:
        if self.model_bundle.issue_model is None:
            raise RuntimeError("model_not_available")
        proba = self.model_bundle.issue_model.predict_proba([text])[0]
        classes = list(self.model_bundle.issue_model.classes_)
        idx = int(np.argmax(proba))
        return str(classes[idx]), float(proba[idx])

    def analyze(
        self, text: str, issue_category_hint: str | None = None
    ) -> AnalysisResult:
        original_text = text or ""
        normalized = normalize_text(original_text)
        if not normalized:
            decision = evaluate_policy(original_text, issue_category_hint)
            return AnalysisResult(
                issue_category=GENERAL_FEEDBACK_CATEGORY,
                severity="LOW",
                actionable=False,
                department=decision.department,
                suggested_action="Request more detail from the reporter.",
                explanation="Missing or empty feedback cannot be classified reliably.",
                method="rule_based",
                confidence=None,
                rule_matches=[],
                human_review_required=True,
            )

        rule_category, rule_confidence, rule_matches = self._score_rules(original_text)
        model_category = None
        model_confidence = None
        used_model = False

        if self.model_bundle.issue_model is not None:
            try:
                model_category, model_confidence = self._predict_with_model(
                    original_text
                )
                used_model = True
            except Exception:  # noqa: BLE001
                used_model = False

        if rule_category == GENERAL_FEEDBACK_CATEGORY and used_model and model_category:
            issue_category = model_category
            confidence = model_confidence
            method = "ml_fallback"
        elif rule_category == GENERAL_FEEDBACK_CATEGORY:
            issue_category = rule_category
            confidence = None
            method = "rule_based"
        else:
            issue_category = rule_category
            confidence = rule_confidence
            method = "hybrid" if used_model else "rule_based"

        decision = evaluate_policy(original_text, issue_category)
        actionable = decision.actionable
        department = decision.department or get_department_for_issue(
            issue_category, original_text
        )
        suggested_action = get_suggested_action(issue_category)

        if issue_category == PRAISE_CATEGORY:
            severity = "LOW"
            actionable = False
            department = "Customer Experience"
            suggested_action = "Share the positive feedback with the store team."
            explanation = (
                "Positive language detected; no operational issue was identified."
            )
            return AnalysisResult(
                issue_category=issue_category,
                severity=severity,
                actionable=actionable,
                department=department,
                suggested_action=suggested_action,
                explanation=explanation,
                method=method,
                confidence=confidence,
                rule_matches=rule_matches,
                human_review_required=False,
            )

        # Respect policy-first escalations even if the classifier predicts something else.
        severity = decision.severity
        explanation_parts = []
        if rule_matches:
            explanation_parts.append(f"Keyword matches: {', '.join(rule_matches[:4])}.")
        if decision.matched_rules:
            explanation_parts.append(
                f"Policy trigger(s): {', '.join(decision.matched_rules)}."
            )
        if decision.human_review_required:
            explanation_parts.append(
                "Ambiguous safety language flagged for human review."
            )
        if confidence is not None and used_model and method != "rule_based":
            explanation_parts.append(f"Model confidence: {confidence:.2f}.")
        if not explanation_parts:
            explanation_parts.append(
                "No strong rule match; operating with fallback classification."
            )

        return AnalysisResult(
            issue_category=issue_category,
            severity=severity,
            actionable=actionable,
            department=department,
            suggested_action=suggested_action,
            explanation=" ".join(explanation_parts),
            method=method,
            confidence=confidence
            if issue_category != GENERAL_FEEDBACK_CATEGORY
            else None,
            rule_matches=rule_matches,
            human_review_required=decision.human_review_required,
        )

    def batch_analyze(self, texts: list[str]) -> pd.DataFrame:
        rows = [self.analyze(text).to_dict() for text in texts]
        return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def default_analyzer() -> FeedbackAnalyzer:
    return FeedbackAnalyzer()
