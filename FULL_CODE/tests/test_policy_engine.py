from classifier import FeedbackAnalyzer
from policy_engine import evaluate_policy


def test_safety_escalation_allergen_label_is_high():
    decision = evaluate_policy(
        "Incorrect allergen label on packaged food.", "Allergen Label Error"
    )
    assert decision.severity == "HIGH"
    assert decision.department == "Merchandising"
    assert decision.actionable is True


def test_unknown_feedback_does_not_fabricate_critical_issue():
    analyzer = FeedbackAnalyzer()
    result = analyzer.analyze("Good shopping experience.")
    assert result.severity == "LOW"
    assert result.actionable is False
    assert result.issue_category in {"Customer Praise", "General Feedback"}


def test_missing_or_empty_text_is_handled_safely():
    analyzer = FeedbackAnalyzer()
    result = analyzer.analyze("")
    assert result.issue_category == "General Feedback"
    assert result.human_review_required is True
