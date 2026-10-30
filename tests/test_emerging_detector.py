import pandas as pd

from src.emerging_detector import detect_emerging_incidents


def _weekly_rows(store_id: str, region: str, issue_category: str, weekly_counts: list[int], other_count: int = 20):
    rows = []
    base = pd.Timestamp("2026-01-05")
    for idx, count in enumerate(weekly_counts):
        week_start = base + pd.Timedelta(weeks=idx)
        for _ in range(other_count):
            rows.append(
                {
                    "store_id": store_id,
                    "region": region,
                    "issue_category": "Customer Praise",
                    "severity": "LOW",
                    "actionable": False,
                    "department": "Customer Experience",
                    "timestamp": week_start + pd.Timedelta(days=1),
                    "feedback_text": "Great shopping experience.",
                }
            )
        for item in range(count):
            rows.append(
                {
                    "store_id": store_id,
                    "region": region,
                    "issue_category": issue_category,
                    "severity": "HIGH",
                    "actionable": True,
                    "department": "Store Operations",
                    "timestamp": week_start + pd.Timedelta(days=2, hours=item),
                    "feedback_text": f"{issue_category} complaint {idx}-{item}",
                }
            )
    return rows


def test_refrigeration_spike_triggers_alert():
    df = pd.DataFrame(_weekly_rows("ST-042", "Visakhapatnam", "Refrigeration Failure", [0, 0, 1, 2, 4, 7], other_count=18))
    alerts = detect_emerging_incidents(df, minimum_support=2, growth_threshold=2.5, spike_threshold=3.0)
    assert not alerts.empty
    assert (alerts["issue_category"] == "Refrigeration Failure").any()


def test_low_volume_store_normalized_trend_triggers_alert():
    df = pd.DataFrame(_weekly_rows("ST-031", "Bengaluru", "Stock Shortage", [0, 0, 1, 1, 2, 3], other_count=5))
    alerts = detect_emerging_incidents(df, minimum_support=2, growth_threshold=2.0, spike_threshold=2.5)
    assert not alerts.empty
    assert (alerts["store_id"] == "ST-031").any()


def test_missing_data_does_not_crash():
    df = pd.DataFrame(
        [
            {
                "store_id": "ST-001",
                "region": "Hyderabad",
                "issue_category": "Hygiene",
                "severity": "HIGH",
                "actionable": True,
                "department": "Store Operations",
                "timestamp": "not-a-date",
                "feedback_text": None,
            },
            {
                "store_id": "ST-001",
                "region": "Hyderabad",
                "issue_category": "Hygiene",
                "severity": "HIGH",
                "actionable": True,
                "department": "Store Operations",
                "timestamp": "2026-01-05",
                "feedback_text": "Dirty aisle",
            },
        ]
    )
    alerts = detect_emerging_incidents(df)
    assert alerts is not None
