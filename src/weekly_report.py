from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from .emerging_detector import build_weekly_store_issue_metrics
from .preprocessing import compact_text
from .routing import get_suggested_action

SEVERITY_WEIGHT = {"LOW": 1.0, "MEDIUM": 2.5, "HIGH": 4.0}


@dataclass(slots=True)
class WeeklyRegionalReport:
    region: str
    week_start: pd.Timestamp
    report_date: pd.Timestamp
    total_feedback: int
    high_severity_issues: int
    emerging_alerts: int
    top_issues: pd.DataFrame
    overview: dict[str, Any]
    formula: str



def calculate_priority_score(
    severity: str,
    normalized_rate: float,
    recent_trend: float,
    actionable: bool,
) -> float:
    severity_score = SEVERITY_WEIGHT.get(str(severity).upper(), 2.0)
    rate_score = min(max(normalized_rate, 0.0) * 120.0, 35.0)
    trend_score = min(max(recent_trend, 0.0) * 15.0, 20.0)
    action_score = 12.0 if actionable else 0.0
    score = severity_score * 10.0 + rate_score + trend_score + action_score
    return round(min(score, 100.0), 2)



def priority_formula_description() -> str:
    return (
        "Priority score = severity weight × 10 + normalized rate contribution + recent trend contribution + actionability bonus. "
        "Severity weights are LOW=1, MEDIUM=2.5, HIGH=4. The score is capped at 100."
    )



def _trend_arrow(value: float) -> str:
    if value > 1.1:
        return "↑"
    if value < 0.9:
        return "↓"
    return "→"



def build_weekly_regional_report(
    df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    *,
    region: str,
    week_start: pd.Timestamp | None = None,
) -> WeeklyRegionalReport:
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.loc[~frame["timestamp"].isna()].copy()
    frame["week_start"] = frame["timestamp"].dt.to_period("W-MON").dt.start_time
    region_frame = frame[frame["region"] == region].copy()
    if region_frame.empty:
        week_start = week_start or pd.Timestamp.today().normalize().to_period("W-MON").start_time
        return WeeklyRegionalReport(
            region=region,
            week_start=week_start,
            report_date=pd.Timestamp.today().normalize(),
            total_feedback=0,
            high_severity_issues=0,
            emerging_alerts=0,
            top_issues=pd.DataFrame(),
            overview={"message": "No records found for this region."},
            formula=priority_formula_description(),
        )

    if week_start is None:
        week_start = region_frame["week_start"].max()
    week_start = pd.Timestamp(week_start)
    current_week = region_frame[region_frame["week_start"] == week_start].copy()
    previous_week = region_frame[region_frame["week_start"] == (week_start - pd.Timedelta(weeks=1))].copy()

    weekly_metrics = build_weekly_store_issue_metrics(region_frame)
    current_metrics = weekly_metrics[weekly_metrics["week_start"] == week_start].copy()
    previous_metrics = weekly_metrics[weekly_metrics["week_start"] == (week_start - pd.Timedelta(weeks=1))].copy()

    issue_rows = []
    for _, row in current_metrics.groupby("issue_category").agg(
        complaints=("issue_count", "sum"),
        normalized_rate=("normalized_rate", "mean"),
        affected_stores=("store_id", pd.Series.nunique),
    ).reset_index().iterrows():
        issue_category = row["issue_category"]
        current_subset = current_week[current_week["issue_category"] == issue_category]
        prev_subset = previous_week[previous_week["issue_category"] == issue_category]
        current_count = int(row["complaints"])
        prev_count = int(prev_subset.shape[0]) if not prev_subset.empty else 0
        trend = (current_count + 1) / (prev_count + 1)
        severity = str(current_subset["severity"].mode().iloc[0]) if not current_subset.empty else "MEDIUM"
        department = str(current_subset["department"].mode().iloc[0]) if not current_subset.empty else "Store Operations"
        actionable = bool(current_subset["actionable"].astype(bool).mean() >= 0.5) if not current_subset.empty else False
        score = calculate_priority_score(severity, float(row["normalized_rate"]), trend, actionable)
        issue_rows.append(
            {
                "Issue Type": issue_category,
                "Severity": severity,
                "Trend": f"{_trend_arrow(trend)} {trend:.2f}x",
                "Affected Stores": int(row["affected_stores"]),
                "Owning Department": department,
                "Complaints": current_count,
                "Normalized Rate": round(float(row["normalized_rate"]), 4),
                "Priority Score": score,
                "Suggested Action": get_suggested_action(issue_category),
                "Evidence Quotes": [compact_text(text, 120) for text in current_subset["feedback_text"].head(3).tolist()],
            }
        )

    top_issues = pd.DataFrame(issue_rows).sort_values("Priority Score", ascending=False).head(5).reset_index(drop=True)
    high_severity_issues = int(current_week[current_week["severity"].astype(str).str.upper() == "HIGH"].shape[0])
    emerging_alerts = int(alerts_df[(alerts_df["region"] == region) & (pd.to_datetime(alerts_df["alert_date"]).dt.to_period("W-MON").dt.start_time == week_start)].shape[0])

    overview = {
        "region": region,
        "reporting_week": week_start,
        "total_feedback": int(current_week.shape[0]),
        "high_severity_issues": high_severity_issues,
        "emerging_alerts": emerging_alerts,
        "top_store": current_week["store_id"].value_counts().idxmax() if not current_week.empty else None,
        "top_issue": current_week["issue_category"].value_counts().idxmax() if not current_week.empty else None,
    }

    return WeeklyRegionalReport(
        region=region,
        week_start=week_start,
        report_date=pd.Timestamp.today().normalize(),
        total_feedback=int(current_week.shape[0]),
        high_severity_issues=high_severity_issues,
        emerging_alerts=emerging_alerts,
        top_issues=top_issues,
        overview=overview,
        formula=priority_formula_description(),
    )



def export_weekly_report_csv(report: WeeklyRegionalReport, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report.top_issues.empty:
        pd.DataFrame([report.overview]).to_csv(output_path, index=False)
        return output_path
    csv_frame = report.top_issues.copy()
    csv_frame.insert(0, "Region", report.region)
    csv_frame.insert(1, "Reporting Week", report.week_start)
    csv_frame.to_csv(output_path, index=False)
    return output_path



def _write_header(canvas, doc, report: WeeklyRegionalReport) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 16)
    canvas.setFillColor(colors.HexColor("#2563EB"))
    canvas.drawString(2 * cm, A4[1] - 1.5 * cm, "StorePulse AI — Weekly Retail Intelligence Report")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(2 * cm, A4[1] - 2.0 * cm, f"Region: {report.region}   |   Reporting Week: {report.week_start:%Y-%m-%d}   |   Generated: {report.report_date:%Y-%m-%d}")
    canvas.restoreState()



def export_weekly_report_pdf(report: WeeklyRegionalReport, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="StorePulseTitle", fontSize=15, leading=18, textColor=colors.HexColor("#0F172A"), spaceAfter=8))
    styles.add(ParagraphStyle(name="StorePulseBody", fontSize=10, leading=13, textColor=colors.HexColor("#334155")))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=2.6 * cm,
        bottomMargin=1.4 * cm,
    )

    story = []
    story.append(Paragraph("StorePulse AI — Weekly Retail Intelligence Report", styles["StorePulseTitle"]))
    story.append(Paragraph(f"Region: <b>{report.region}</b>", styles["StorePulseBody"]))
    story.append(Paragraph(f"Reporting week: <b>{report.week_start:%Y-%m-%d}</b> | Generated: <b>{report.report_date:%Y-%m-%d}</b>", styles["StorePulseBody"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(report.formula, styles["StorePulseBody"]))
    story.append(Spacer(1, 0.3 * cm))

    overview_data = [
        ["Total Feedback", report.total_feedback, "High Severity", report.high_severity_issues],
        ["Emerging Alerts", report.emerging_alerts, "Top Issue", report.overview.get("top_issue", "—")],
    ]
    overview_table = Table(overview_data, colWidths=[4 * cm, 4 * cm, 4 * cm, 5 * cm])
    overview_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(overview_table)
    story.append(Spacer(1, 0.5 * cm))

    if not report.top_issues.empty:
        table_data = [["Issue Type", "Severity", "Trend", "Stores", "Department", "Complaints", "Priority"]]
        for _, row in report.top_issues.iterrows():
            table_data.append(
                [
                    row["Issue Type"],
                    row["Severity"],
                    row["Trend"],
                    str(int(row["Affected Stores"])),
                    row["Owning Department"],
                    str(int(row["Complaints"])),
                    f"{float(row['Priority Score']):.1f}",
                ]
            )
        issues_table = Table(table_data, repeatRows=1, colWidths=[4.2 * cm, 2.1 * cm, 1.8 * cm, 1.4 * cm, 3.3 * cm, 1.4 * cm, 1.6 * cm])
        issues_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("FONTSIZE", (0, 1), (-1, -1), 8.2),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
                ]
            )
        )
        story.append(Paragraph("Top Priority Issues", styles["StorePulseBody"]))
        story.append(issues_table)
        story.append(Spacer(1, 0.4 * cm))

        for _, row in report.top_issues.iterrows():
            story.append(Paragraph(f"<b>{row['Issue Type']}</b> — {row['Severity']} | Department: {row['Owning Department']}", styles["StorePulseBody"]))
            story.append(Paragraph(f"Suggested action: {row['Suggested Action']}", styles["StorePulseBody"]))
            if row.get("Evidence Quotes"):
                quotes = "<br/>".join([f"• {quote}" for quote in row["Evidence Quotes"]])
                story.append(Paragraph(f"Evidence: {quotes}", styles["StorePulseBody"]))
            story.append(Spacer(1, 0.2 * cm))

    doc.build(story, onFirstPage=lambda canvas, doc: _write_header(canvas, doc, report), onLaterPages=lambda canvas, doc: _write_header(canvas, doc, report))
    return output_path
