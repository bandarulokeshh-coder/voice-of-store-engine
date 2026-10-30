import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(folder):
    feedback = pd.read_csv(folder / "feedback.csv")
    stores = pd.read_csv(folder / "store_master.csv")
    labels = pd.read_csv(folder / "feedback_labels_train.csv")
    weeks = pd.read_csv(folder / "week_calendar.csv")
    taxonomy = pd.read_csv(folder / "issue_taxonomy.csv")
    return feedback, stores, labels, weeks, taxonomy


def poisson_surprise(count, baseline, store_share):
    if baseline == 0:
        return float("inf") if count > 0 else 0
    expected = baseline * store_share
    if expected == 0:
        return float("inf") if count > 0 else 0
    p = 1 - np.exp(-expected)
    return -np.log(p) if p > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", required=True, type=Path)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    folder = args.feedback.parent
    feedback, _stores, labels, _weeks, _taxonomy = load_data(folder)

    asof = pd.Timestamp(args.as_of)
    asof_monday = asof - pd.Timedelta(days=asof.weekday())

    feedback["created_at"] = pd.to_datetime(feedback["created_at"], errors="coerce")
    feedback = feedback[feedback["created_at"] <= asof_monday]

    df = feedback.merge(
        labels[["feedback_id", "issue_type_id"]], on="feedback_id", how="left"
    )

    results = []
    for (store_id, issue_type), group in df.groupby(["store_id", "issue_type_id"]):
        total = len(group)
        baseline = total / len(df) if len(df) > 0 else 0
        store_share = total / len(df) if len(df) > 0 else 0
        surprise = poisson_surprise(total, baseline, store_share)
        if surprise >= 2.5 and total >= 2:
            results.append(
                {
                    "store_id": store_id,
                    "issue_type_id": issue_type,
                    "count": total,
                    "surprise": surprise,
                }
            )

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.sort_values("surprise", ascending=False)
        result_df = result_df.groupby("store_id").head(3)

    output = result_df.copy()
    output["as_of_week_start"] = args.as_of
    output = output[["as_of_week_start", "store_id", "issue_type_id", "surprise"]]
    output.columns = ["as_of_week_start", "store_id", "issue_type_id", "confidence"]
    output["evidence_feedback_ids"] = (
        ";".join(output["store_id"].tolist()) if len(output) > 0 else ""
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"Detected {len(output)} flags in {args.out}")


if __name__ == "__main__":
    main()
