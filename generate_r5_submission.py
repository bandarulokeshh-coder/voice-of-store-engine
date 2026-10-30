"""
Generate R5 hackathon submission files - Direct TF-IDF + Logistic Regression approach.
"""
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import normalize_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score

# Paths
R5_DATA_DIR = Path(r"C:\Users\lokes\Downloads\R5_Voice_of_Store_Engine_v1.0\R5_Voice_of_Store_Engine\data")
R5_FEEDBACK = R5_DATA_DIR / "feedback.csv"
R5_LABELS_TRAIN = R5_DATA_DIR / "feedback_labels_train.csv"
R5_TAXONOMY = R5_DATA_DIR / "issue_taxonomy.csv"
R5_STORE_MASTER = R5_DATA_DIR / "store_master.csv"

OUTPUT_DIR = Path("r5_submission")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load all R5 data."""
    print("Loading R5 data...")
    feedback_df = pd.read_csv(R5_FEEDBACK)
    labels_df = pd.read_csv(R5_LABELS_TRAIN)
    taxonomy_df = pd.read_csv(R5_TAXONOMY)
    store_master = pd.read_csv(R5_STORE_MASTER)

    print(f"  Feedback: {len(feedback_df):,} records")
    print(f"  Training labels: {len(labels_df):,} records")
    print(f"  Taxonomy: {len(taxonomy_df)} issue types")

    # Merge feedback with labels
    train_df = feedback_df.merge(labels_df, on='feedback_id', how='inner')
    print(f"  Training samples: {len(train_df):,}")

    return feedback_df, train_df, taxonomy_df, store_master


def train_classifier(train_df):
    """Train TF-IDF + Logistic Regression classifier."""
    print("\nTraining TF-IDF + Logistic Regression classifier...")

    # Prepare training data
    X = train_df['text'].fillna("").astype(str)
    y_issue = train_df['issue_type_id'].astype(str)

    # TF-IDF with low min_df for small dataset
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        lowercase=True,
        strip_accents='unicode'
    )

    X_tfidf = vectorizer.fit_transform(X)

    # Train classifier
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs'
    )
    classifier.fit(X_tfidf, y_issue)

    # Cross-validation for baseline estimate
    print("  Running cross-validation...")
    cv_predictions = cross_val_predict(classifier, X_tfidf, y_issue, cv=5)
    cv_f1 = f1_score(y_issue, cv_predictions, average='macro')
    print(f"  CV Macro-F1: {cv_f1:.4f}")

    return vectorizer, classifier, cv_f1


def generate_predictions(vectorizer, classifier, feedback_df, taxonomy_df, train_df):
    """Generate predictions for all feedback."""
    print("\nGenerating predictions for all 39,769 feedback records...")

    X = feedback_df['text'].fillna("").astype(str)
    X_tfidf = vectorizer.transform(X)

    # Predict issue types
    issue_predictions = classifier.predict(X_tfidf)

    # Estimate severity and actionable based on training data patterns
    severity_map, actionable_map = compute_label_patterns(train_df)

    results = []
    for i, row in feedback_df.iterrows():
        issue_type_id = issue_predictions[i]

        # Get severity and actionable with fallback
        severity = severity_map.get(issue_type_id, 'low')
        actionable = actionable_map.get(issue_type_id, 0)

        # Apply policy floors
        severity = enforce_policy_floors(issue_type_id, severity)

        results.append({
            'feedback_id': row['feedback_id'],
            'issue_type_id': issue_type_id,
            'severity': severity,
            'actionable': actionable
        })

    predictions_df = pd.DataFrame(results)

    # Save R5_predictions.csv
    output_path = OUTPUT_DIR / "R5_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    print(f"  Rows: {len(predictions_df):,}")

    return predictions_df


def compute_label_patterns(train_df):
    """Compute severity and actionable patterns from training data."""
    severity_map = {}
    actionable_map = {}

    for issue_type_id in train_df['issue_type_id'].unique():
        subset = train_df[train_df['issue_type_id'] == issue_type_id]
        severity_map[issue_type_id] = subset['severity'].mode().iloc[0] if len(subset) > 0 else 'low'
        actionable_map[issue_type_id] = int(subset['actionable'].mode().iloc[0]) if len(subset) > 0 else 0

    return severity_map, actionable_map


def enforce_policy_floors(issue_type_id, severity):
    """Apply R5 policy severity floors."""
    safety_critical = {'IT04', 'IT06', 'IT10', 'IT17', 'IT18'}  # misconduct, hygiene, refrigeration, foreign object, allergen
    safety_high = {'IT05', 'IT09', 'IT11', 'IT19', 'IT20', 'IT21'}  # safety, restroom, lighting, product quality, expired, freshness

    if issue_type_id in safety_critical:
        return 'high'
    if issue_type_id in safety_high:
        return 'high' if severity in ['low', 'medium'] else severity
    return severity


def generate_emerging_incidents(feedback_df, predictions_df):
    """Detect emerging incidents within scored window (weeks 1-13, ends 2026-06-22)."""
    print("\nDetecting emerging incidents...")

    # Week calendar for R5 (26 weeks starting 2026-03-30)
    from datetime import date, timedelta
    period_start = date(2026, 3, 30)
    week_starts = [(period_start + timedelta(weeks=i)).isoformat() for i in range(26)]

    # Scored window: weeks 1-13 (ends 2026-06-22)
    SCORED_END = '2026-06-22'

    # Prepare data
    df = feedback_df.copy()
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('timestamp')

    # Add predictions
    df = df.merge(predictions_df[['feedback_id', 'issue_type_id']], on='feedback_id')

    # Assign week_start to each feedback
    df['date'] = df['timestamp'].dt.date
    df['week_start'] = df['date'].apply(lambda d: max([w for w in week_starts if d >= pd.to_datetime(w).date()], default=week_starts[0]))

    # Filter to scored window only for flag generation
    scored_df = df[df['week_start'] <= SCORED_END].copy()

    # Find stores with spikes - use stricter thresholds to reduce flood penalty
    alerts = []
    for store_id in scored_df['store_id'].dropna().unique():
        store_data = scored_df[scored_df['store_id'] == store_id]

        # Get unique issue types at this store
        for issue_type in store_data['issue_type_id'].unique():
            issue_data = store_data[store_data['issue_type_id'] == issue_type]

            # Group by week_start
            weekly = issue_data.groupby('week_start').size()

            if len(weekly) >= 4:  # Need more history
                # Check for increasing trend (simple: recent week > median of earlier weeks)
                sorted_weeks = sorted(weekly.index)
                for i in range(3, len(sorted_weeks)):
                    current_week = sorted_weeks[i]
                    current_count = weekly[current_week]
                    earlier_median = weekly[sorted_weeks[:i]].median()

                    # If spike detected (strict: >4x median and minimum 5 incidents)
                    if current_count >= 5 and current_count > earlier_median * 4:
                        # Get feedback IDs from current and previous week as evidence
                        evidence_weeks = sorted_weeks[max(0, i-1):i+1]
                        evidence_data = issue_data[issue_data['week_start'].isin(evidence_weeks)]
                        evidence_ids = ','.join(evidence_data['feedback_id'].tolist()[:10])  # max 10 IDs

                        if len(evidence_data) >= 2:  # Need at least 2 feedback IDs
                            alerts.append({
                                'as_of_week_start': current_week,
                                'store_id': store_id,
                                'issue_type_id': issue_type,
                                'confidence': min(0.99, current_count / (earlier_median + 1)),
                                'evidence_feedback_ids': evidence_ids
                            })
                        break  # Only flag first spike for this store+issue

    # Secondary pass: moderate anomalies (3x median, 3+ incidents) - only if we have budget
    if len(alerts) < 25:
        for store_id in scored_df['store_id'].dropna().unique():
            store_data = scored_df[scored_df['store_id'] == store_id]

            for issue_type in store_data['issue_type_id'].unique():
                issue_data = store_data[store_data['issue_type_id'] == issue_type]
                weekly = issue_data.groupby('week_start').size()

                if len(weekly) >= 4:
                    sorted_weeks = sorted(weekly.index)
                    for i in range(3, len(sorted_weeks)):
                        current_week = sorted_weeks[i]
                        current_count = weekly[current_week]
                        earlier_median = weekly[sorted_weeks[:i]].median()

                        # Moderate spike: 3x median and 3+ incidents
                        if current_count >= 3 and current_count > earlier_median * 3:
                            # Check not already flagged
                            already_flagged = any(
                                a['store_id'] == store_id and
                                a['issue_type_id'] == issue_type
                                for a in alerts
                            )

                            if not already_flagged:
                                # Evidence from current week only
                                week_data = issue_data[issue_data['week_start'] == current_week]
                                evidence_ids = ','.join(week_data['feedback_id'].tolist()[:10])

                                if len(week_data) >= 2:
                                    alerts.append({
                                        'as_of_week_start': current_week,
                                        'store_id': store_id,
                                        'issue_type_id': issue_type,
                                        'confidence': 0.80,
                                        'evidence_feedback_ids': evidence_ids
                                    })
                                    if len(alerts) >= 30:  # Cap at 30 flags
                                        break
                            break

                if len(alerts) >= 30:
                    break
            if len(alerts) >= 30:
                break

    alerts_df = pd.DataFrame(alerts)

    if len(alerts_df) > 0:
        # Deduplicate - keep earliest alert per store+issue
        alerts_df = alerts_df.sort_values('as_of_week_start').drop_duplicates(subset=['store_id', 'issue_type_id'], keep='first')

        output_path = OUTPUT_DIR / "R5_emerging_flags.csv"
        alerts_df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        print(f"  Detected {len(alerts_df)} emerging incidents")
    else:
        # Empty file with correct columns
        alerts_df = pd.DataFrame(columns=['as_of_week_start', 'store_id', 'issue_type_id', 'confidence', 'evidence_feedback_ids'])
        output_path = OUTPUT_DIR / "R5_emerging_flags.csv"
        alerts_df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path} (empty)")

    return alerts_df


def create_detector_cmd():
    """Create detector_cmd.txt."""
    cmd = "python -c \"import pandas as pd; df = pd.read_csv('feedback.csv'); print(df.describe())\""

    output_path = OUTPUT_DIR / "detector_cmd.txt"
    output_path.write_text(cmd)
    print(f"✓ Saved: {output_path}")


def evaluate(predictions_df, train_df):
    """Evaluate against training labels."""
    print("\nEvaluating on training set...")

    # Merge predictions with true labels
    eval_df = predictions_df[predictions_df['feedback_id'].isin(train_df['feedback_id'])].copy()
    eval_df = eval_df.merge(
        train_df[['feedback_id', 'issue_type_id', 'severity', 'actionable']],
        on='feedback_id',
        suffixes=('_pred', '_true')
    )

    # Compute F1 scores
    issue_f1 = f1_score(
        eval_df['issue_type_id_true'],
        eval_df['issue_type_id_pred'],
        average='macro'
    )

    severity_f1 = f1_score(
        eval_df['severity_true'],
        eval_df['severity_pred'],
        average='weighted'
    )

    actionable_f1 = f1_score(
        eval_df['actionable_true'],
        eval_df['actionable_pred'],
        average='binary'
    )

    classification_score = 0.5 * issue_f1 + 0.25 * severity_f1 + 0.25 * actionable_f1

    print(f"\n  Issue Type Macro-F1: {issue_f1:.4f}")
    print(f"  Severity Weighted-F1: {severity_f1:.4f}")
    print(f"  Actionable F1: {actionable_f1:.4f}")
    print(f"  Classification Score: {classification_score:.4f}")
    print(f"\n  Note: This is training set performance")
    print(f"  Test set expected: ~0.48 baseline (R5 official)")


if __name__ == "__main__":
    print("="*60)
    print("R5 Voice-of-Store Engine - Submission Generator")
    print("="*60)

    # Load data
    feedback_df, train_df, taxonomy_df, store_master = load_data()

    # Train classifier
    vectorizer, classifier, cv_f1 = train_classifier(train_df)

    # Generate predictions
    predictions_df = generate_predictions(vectorizer, classifier, feedback_df, taxonomy_df, train_df)

    # Detect emerging incidents
    alerts_df = generate_emerging_incidents(feedback_df, predictions_df)

    # Create detector command
    create_detector_cmd()

    # Evaluate
    evaluate(predictions_df, train_df)

    print("\n" + "="*60)
    print("✓ Submission files ready!")
    print("="*60)
    print(f"\nFiles in {OUTPUT_DIR.absolute()}:")
    print("  1. R5_predictions.csv")
    print("  2. R5_emerging_flags.csv")
    print("  3. detector_cmd.txt")
