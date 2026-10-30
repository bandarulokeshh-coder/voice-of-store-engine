"""Load and prepare competition dataset for dashboard"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

FEEDBACK_PATH = DATA_DIR / "feedback.csv"
LABELS_PATH = DATA_DIR / "feedback_labels_train.csv"
STORE_MASTER_PATH = DATA_DIR / "store_master.csv"
WEEK_CALENDAR_PATH = DATA_DIR / "week_calendar.csv"
ISSUE_TAXONOMY_PATH = DATA_DIR / "issue_taxonomy.csv"


def load_competition_dataset() -> pd.DataFrame:
    """Load competition feedback with labels and enrich for dashboard"""

    # Load feedback
    feedback = pd.read_csv(FEEDBACK_PATH)

    # Load labels (only ~3000 labeled samples)
    labels = pd.read_csv(LABELS_PATH)

    # Merge labels with feedback
    df = feedback.merge(labels, on='feedback_id', how='left')

    # Load store master for region info
    stores = pd.read_csv(STORE_MASTER_PATH)
    # Select only available columns
    store_cols = [col for col in ['store_id', 'region', 'size_sqft'] if col in stores.columns]
    df = df.merge(stores[store_cols], on='store_id', how='left')
    # Rename size_sqft to store_size if it exists
    if 'size_sqft' in df.columns:
        df.rename(columns={'size_sqft': 'store_size'}, inplace=True)

    # Load week calendar
    weeks = pd.read_csv(WEEK_CALENDAR_PATH)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['date'] = df['timestamp'].dt.date

    # Map to week_start using week_calendar
    weeks['week_start'] = pd.to_datetime(weeks['week_start'])
    weeks['week_end'] = pd.to_datetime(weeks['week_end'])

    def find_week_start(date):
        if pd.isna(date):
            return None
        for _, row in weeks.iterrows():
            if row['week_start'].date() <= date <= row['week_end'].date():
                return row['week_start']
        return None

    df['week_start'] = df['date'].apply(find_week_start)

    # Rename columns to match dashboard expectations
    df = df.rename(columns={
        'text': 'feedback_text',
        'issue_type_id': 'issue_category',
    })

    # Add required columns for dashboard
    df['department'] = 'Operations'  # Default, can be enriched
    df['dataset_mode'] = 'train'
    df['incident_tag'] = None
    df['known_incident_flag'] = False
    df['language_hint'] = 'multilingual'
    df['feedback_text_normalized'] = df['feedback_text'].fillna('')

    # Convert actionable to boolean
    df['actionable'] = df['actionable'].fillna(0).astype(bool)

    # Store-week ID
    df['store_week_id'] = df['store_id'].astype(str) + '_' + df['week_start'].astype(str)
    df['week_label'] = df['week_start'].dt.strftime('%Y-W%U')

    return df


def load_competition_for_r5_generation() -> pd.DataFrame:
    """Load competition data formatted for R5 submission generation"""

    # Load base feedback
    feedback = pd.read_csv(FEEDBACK_PATH)
    labels = pd.read_csv(LABELS_PATH)

    # Merge
    df = feedback.merge(labels, on='feedback_id', how='left')

    # Load stores
    stores = pd.read_csv(STORE_MASTER_PATH)
    # Select only available columns
    store_cols = [col for col in ['store_id', 'region', 'size_sqft'] if col in stores.columns]
    df = df.merge(stores[store_cols], on='store_id', how='left')
    # Rename size_sqft to store_size if it exists
    if 'size_sqft' in df.columns:
        df.rename(columns={'size_sqft': 'store_size'}, inplace=True)

    # Load weeks
    weeks = pd.read_csv(WEEK_CALENDAR_PATH)
    weeks['week_start'] = pd.to_datetime(weeks['week_start'])
    weeks['week_end'] = pd.to_datetime(weeks['week_end'])

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['date'] = df['timestamp'].dt.date

    # Map to week_start
    def find_week_start(date):
        if pd.isna(date):
            return None
        for _, row in weeks.iterrows():
            if row['week_start'].date() <= date <= row['week_end'].date():
                return row['week_start']
        return None

    df['week_start'] = df['date'].apply(find_week_start)

    # Keep only labeled samples with valid week_start
    df = df[df['issue_type_id'].notna() & df['week_start'].notna()].copy()

    return df
