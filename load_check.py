#!/usr/bin/env python
import pandas as pd
from pathlib import Path

ROOT = Path(r'C:\Users\lokes\TALENTAP') / "data"

# Load feedback
feedback = pd.read_csv(ROOT / "feedback.csv", nrows=100)

# Load labels
labels = pd.read_csv(ROOT / "feedback_labels_train.csv", nrows=100)

# Merge labels with feedback
df = feedback.merge(labels, on='feedback_id', how='left')

# Load store master
stores = pd.read_csv(ROOT / "store_master.csv")

# Merge stores
merge_cols = ['store_id']
df = df.merge(stores[merge_cols], on='store_id', how='left', suffixes=('', '_store'))

# Parse timestamps
df['timestamp'] = pd.to_datetime(df['created_at'])
df['date'] = df['timestamp'].dt.date

# Load week calendar
weeks = pd.read_csv(ROOT / "week_calendar.csv")
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
df['department'] = 'Operations'
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

print("Final dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print()
print("First 5 rows:")
print(df.head())
print()
print("Severity distribution:")
print(df['severity'].value_counts())
print()
print("Issue category distribution:")
print(df['issue_category'].value_counts())
print()
print("Store IDs:", df['store_id'].unique())
print("Region distribution:", df['region'].value_counts() if 'region' in df.columns else "No region column")