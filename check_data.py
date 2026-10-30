#!/usr/bin/env python
import pandas as pd
from pathlib import Path

ROOT = Path(r'C:\Users\lokes\TALENTAP')
DATA_DIR = ROOT / "src" / "data" if (ROOT / "src" / "data").exists() else ROOT / "data"

print("Data directory:", DATA_DIR)
print()

# Check competition data paths
feedback_path = DATA_DIR / "feedback.csv"
labels_path = DATA_DIR / "feedback_labels_train.csv"
store_master_path = DATA_DIR / "store_master.csv"
week_calendar_path = DATA_DIR / "week_calendar.csv"
issue_taxonomy_path = DATA_DIR / "issue_taxonomy.csv"

print("Feedback CSV exists:", feedback_path.exists())
print("Labels CSV exists:", labels_path.exists())
print("Store master exists:", store_master_path.exists())
print("Week calendar exists:", week_calendar_path.exists())
print("Issue taxonomy exists:", issue_taxonomy_path.exists())
print()

# Read sample data
if feedback_path.exists():
    feedback = pd.read_csv(feedback_path, nrows=3)
    print("Feedback columns:", feedback.columns.tolist())
    print("Feedback shape:", feedback.shape)
    print("Feedback head:")
    print(feedback)
    print()

if labels_path.exists():
    labels = pd.read_csv(labels_path, nrows=3)
    print("Labels columns:", labels.columns.tolist())
    print("Labels shape:", labels.shape)
    print("Labels head:")
    print(labels)