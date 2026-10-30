#!/usr/bin/env python
import pandas as pd
from pathlib import Path

ROOT = Path(r'C:\Users\lokes\TALENTAP') / "data"

# Read store master
store_master = pd.read_csv(ROOT / "store_master.csv", nrows=5)
print("Store master columns:", store_master.columns.tolist())
print("Store master shape:", store_master.shape)
print("Store master head:")
print(store_master)
print()

# Read week calendar
week_calendar = pd.read_csv(ROOT / "week_calendar.csv", nrows=5)
print("Week calendar columns:", week_calendar.columns.tolist())
print("Week calendar head:")
print(week_calendar)
print()

# Read issue taxonomy
issue_taxonomy = pd.read_csv(ROOT / "issue_taxonomy.csv", nrows=10)
print("Issue taxonomy columns:", issue_taxonomy.columns.tolist())
print("Issue taxonomy head:")
print(issue_taxonomy.head(10))