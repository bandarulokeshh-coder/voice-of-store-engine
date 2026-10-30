"""
Load R5 hackathon dataset into TALENTAP format and generate submission files.
"""
import pandas as pd
import sys
from pathlib import Path

# R5 data paths
R5_DATA_DIR = Path(r"C:\Users\lokes\Downloads\R5_Voice_of_Store_Engine_v1.0\R5_Voice_of_Store_Engine\data")
R5_FEEDBACK = R5_DATA_DIR / "feedback.csv"
R5_STORE_MASTER = R5_DATA_DIR / "store_master.csv"
R5_LABELS_TRAIN = R5_DATA_DIR / "feedback_labels_train.csv"
R5_ISSUE_TAXONOMY = R5_DATA_DIR / "issue_taxonomy.csv"
R5_WEEK_CALENDAR = R5_DATA_DIR / "week_calendar.csv"

# Output paths
OUTPUT_DIR = Path("r5_submission")
OUTPUT_DIR.mkdir(exist_ok=True)

def preprocess_r5_for_talentap():
    """Convert R5 dataset to TALENTAP format."""
    print("Loading R5 feedback data...")
    feedback_df = pd.read_csv(R5_FEEDBACK)
    print(f"  Loaded {len(feedback_df):,} feedback records")

    print("Loading store master...")
    store_master = pd.read_csv(R5_STORE_MASTER)

    print("Preprocessing...")
    # Map R5 columns to TALENTAP format
    talentap_df = feedback_df.rename(columns={
        'text': 'feedback_text',
        'created_at': 'timestamp',
        # store_id and channel already match
    })

    # Add region by joining with store_master
    talentap_df = talentap_df.merge(
        store_master[['store_id', 'region']],
        on='store_id',
        how='left'
    )

    # Fill missing regions with "Unknown"
    talentap_df['region'] = talentap_df['region'].fillna('Unknown')

    # Reorder columns to TALENTAP format
    talentap_columns = ['feedback_id', 'feedback_text', 'timestamp', 'store_id', 'region', 'channel', 'rating']
    talentap_df = talentap_df[talentap_columns]

    # Save preprocessed data
    output_path = OUTPUT_DIR / "r5_feedback_talentap.csv"
    talentap_df.to_csv(output_path, index=False)
    print(f"\nPreprocessed data saved to: {output_path}")
    print(f"  Rows: {len(talentap_df):,}")
    print(f"  Columns: {list(talentap_df.columns)}")

    # Show sample
    print("\nSample rows:")
    print(talentap_df.head(3).to_string())

    # Show statistics
    print("\nData statistics:")
    print(f"  Date range: {talentap_df['timestamp'].min()} to {talentap_df['timestamp'].max()}")
    print(f"  Channels: {talentap_df['channel'].value_counts().to_dict()}")
    print(f"  Regions: {talentap_df['region'].value_counts().to_dict()}")
    print(f"  Missing store_id: {talentap_df['store_id'].isna().sum():,}")

    return talentap_df, output_path

if __name__ == "__main__":
    df, path = preprocess_r5_for_talentap()
    print(f"\n✓ Ready to load into TALENTAP")
    print(f"  Upload this file in TALENTAP Settings: {path.absolute()}")
