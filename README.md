# StorePulse AI
**Turning Store Feedback Into Intelligent Action**

StorePulse AI is a demo-ready retail intelligence platform built for the **TalentAP Mega Hackathon 2026** problem statement:

> **R5 — Turning Store Chatter Into Action: The Voice-of-Store Engine**

The app processes multilingual retail feedback, detects operational issues, assigns department owners, flags policy-driven safety escalations, surfaces emerging incidents, and generates weekly regional action briefs.

---

## Project Overview
Retail organizations receive large volumes of unstructured feedback from customer reviews, in-app comments, staff notes, and call-centre transcripts. StorePulse AI turns this feedback into operational action by:

- classifying issues from multilingual text
- applying policy-first safety escalation rules
- routing issues to the correct department
- detecting rising store-level incident patterns early
- generating weekly regional reports
- highlighting evidence-backed explanations and duplicate reports

The platform is designed to run locally and does **not require any paid API**.

---

## Key Features

### 1) Executive Dashboard
- Total feedback analyzed
- High-severity issues
- Emerging incidents
- Stores monitored
- Actionable feedback percentage
- Departments requiring attention
- Weekly top issues
- Feedback trend charts
- Issue distribution charts
- Department workload charts
- Filters for region, store, date range, channel, severity, and category

### 2) Multilingual AI Feedback Analyzer
Supports:
- English
- Hinglish
- Common transliterated Telugu
- Transliterated Tamil
- Transliterated Marathi
- Common spelling mistakes and retail shorthand

The analyzer uses:
- a transparent rule-based baseline
- optional lightweight ML training when labeled data is available
- policy-first safety escalation for allergen, hygiene, contamination, misconduct, and unsafe-product issues

### 3) Policy-First Safety Escalation
Any feedback mentioning safety-related concerns such as:
- allergens
- food contamination
- hygiene
- staff misconduct
- harassment
- unsafe products

is escalated to **HIGH** severity.

### 4) Emerging Incident Detection
A store-level early-warning system that monitors:
- complaint counts
- normalized complaint rates
- historical baselines
- growth rates
- robust median/MAD spikes

It can detect incidents early even in small stores with low complaint volume.

### 5) Weekly Regional Action Brief
For each region the app can generate:
- regional overview
- top 5 priority issues
- action suggestions
- evidence quotes
- downloadable PDF and CSV reports

### 6) Automatic Department Routing
Operational issues are routed to:
- Store Operations
- Supply Chain
- Merchandising
- Digital
- HR

### 7) Evidence-Based Explainability
Every key alert explains:
- why it triggered
- which rule or pattern matched
- which department owns the issue
- supporting feedback quotes from the actual dataset

### 8) Cross-Channel Duplicate Detection
The app groups probable duplicate incidents reported across multiple channels while keeping raw records intact.

### 9) Evaluation & Model Insights
Shows demonstration metrics for:
- issue type
- severity
- actionability
- emerging detection lead-time

---

## Technology Stack

### Frontend / App Framework
- Streamlit

### Data Processing
- Pandas
- NumPy

### NLP / ML
- Sentence-Transformers (optional)
- Scikit-learn
- Rule-based multilingual fallback

### Visualization
- Plotly

### Database
- SQLite

### Reporting
- ReportLab for PDF exports
- Pandas CSV exports

---

## Architecture

```text
storepulse-ai/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_feedback.csv
├── src/
│   ├── preprocessing.py
│   ├── classifier.py
│   ├── policy_engine.py
│   ├── routing.py
│   ├── emerging_detector.py
│   ├── weekly_report.py
│   ├── duplicate_detector.py
│   ├── evaluation.py
│   ├── synthetic_data.py
│   └── storage.py
├── config/
│   └── issue_taxonomy.json
├── assets/
│   └── logo.png
└── tests/
    ├── test_policy_engine.py
    ├── test_emerging_detector.py
    └── test_routing.py
```

---

## Installation Instructions

### 1) Create and activate a virtual environment

#### Windows PowerShell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Start the Streamlit app
```bash
streamlit run app.py
```

### Optional: run tests
```bash
pytest -q
```

---

## Dataset Format

### Synthetic Demonstration Data
The app generates a labeled synthetic dataset with at least 2,000 records and marks it as:

**Synthetic Demonstration Data**

### Required fields
- `feedback_id`
- `store_id`
- `region`
- `timestamp`
- `channel`
- `feedback_text`
- `issue_category`
- `severity`
- `actionable`
- `department`

### Additional fields
- `dataset_mode`
- `incident_tag`
- `known_incident_flag`
- `language_hint`
- `week_start`

### Official Dataset Upload
You can upload a CSV and map columns for:
- feedback text
- date
- store ID
- region
- channel

Optional uploads:
- store master
- taxonomy
- labeled training data

Missing values and invalid dates are handled safely.

---

## How Classification Works

StorePulse AI uses a transparent layered approach:

1. **Normalization**
   - lowercasing
   - punctuation cleanup
   - transliteration-friendly replacements
   - multilingual token normalization

2. **Rule-based classification**
   - keyword and policy pattern matches
   - multilingual retail terms
   - safety escalation rules

3. **Optional supervised ML**
   - if labeled data is available, the app can train a lightweight TF-IDF + Logistic Regression model
   - if ML is unavailable, the app falls back to the rule engine without crashing

4. **Policy-first severity logic**
   - allergen, hygiene, contamination, harassment, staff misconduct, and unsafe products are forced to HIGH severity

The app does **not** invent confidence percentages for pure rule-based decisions.

---

## How Emerging Detection Works

The early-warning system calculates, for each store and issue category:

- weekly issue count
- normalized complaint rate
- historical median baseline
- median absolute deviation (MAD)
- safe growth ratios

### Default alert conditions
- `minimum_support = 2`
- `growth_threshold = 2.5`
- `spike_threshold = 3.0`

### Core formula
```text
normalized_rate = issue_count / max(total_store_feedback, 1)
```

The detector compares the current week against the previous 4–6 weeks using a robust baseline. It handles zero-history stores safely and can still detect small but meaningful spikes.

---

## Limitations
- The default demo dataset is synthetic and clearly labeled as such.
- Rule-based classification is transparent but less powerful than a fully trained multilingual model.
- Optional sentence-transformers support depends on local installation and model availability.
- Some evaluation results are demo-only unless real ground-truth labels are uploaded.
- Duplicate detection is heuristic and intended for operations review, not automatic merging.

---

## Testing Instructions

Run the automated tests:

```bash
pytest -q
```

Included scenarios:
- Safety escalation for allergen issues
- Department routing for payment failures
- Emerging refrigeration spike detection
- Low-volume store trend detection
- Missing data handling
- Unknown feedback should not fabricate a critical issue

---

## Team Contribution
Suggested hackathon roles:
- **Full-stack developer:** Streamlit app, UI, uploads, exports
- **ML/NLP engineer:** multilingual normalization, rule engine, optional ML training
- **Data scientist:** incident detection, evaluation metrics, report scoring
- **UX designer:** dashboard layout, cards, filters, report styling

---

## Windows + VS Code Quick Start

1. Open the project folder in **VS Code**.
2. Open **Terminal** in VS Code.
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
4. Activate it:
   ```bash
   .venv\Scripts\activate
   ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Launch the app:
   ```bash
   streamlit run app.py
   ```
7. Open the local URL Streamlit prints in the terminal.

If Streamlit does not open automatically, copy the local URL into Chrome or Edge.

---

## Notes for Judges
- Demo Mode is available from the dashboard.
- All demo records are synthetic and labeled accordingly.
- The project is designed to work locally without paid APIs.
- Weekly reports and emerging alerts use actual processed records from the loaded dataset.
