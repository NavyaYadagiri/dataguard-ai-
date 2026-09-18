# DataGuard AI — Healthcare Data Quality Agent

An intelligent data quality agent that automatically profiles healthcare datasets, detects anomalies, and generates actionable recommendations using AI.

## What It Does

DataGuard AI runs a 6-step agentic workflow:

1. **Load** — Ingests CSV data and inspects its structure
2. **Profile** — Analyzes every column for types, nulls, uniqueness, and statistics
3. **Quality Checks** — Runs 9 automated rules to catch data problems
4. **Score** — Calculates a composite quality score (0–100)
5. **AI Recommendations** — Sends the profile to an LLM for domain-specific insights
6. **Report** — Generates a formatted report with prioritized findings

## Quality Checks

| # | Check | Why It Matters |
|---|-------|---------------|
| 1 | Missing values | Nulls in critical columns break joins and downstream pipelines |
| 2 | Duplicate rows | Duplicate claims inflate costs and skew reporting |
| 3 | Duplicate primary keys | Violates entity integrity, breaks referential joins |
| 4 | Negative amounts | Claim amounts should be positive unless adjustments |
| 5 | Overpayments | Paid > billed could indicate fraud or processing errors |
| 6 | Future dates | Date of birth in the future is always invalid |
| 7 | Inconsistent casing | Breaks deduplication and master data matching |
| 8 | Invalid gender codes | Non-standard codes cause CMS claim rejections |
| 9 | Statistical outliers | Flags anomalies for human review |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/dataguard-ai.git
cd dataguard-ai

# Install dependencies
pip install -r requirements.txt

# Run with sample data
python data_quality_agent.py

# Run with your own CSV
python data_quality_agent.py your_file.csv
```

## AI-Powered Recommendations (Optional)

To enable AI recommendations, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
python data_quality_agent.py
```

Without the API key, the agent still runs all profiling and quality checks — it uses rule-based recommendations instead.

## Sample Output

```
QUALITY SCORE
----------------------------------------
  [██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 45/100
  Grade: POOR — BLOCK PIPELINE

ISSUES DETECTED
  🔴 [CRITICAL] Negative values in "claim_amount"
  🔴 [CRITICAL] 2 claim(s) where paid exceeds billed amount
  🟡 [WARNING] 1 future date(s) in "date_of_birth"
  🟡 [WARNING] 1 non-standard gender code(s)
  🔵 [INFO] Inconsistent casing in "member_name"
```

## Project Structure

```
dataguard-ai/
├── data_quality_agent.py        # Main agent script
├── sample_healthcare_claims.csv # Sample data with intentional issues
├── requirements.txt             # Python dependencies
└── README.md
```

## Tech Stack

- **Python** — Core language
- **Pandas** — Data profiling and analysis
- **NumPy** — Statistical calculations
- **Anthropic Claude API** — AI-powered recommendations

## Production Considerations

In a production pipeline, this agent would:
- Run as a step in an **Airflow DAG** after each data load
- Use **Great Expectations** or **dbt tests** for rule management
- Publish quality scores to a **metadata catalog** for lineage tracking
- Trigger **PagerDuty/Slack alerts** when score drops below threshold
- Block pipeline progression when critical issues are found

## Author

**Navya** — Data Analytics Professional  
Built as a demonstration of data profiling, quality analysis, and agentic AI for healthcare data engineering.
