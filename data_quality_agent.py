"""
=============================================================================
DataGuard AI — Healthcare Data Quality Agent
=============================================================================

WHAT THIS IS:
    An intelligent data quality agent that automatically profiles a healthcare
    dataset, detects quality issues, and uses an LLM to generate actionable
    recommendations. Built to demonstrate data profiling, data quality analysis,
    and agentic AI capabilities.

HOW TO RUN:
    pip install pandas anthropic
    python data_quality_agent.py

WHAT IT DOES (the "agentic" workflow):
    Step 1: Load and inspect the dataset
    Step 2: Profile every column (types, nulls, uniqueness, statistics)
    Step 3: Run automated quality checks (9 different rules)
    Step 4: Calculate a composite quality score
    Step 5: Send the profile to an LLM for intelligent recommendations
    Step 6: Generate a formatted report

AUTHOR: Navya
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys

# Optional: for AI-powered recommendations
try:
    import anthropic
    HAS_AI = True
except ImportError:
    HAS_AI = False
    print("Note: 'anthropic' package not installed. Running without AI recommendations.")
    print("Install with: pip install anthropic\n")


# =============================================================================
# STEP 1: LOAD THE DATASET
# =============================================================================
# WHY THIS MATTERS:
#   Before any analysis, we need to read the data and understand its basic
#   shape. In production, this would connect to a database, S3 bucket, or
#   receive data from an upstream pipeline.
# =============================================================================

def load_dataset(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    In a real pipeline, this function would:
    - Connect to multiple data sources (databases, APIs, cloud storage)
    - Handle different file formats (CSV, Parquet, JSON)
    - Log the ingestion event for data lineage tracking
    """
    print(f"Loading dataset: {file_path}")
    print("-" * 60)
    
    df = pd.read_csv(file_path)
    
    print(f"  Rows loaded:    {len(df):,}")
    print(f"  Columns found:  {len(df.columns)}")
    print(f"  Column names:   {', '.join(df.columns)}")
    print(f"  Memory usage:   {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print()
    
    return df


# =============================================================================
# STEP 2: PROFILE EVERY COLUMN
# =============================================================================
# WHY THIS MATTERS:
#   Data profiling is the foundation of data quality. Before you can find
#   problems, you need to understand what you're working with — what types
#   of data each column holds, how complete it is, how many unique values
#   it has, and what the distribution looks like.
#
#   This maps directly to the JD requirement:
#   "Perform data profiling analysis and data quality analysis"
# =============================================================================

def profile_columns(df):
    """
    Generate a detailed profile for every column in the dataset.
    
    For each column, we calculate:
    - Data type (inferred from actual values, not just pandas dtype)
    - Null count and percentage
    - Unique value count and percentage
    - Most frequent value
    - For numeric columns: min, max, mean, median, std dev
    - For date columns: min date, max date, date range
    """
    print("Profiling columns...")
    print("-" * 60)
    
    profiles = {}
    
    for col in df.columns:
        profile = {}
        
        # --- Basic counts ---
        total = len(df)
        null_count = df[col].isna().sum()
        non_null = df[col].dropna()
        
        profile['total_rows'] = total
        profile['null_count'] = int(null_count)
        profile['null_pct'] = round((null_count / total) * 100, 1)
        profile['non_null_count'] = int(total - null_count)
        
        # --- Uniqueness ---
        # High uniqueness in an ID column = good (means no duplicates)
        # High uniqueness in a category column = suspicious (too many categories)
        unique_count = non_null.nunique()
        profile['unique_count'] = int(unique_count)
        profile['unique_pct'] = round((unique_count / total) * 100, 1) if total > 0 else 0
        
        # --- Most frequent value ---
        # Helps spot data skew and default values
        if len(non_null) > 0:
            top_value = non_null.value_counts().head(1)
            profile['top_value'] = str(top_value.index[0])
            profile['top_value_count'] = int(top_value.values[0])
        else:
            profile['top_value'] = 'N/A'
            profile['top_value_count'] = 0
        
        # --- Infer the actual data type ---
        # pandas might say "object" but we want to know if it's really
        # a date, a number stored as text, or a true string
        profile['pandas_dtype'] = str(df[col].dtype)
        profile['inferred_type'] = infer_column_type(non_null)
        
        # --- Numeric statistics ---
        # Only calculated for numeric columns
        if profile['inferred_type'] == 'numeric':
            nums = pd.to_numeric(non_null, errors='coerce').dropna()
            if len(nums) > 0:
                profile['min'] = float(nums.min())
                profile['max'] = float(nums.max())
                profile['mean'] = round(float(nums.mean()), 2)
                profile['median'] = round(float(nums.median()), 2)
                profile['std_dev'] = round(float(nums.std()), 2)
        
        # --- Date statistics ---
        if profile['inferred_type'] == 'date':
            dates = pd.to_datetime(non_null, errors='coerce').dropna()
            if len(dates) > 0:
                profile['min_date'] = str(dates.min().date())
                profile['max_date'] = str(dates.max().date())
        
        profiles[col] = profile
        
        # Print a summary line for each column
        null_indicator = f"  ⚠ {profile['null_pct']}% null" if profile['null_pct'] > 0 else ""
        print(f"  {col:<25} {profile['inferred_type']:<10} "
              f"unique: {profile['unique_count']:<6}{null_indicator}")
    
    print()
    return profiles


def infer_column_type(series):
    """
    Infer the actual semantic type of a column by examining its values.
    
    WHY NOT JUST USE df.dtypes?
    Because pandas often reads everything as 'object' (string) when the
    CSV has mixed types or nulls. A column of dates like '2026-01-15'
    will show up as 'object' in pandas, but we need to know it's a date
    so we can run date-specific quality checks.
    """
    if len(series) == 0:
        return 'empty'
    
    sample = series.head(20).astype(str)
    
    # Check if values look like numbers
    numeric_count = sum(1 for v in sample if is_numeric(v))
    if numeric_count > len(sample) * 0.8:
        return 'numeric'
    
    # Check if values look like dates
    date_count = sum(1 for v in sample if is_date(v))
    if date_count > len(sample) * 0.8:
        return 'date'
    
    return 'string'


def is_numeric(value):
    """Check if a string value is numeric."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_date(value):
    """Check if a string value looks like a date."""
    try:
        if '-' in str(value) and len(str(value)) >= 8:
            datetime.strptime(str(value)[:10], '%Y-%m-%d')
            return True
    except (ValueError, TypeError):
        pass
    return False


# =============================================================================
# STEP 3: RUN AUTOMATED QUALITY CHECKS
# =============================================================================
# WHY THIS MATTERS:
#   This is the core of the agent — applying business rules to catch data
#   problems automatically. Each check is a rule that would run inside a
#   data pipeline (e.g., in dbt tests, Great Expectations, or custom
#   validation scripts).
#
#   In healthcare data, bad data isn't just annoying — it can lead to:
#   - Wrong claim payments (financial impact)
#   - Incorrect member records (compliance/HIPAA risk)
#   - Bad analytics that drive wrong business decisions
# =============================================================================

def run_quality_checks(df, profiles):
    """
    Run 9 automated quality checks against the dataset.
    
    Each check returns an issue dict with:
    - severity: 'critical', 'warning', or 'info'
    - title: short description of the problem
    - description: detailed explanation with business context
    - column: which column is affected
    - count: how many rows are affected
    """
    print("Running quality checks...")
    print("-" * 60)
    
    issues = []
    
    # ---------------------------------------------------------------
    # CHECK 1: Missing Values (Null Detection)
    # ---------------------------------------------------------------
    # WHY: Nulls in critical columns can break joins, cause incorrect
    #       calculations, and violate data contracts with downstream systems.
    #       In healthcare: a null member_id means we can't link a claim
    #       to a patient — that's a serious integrity issue.
    # ---------------------------------------------------------------
    for col in df.columns:
        null_pct = profiles[col]['null_pct']
        null_count = profiles[col]['null_count']
        
        if null_pct > 10:
            issues.append({
                'severity': 'critical',
                'title': f'High null rate in "{col}"',
                'description': (f'{null_pct}% of values are missing ({null_count} of '
                              f'{len(df)} rows). This could cause failures in '
                              f'downstream pipelines and inaccurate reporting.'),
                'column': col,
                'count': null_count
            })
        elif null_pct > 0:
            issues.append({
                'severity': 'warning',
                'title': f'Missing values in "{col}"',
                'description': (f'{null_pct}% null ({null_count} rows). Review whether '
                              f'nulls are expected or indicate a data collection gap.'),
                'column': col,
                'count': null_count
            })
    
    # ---------------------------------------------------------------
    # CHECK 2: Exact Duplicate Rows
    # ---------------------------------------------------------------
    # WHY: Duplicate rows in a claims table mean the same claim was
    #       recorded twice. This inflates total cost metrics and produces
    #       wrong reports. In production, duplicates often come from:
    #       - ETL jobs running twice without idempotency
    #       - Source systems sending the same record in multiple batches
    # ---------------------------------------------------------------
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append({
            'severity': 'critical',
            'title': f'{dup_count} exact duplicate row(s) detected',
            'description': ('Exact duplicate rows found. In a claims dataset, this '
                          'could indicate duplicate claim submissions that inflate '
                          'costs and skew reporting. Check ETL idempotency.'),
            'column': 'ALL',
            'count': int(dup_count)
        })
    
    # ---------------------------------------------------------------
    # CHECK 3: Duplicate Primary Keys
    # ---------------------------------------------------------------
    # WHY: If claim_id has duplicates, the table violates entity integrity.
    #       Every claim should have a unique identifier. Duplicate IDs make
    #       it impossible to reliably join this table to other tables.
    # ---------------------------------------------------------------
    id_columns = [c for c in df.columns if 'claim_id' in c.lower() or c.lower() == 'id']
    for col in id_columns:
        total = df[col].dropna().shape[0]
        unique = df[col].dropna().nunique()
        dup_ids = total - unique
        if dup_ids > 0:
            issues.append({
                'severity': 'critical',
                'title': f'Duplicate values in primary key "{col}"',
                'description': (f'{dup_ids} duplicate ID(s) found. Primary key columns '
                              f'must be unique to maintain referential integrity. '
                              f'This breaks joins to downstream tables.'),
                'column': col,
                'count': dup_ids
            })
    
    # ---------------------------------------------------------------
    # CHECK 4: Negative Amounts
    # ---------------------------------------------------------------
    # WHY: Claim amounts should be positive unless they represent
    #       adjustments or reversals. An unexplained negative amount
    #       is almost always a data entry error or ETL transformation bug.
    # ---------------------------------------------------------------
    amount_cols = [c for c in df.columns 
                   if any(word in c.lower() for word in ['amount', 'cost', 'price', 'paid'])]
    for col in amount_cols:
        nums = pd.to_numeric(df[col], errors='coerce')
        neg_count = (nums < 0).sum()
        if neg_count > 0:
            issues.append({
                'severity': 'critical',
                'title': f'Negative values in "{col}"',
                'description': (f'{neg_count} negative value(s) detected. Claim amounts '
                              f'should not be negative unless representing adjustments '
                              f'or reversals — verify with the source system.'),
                'column': col,
                'count': int(neg_count)
            })
    
    # ---------------------------------------------------------------
    # CHECK 5: Paid Amount Exceeds Claim Amount
    # ---------------------------------------------------------------
    # WHY: A health plan should never pay MORE than what was billed.
    #       If paid_amount > claim_amount, this could indicate:
    #       - A processing error in the claims adjudication system
    #       - Duplicate payment
    #       - Potential fraud that needs investigation
    # ---------------------------------------------------------------
    if 'claim_amount' in df.columns and 'paid_amount' in df.columns:
        claim_amt = pd.to_numeric(df['claim_amount'], errors='coerce')
        paid_amt = pd.to_numeric(df['paid_amount'], errors='coerce')
        overpaid = ((paid_amt > claim_amt) & claim_amt.notna() & paid_amt.notna()).sum()
        if overpaid > 0:
            issues.append({
                'severity': 'critical',
                'title': f'{overpaid} claim(s) where paid exceeds billed amount',
                'description': ('Paid amount should never exceed the billed claim '
                              'amount. This could indicate processing errors, '
                              'duplicate payments, or fraudulent claims requiring '
                              'investigation by the SIU (Special Investigations Unit).'),
                'column': 'paid_amount',
                'count': int(overpaid)
            })
    
    # ---------------------------------------------------------------
    # CHECK 6: Future Dates
    # ---------------------------------------------------------------
    # WHY: A date of birth in the future is always invalid. A claim date
    #       in the future might be a pre-authorization, but it's worth
    #       flagging because it's usually a data entry error.
    # ---------------------------------------------------------------
    date_cols = [c for c in df.columns 
                 if any(word in c.lower() for word in ['date', 'dob', 'birth'])]
    today = pd.Timestamp.now()
    for col in date_cols:
        dates = pd.to_datetime(df[col], errors='coerce')
        future_count = (dates > today).sum()
        if future_count > 0:
            issues.append({
                'severity': 'warning',
                'title': f'{future_count} future date(s) in "{col}"',
                'description': (f'Dates in the future detected. For date_of_birth, '
                              f'this is always invalid. For claim_date, verify whether '
                              f'these are pre-authorized claims or data entry errors.'),
                'column': col,
                'count': int(future_count)
            })
    
    # ---------------------------------------------------------------
    # CHECK 7: Inconsistent Casing
    # ---------------------------------------------------------------
    # WHY: If "John Smith" and "john smith" both appear, they look like
    #       different people but they're the same. Inconsistent casing
    #       breaks deduplication, matching, and master data management.
    #       This is especially critical for member names and provider names
    #       in healthcare data.
    # ---------------------------------------------------------------
    name_cols = [c for c in df.columns if 'name' in c.lower()]
    for col in name_cols:
        values = df[col].dropna().astype(str)
        if len(values) > 0:
            has_title = values.apply(lambda x: x[0].isupper() if len(x) > 0 else False).any()
            has_lower = values.apply(lambda x: x == x.lower() and len(x) > 1).any()
            if has_title and has_lower:
                lower_examples = values[values.apply(lambda x: x == x.lower() and len(x) > 1)]
                example = lower_examples.iloc[0] if len(lower_examples) > 0 else ''
                issues.append({
                    'severity': 'info',
                    'title': f'Inconsistent casing in "{col}"',
                    'description': (f'Mixed capitalization detected (e.g., "{example}"). '
                                  f'Standardize to title case for consistent matching, '
                                  f'deduplication, and master data management.'),
                    'column': col,
                    'count': int(has_lower)
                })
    
    # ---------------------------------------------------------------
    # CHECK 8: Invalid Gender Codes
    # ---------------------------------------------------------------
    # WHY: Healthcare data uses standardized gender codes (M/F/U) for
    #       claims processing, regulatory reporting, and clinical analytics.
    #       Non-standard codes cause rejections when submitting claims
    #       to CMS or state Medicaid agencies.
    # ---------------------------------------------------------------
    if 'gender' in df.columns:
        valid_genders = {'M', 'F', 'U', 'Male', 'Female', 'Unknown'}
        gender_values = df['gender'].dropna().astype(str).str.strip()
        invalid = gender_values[~gender_values.str.upper().isin({v.upper() for v in valid_genders})]
        if len(invalid) > 0:
            issues.append({
                'severity': 'warning',
                'title': f'{len(invalid)} non-standard gender code(s)',
                'description': (f'Values outside the expected set (M/F/U) found: '
                              f'"{invalid.iloc[0]}". Non-standard codes should be '
                              f'mapped to standard values or flagged for review.'),
                'column': 'gender',
                'count': len(invalid)
            })
    
    # ---------------------------------------------------------------
    # CHECK 9: Statistical Outliers
    # ---------------------------------------------------------------
    # WHY: A claim for $22,000 when the average is $1,500 could be:
    #       - A legitimate high-cost claim (cancer treatment, surgery)
    #       - A data entry error (extra zero)
    #       - Fraud (inflated billing)
    #       We flag outliers so analysts can investigate — the agent
    #       doesn't decide, it surfaces the anomaly for human review.
    # ---------------------------------------------------------------
    for col in amount_cols:
        nums = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(nums) > 5:
            mean = nums.mean()
            std = nums.std()
            if std > 0:
                outliers = nums[abs(nums - mean) > 3 * std]
                if len(outliers) > 0:
                    issues.append({
                        'severity': 'info',
                        'title': f'{len(outliers)} statistical outlier(s) in "{col}"',
                        'description': (f'Values more than 3 standard deviations from '
                                      f'the mean (${mean:,.2f} ± ${std:,.2f}). Could be '
                                      f'legitimate high-cost claims or data entry errors. '
                                      f'Outlier values: {", ".join(f"${v:,.2f}" for v in outliers)}'),
                        'column': col,
                        'count': len(outliers)
                    })
    
    # Print summary
    severity_counts = {}
    for issue in issues:
        sev = issue['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    print(f"  Critical:  {severity_counts.get('critical', 0)}")
    print(f"  Warnings:  {severity_counts.get('warning', 0)}")
    print(f"  Info:      {severity_counts.get('info', 0)}")
    print()
    
    return issues


# =============================================================================
# STEP 4: CALCULATE COMPOSITE QUALITY SCORE
# =============================================================================
# WHY THIS MATTERS:
#   A single score makes it easy for stakeholders (and pipeline alerts)
#   to quickly understand overall data health. In production, you'd set
#   thresholds: score < 70 = block the pipeline, 70-90 = alert the team,
#   90+ = proceed normally.
# =============================================================================

def calculate_quality_score(issues):
    """
    Calculate a composite data quality score from 0 to 100.
    
    Scoring logic:
    - Start at 100 (perfect)
    - Deduct 12 points per critical issue
    - Deduct 5 points per warning
    - Deduct 2 points per info-level issue
    """
    score = 100
    
    for issue in issues:
        if issue['severity'] == 'critical':
            score -= 12
        elif issue['severity'] == 'warning':
            score -= 5
        elif issue['severity'] == 'info':
            score -= 2
    
    score = max(0, min(100, score))
    
    # Determine grade
    if score >= 90:
        grade = 'EXCELLENT'
    elif score >= 75:
        grade = 'GOOD'
    elif score >= 60:
        grade = 'NEEDS ATTENTION'
    else:
        grade = 'POOR — BLOCK PIPELINE'
    
    print(f"Data Quality Score: {score}/100 ({grade})")
    print("-" * 60)
    print()
    
    return score, grade


# =============================================================================
# STEP 5: AI-POWERED RECOMMENDATIONS (THE "AGENTIC" PART)
# =============================================================================
# WHY THIS MATTERS:
#   This is what makes this an "agent" rather than just a script. A
#   traditional data quality tool runs rules and reports results.
#   An agent takes those results, understands the CONTEXT (healthcare
#   claims data), and generates intelligent, actionable recommendations
#   that a human would need domain expertise to produce.
#
#   The agent doesn't just say "you have nulls" — it says "nulls in
#   date_of_birth could cause HEDIS quality measure calculations to
#   fail, which affects your star ratings."
# =============================================================================

def get_ai_recommendations(df, profiles, issues, score):
    """
    Send the profiling results to an LLM for intelligent analysis.
    
    The LLM acts as a "senior data quality engineer" — it receives
    the raw profiling data and issues, applies healthcare domain
    knowledge, and returns prioritized, actionable recommendations.
    """
    if not HAS_AI:
        return generate_fallback_recommendations(issues, score)
    
    print("AI Agent generating recommendations...")
    print("-" * 60)
    
    # Build a concise summary for the LLM
    # (We don't send raw data — just the profile and issues)
    profile_summary = "\n".join([
        f"  {col}: type={p['inferred_type']}, nulls={p['null_pct']}%, "
        f"unique={p['unique_count']}, top='{p['top_value']}'"
        for col, p in profiles.items()
    ])
    
    issues_summary = "\n".join([
        f"  [{i['severity'].upper()}] {i['title']}: {i['description']}"
        for i in issues
    ])
    
    prompt = f"""You are a senior data quality engineer at a healthcare payer.
Analyze this profiling report and provide actionable recommendations.

DATASET: {len(df)} rows, {len(df.columns)} columns
QUALITY SCORE: {score}/100

COLUMN PROFILES:
{profile_summary}

ISSUES DETECTED:
{issues_summary}

Provide your analysis in this format:

SUMMARY
(2-3 sentence overview)

TOP PRIORITY FIXES
(Numbered list of the 3 most critical issues to fix first, with specific actions)

HEALTHCARE-SPECIFIC CONCERNS
(HIPAA, claims processing, or regulatory concerns)

PIPELINE RECOMMENDATIONS
(Specific validation rules, monitoring, or checks to add to prevent these issues)"""

    try:
        client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        recommendation = message.content[0].text
        print("  AI recommendations generated successfully.\n")
        return recommendation
    
    except Exception as e:
        print(f"  AI unavailable ({e}). Using rule-based recommendations.\n")
        return generate_fallback_recommendations(issues, score)


def generate_fallback_recommendations(issues, score):
    """
    Generate rule-based recommendations when the AI is not available.
    
    This ensures the agent ALWAYS produces useful output, even without
    an API key. In production, you'd want both: automated rules for
    speed, plus AI analysis for nuanced insights.
    """
    lines = []
    lines.append("SUMMARY")
    lines.append(f"Dataset quality score is {score}/100. "
                 f"Found {len(issues)} issue(s) requiring attention.\n")
    
    lines.append("TOP PRIORITY FIXES")
    critical = [i for i in issues if i['severity'] == 'critical']
    for idx, issue in enumerate(critical[:3], 1):
        lines.append(f"  {idx}. {issue['title']}")
        lines.append(f"     Action: {issue['description']}\n")
    
    lines.append("PIPELINE RECOMMENDATIONS")
    lines.append("  - Add null checks on critical columns before loading to warehouse")
    lines.append("  - Implement deduplication logic with idempotency keys")
    lines.append("  - Add range validation for amount columns (must be >= 0)")
    lines.append("  - Add referential integrity checks on ID columns")
    lines.append("  - Set up automated alerts when quality score drops below 70")
    
    return "\n".join(lines)


# =============================================================================
# STEP 6: GENERATE THE FINAL REPORT
# =============================================================================
# WHY THIS MATTERS:
#   The output needs to be consumable by both technical and non-technical
#   stakeholders. A data engineer wants the column-level details. A manager
#   wants the score and top issues. The AI recommendations bridge both.
# =============================================================================

def generate_report(df, profiles, issues, score, grade, recommendations):
    """Generate a formatted quality report."""
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  DATAGUARD AI — DATA QUALITY REPORT")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # --- Dataset Overview ---
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"  Rows:           {len(df):,}")
    report_lines.append(f"  Columns:        {len(df.columns)}")
    total_cells = len(df) * len(df.columns)
    total_nulls = sum(p['null_count'] for p in profiles.values())
    completeness = ((1 - total_nulls / total_cells) * 100) if total_cells > 0 else 0
    report_lines.append(f"  Total cells:    {total_cells:,}")
    report_lines.append(f"  Missing cells:  {total_nulls:,}")
    report_lines.append(f"  Completeness:   {completeness:.1f}%")
    report_lines.append("")
    
    # --- Quality Score ---
    report_lines.append("QUALITY SCORE")
    report_lines.append("-" * 40)
    bar_filled = int(score / 2)
    bar_empty = 50 - bar_filled
    bar = "█" * bar_filled + "░" * bar_empty
    report_lines.append(f"  [{bar}] {score}/100")
    report_lines.append(f"  Grade: {grade}")
    report_lines.append("")
    
    # --- Issues ---
    report_lines.append("ISSUES DETECTED")
    report_lines.append("-" * 40)
    for issue in issues:
        icon = {'critical': '🔴', 'warning': '🟡', 'info': '🔵'}.get(issue['severity'], '⚪')
        report_lines.append(f"  {icon} [{issue['severity'].upper()}] {issue['title']}")
        report_lines.append(f"     {issue['description']}")
        report_lines.append("")
    
    # --- Column Details ---
    report_lines.append("COLUMN PROFILES")
    report_lines.append("-" * 40)
    header = f"  {'Column':<25} {'Type':<10} {'Nulls':<10} {'Unique':<10} {'Top Value'}"
    report_lines.append(header)
    report_lines.append("  " + "-" * 80)
    for col, p in profiles.items():
        null_str = f"{p['null_pct']}%" if p['null_pct'] > 0 else "0%"
        report_lines.append(
            f"  {col:<25} {p['inferred_type']:<10} {null_str:<10} "
            f"{p['unique_count']:<10} {p['top_value'][:30]}"
        )
    report_lines.append("")
    
    # --- AI Recommendations ---
    report_lines.append("AI-POWERED RECOMMENDATIONS")
    report_lines.append("-" * 40)
    report_lines.append(recommendations)
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("  End of Report")
    report_lines.append("=" * 70)
    
    report = "\n".join(report_lines)
    return report


# =============================================================================
# MAIN: THE AGENT WORKFLOW
# =============================================================================
# This is the orchestrator that runs all steps in sequence.
# In production, each step would be a task in an Airflow DAG or
# a stage in a CI/CD pipeline.
# =============================================================================

def main():
    """
    Main agent workflow:
    1. Load data
    2. Profile columns
    3. Run quality checks
    4. Calculate score
    5. Get AI recommendations
    6. Generate report
    """
    # Determine which file to analyze
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "sample_healthcare_claims.csv"
    
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           DataGuard AI — Data Quality Agent                ║")
    print("║           Healthcare Dataset Profiling & Analysis          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Step 1: Load
    df = load_dataset(file_path)
    
    # Step 2: Profile
    profiles = profile_columns(df)
    
    # Step 3: Quality checks
    issues = run_quality_checks(df, profiles)
    
    # Step 4: Score
    score, grade = calculate_quality_score(issues)
    
    # Step 5: AI recommendations
    recommendations = get_ai_recommendations(df, profiles, issues, score)
    
    # Step 6: Report
    report = generate_report(df, profiles, issues, score, grade, recommendations)
    
    # Print the full report
    print(report)
    
    # Save the report to a file
    report_path = "data_quality_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Also save issues as JSON for pipeline integration
    json_path = "data_quality_issues.json"
    with open(json_path, 'w') as f:
        json.dump({
            'score': score,
            'grade': grade,
            'total_issues': len(issues),
            'issues': issues,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Issues JSON saved to: {json_path}")
    print()


if __name__ == "__main__":
    main()
