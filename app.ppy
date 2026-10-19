"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Natural-language query engine for the commercial lending portfolio.

Pipeline (same as the notebook):
    1. Intent classification  -> verified template (VQ1-VQ10) or generated SQL
    2. Query construction     -> template from the library, or LLM-generated SELECT
    3. Validation gate        -> read-only shape, schema check, EXPLAIN dry run,
                                 LLM relevance check, template integrity check
    4. Retry once             -> generated route only, then escalate to a human analyst
    5. Execution              -> read-only SQLite connection
    6. Response generation    -> business narrative + SQL + raw data + confidence

Run locally:
    streamlit run app.py
"""

from typing import TypedDict, List, Any, Dict, Optional
import json
import os
import re
import sqlite3
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk Query Engine | Northbridge Bank",
    page_icon="🏦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent


# ======================================================================================
# DATABASE SCHEMA (provided to the LLM in the prompts)
# ======================================================================================

database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ======================================================================================
# VERIFIED QUERY TEMPLATE LIBRARY
# ======================================================================================

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions,
    ROUND(
        SUM(CASE
            WHEN l.asset_classification IN
                ('Substandard', 'Doubtful', 'Loss')
            THEN l.total_outstanding
            ELSE 0
        END) / 1000000.0, 2
    ) AS npa_exposure_millions
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_millions DESC;"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_millions DESC;"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 4)
        AS avg_provision_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC;"""

# VQ5: Top 10 Loan Exposures
sql_5 = """SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2)
        AS outstanding_millions,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10;"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL
    AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5;"""

# VQ7: All Overdue Loan Accounts
sql_7 = """SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2)
        AS outstanding_millions,
    l.days_past_due,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;"""

# VQ8: DPD Bucket Distribution
sql_8 = """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END;"""

# VQ9: Latest Rating Downgrades
sql_9 = """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
    AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2)
        AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ======================================================================================
# TOOL DEFINITIONS (identical logic to the notebook)
# ======================================================================================
# NOTE: `llm` and `evaluator_llm` are module-level globals, created in the
# "App setup" section further below (before any of these functions is called).

def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are a query router for a credit risk analytics system at Northbridge Bank.

Your job is to determine whether a business user's question can be
answered by an existing verified SQL template or requires fresh SQL generation.

Compare the user's question against the verified query library based
on its meaning, not just its exact wording.

IMPORTANT ROUTING RULES:
- Select the "verified" route only when an existing template can answer
  the complete question.
- Compare the requested metrics, filters, grouping, and time period.
- Similar keywords alone are not enough to select a verified template.
- Questions about sector-level outstanding exposure and non-performing
  loan exposure should match VQ1 when it provides the necessary metrics,
  even if the question names a specific sector.
- VQ3 provides the latest quarter's stage distribution.
- Do NOT select VQ3 for Stage 3 EAD/ECL trends over multiple quarters.
- Questions about Stage 3 EAD/ECL trends over four quarters require
  the "generated" route.
- If no verified template fully answers the question, select "generated"
  and set query_id to null.



{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

You are a senior SQL developer specializing in credit risk analytics using a SQLite database.

Generate an accurate SQL query based on the user's question and the provided database schema.

Use only the available tables and columns. Generate read-only SELECT queries and return only SQL without explanations.

For quarterly trend questions:
    - Use the latest four reporting dates from the provisioning table.
    - Do not generate future reporting dates.
    - For Stage 3, filter ifrs9_stage = 3.
    - Calculate EAD using SUM(ead_amount).
    - Calculate ECL using SUM(ecl_amount).
    - Group by reporting_date.

{user_question}

{schema_context}

"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Clean the SQL query of any trailing semicolons or spaces for dry runs
    candidate_sql_clean = candidate_sql.strip().rstrip(';').strip()

    # Check 1: Read-only shape check
    sql_upper = candidate_sql_clean.upper()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql_clean:
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql_clean)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql_clean.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql_clean}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

You are a senior data validator specializing in credit risk analytics for Northbridge Bank using a SQLite database. Your job is to determine whether a SQL query correctly answers a business user's question.

Evaluate the SQL query against the user's question and the provided database schema.

Verify that the query uses the correct tables, columns, joins, filters, and calculations.

Ensure the query follows SQLite syntax and contains only read-only SELECT operations.

Identify any errors or inconsistencies and provide a brief explanation of your findings.

{track_context}

{user_question}

{candidate_sql_clean}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    # FIX: Use SELECT * FROM (...) LIMIT 0 to avoid syntax errors with existing templates that already contain a LIMIT clause.
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';').strip()
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql_clean}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    Ensures the query is optimized, contians NO multiple statements, and uses correct aggregation contexts and date filters.
    '''

    retry_prompt = f"""
An analytical SQLite query validation. Rewrite it to fix the error while preserving the original user intent.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### CRITICAL RULES TO FIX PREVIOUS ERRORS
1. Do NOT write multiple SELECT statements separated by semicolons. Use a single unified query (with CTEs if necessary).
2. If analyzing restructured loans (is_restructured = 1), ensure you correctly filter or join the provision data to include only the relevant restructured loans.
3. Make sure any ratio or percentage calculations (like impaired percentage) are properly aggregated in the main query to avoid incorrect results or misuse of window functions.
4. NPA / Impairment definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss') OR ifrs9_stage = 3.
5. Return ONLY the raw SQLite query, without markdown, without comments, and without explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

You are a senior credit risk analyst at Northbridge Bank. Your job is to translate SQL query results into clear and accurate business insights.

Review the business user's question and the provided query results.

Generate a concise, professional response that directly answers the user's question.

Use only the information provided in the query results. Do not make assumptions or invent information.

Highlight relevant figures, trends, or findings when appropriate.

Use simple business language and avoid unnecessary technical explanations.

{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ======================================================================================
# PIPELINE ORCHESTRATION
# ======================================================================================

def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages in
                      log['pipeline_trace'] (the notebook printed them; the
                      Streamlit app shows them in an expander instead).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        # Streamlit additions (UI / audit only - they do not affect pipeline logic)
        'warnings': [],
        'pipeline_trace': []
    }

    def _emit(message):
        if verbose:
            log['pipeline_trace'].append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    _emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    _emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    _emit(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    _emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        _emit(f"    Failed check: {gate.get('failed_check')}")
        _emit(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        _emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        _emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            _emit(f"    Retry failed check: {gate.get('failed_check')}")
            _emit(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        _emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = list(exec_result['warnings'])

    _emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        _emit(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    _emit(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ======================================================================================
# APP SETUP: configuration, read-only database connection, LLMs
# ======================================================================================

def get_setting(name, default=None):
    """Look up a setting in Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value not in (None, ""):
                return str(value)
    except Exception:
        pass
    value = os.environ.get(name)
    return value if value else default


def resolve_path(path_str):
    """Resolve a path relative to the folder that contains app.py."""
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (APP_DIR / p)


def load_config_json():
    """Optional fallback, same idea as the notebook's config.json."""
    cfg_path = resolve_path(get_setting("CONFIG_JSON_PATH", "config.json"))
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception:
            return {}
    return {}


_config_json = load_config_json()
OPENAI_API_KEY = get_setting("OPENAI_API_KEY") or _config_json.get("OPENAI_API_KEY")
OPENAI_API_BASE = get_setting("OPENAI_API_BASE") or _config_json.get("OPENAI_API_BASE")
MODEL_NAME = get_setting("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = resolve_path(get_setting("DB_PATH", "credit_risk_portfolio.db"))
AUDIT_LOG_PATH = get_setting("AUDIT_LOG_PATH")  # optional: append every run to a JSONL file

# Store API credentials in environment variables (langchain-openai reads them)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Northbridge Bank · Commercial lending portfolio · Read-only analytical queries · "
    "Verified SQL templates with validated, LLM-generated SQL for everything else"
)

setup_errors = []
if not OPENAI_API_KEY:
    setup_errors.append(
        "**OPENAI_API_KEY is not set.** Add it to `.streamlit/secrets.toml` (or the Streamlit Cloud "
        "*Secrets* box), set it as an environment variable, or put it in a `config.json` next to `app.py`."
    )
if not DB_PATH.exists():
    setup_errors.append(
        f"**Database not found:** `{DB_PATH}`. Place `credit_risk_portfolio.db` next to `app.py` "
        "or set `DB_PATH`."
    )
if setup_errors:
    for msg in setup_errors:
        st.error(msg)
    st.stop()


@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    """Read-only SQLite connection (URI mode with ?mode=ro blocks any write operation)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_resource(show_spinner=False)
def get_llms(model_name: str):
    """Same two LLM instances as the notebook (generator/router and evaluator)."""
    _llm = ChatOpenAI(model=model_name, temperature=0)
    _evaluator_llm = ChatOpenAI(model=model_name, temperature=0)
    return _llm, _evaluator_llm


try:
    conn = get_connection(str(DB_PATH))
    conn.execute("SELECT 1 FROM loan_master LIMIT 1").fetchall()
except Exception as exc:
    st.error(f"Could not open the database in read-only mode: {exc}")
    st.stop()

try:
    llm, evaluator_llm = get_llms(MODEL_NAME)
except Exception as exc:
    st.error(f"Could not initialise the LLM client: {exc}")
    st.stop()


# ======================================================================================
# UI HELPERS
# ======================================================================================

DEFAULT_EXAMPLE_QUESTIONS = [
    "How many loans are overdue?",
    "What is the total outstanding and NPA exposure for the Real Estate sector?",
    "Show me the days-past-due aging profile of the portfolio.",
    "What is the average interest rate by sector?",
    "Show the Stage 3 EAD and ECL trend over the last four quarters.",
    "Who are the top 5 business groups by outstanding exposure?",
]


def find_ground_truth_path():
    candidates = [get_setting("GROUND_TRUTH_PATH"), "test_queries-2.csv", "test_queries.csv"]
    for cand in candidates:
        if cand:
            p = resolve_path(cand)
            if p.exists():
                return p
    return None


def escape_dollars(text):
    """Streamlit markdown treats $...$ as LaTeX; escape so currency renders normally."""
    return str(text).replace("$", r"\$")


def format_confidence(conf):
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return f"{conf:.0%}"
    return "N/A" if conf is None else str(conf)


def make_error_result(question, error):
    """Placeholder result used when the pipeline itself raises (e.g. API/network error)."""
    return {
        'log': {}, 'dataframe': None, 'user_question': question, 'route': None,
        'query_id': None, 'match_reason': None, 'candidate_sql': None, 'gate_result': None,
        'retry_used': False, 'escalated': False, 'executed_sql': None, 'row_count': None,
        'confidence': 'ERROR', 'narrative': error, 'warnings': [], 'pipeline_trace': [],
        'error': error,
    }


def record_audit(question, result, error, elapsed_seconds):
    """Append an audit record for every analytical output (session + optional JSONL file)."""
    gate = (result or {}).get('gate_result') or {}
    if error:
        status = "ERROR"
    elif result.get('escalated'):
        status = "ESCALATED"
    else:
        status = "ANSWERED"
    record = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'status': status,
        'user_question': question,
        'route': (result or {}).get('route'),
        'query_id': (result or {}).get('query_id'),
        'match_reason': (result or {}).get('match_reason'),
        'retry_used': (result or {}).get('retry_used'),
        'validation_passed': gate.get('passed'),
        'failed_check': gate.get('failed_check'),
        'validation_details': gate.get('details'),
        'confidence': (result or {}).get('confidence'),
        'candidate_sql': (result or {}).get('candidate_sql'),
        'executed_sql': (result or {}).get('executed_sql'),
        'row_count': (result or {}).get('row_count'),
        'warnings': '; '.join((result or {}).get('warnings') or []),
        'narrative': (result or {}).get('narrative'),
        'error': error,
        'elapsed_seconds': round(elapsed_seconds, 2),
    }
    st.session_state.audit_log.append(record)
    if AUDIT_LOG_PATH:
        try:
            with open(resolve_path(AUDIT_LOG_PATH), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # audit file is best-effort; the in-session audit trail is always kept


def execute_pipeline(question):
    """Run the pipeline once, never raise, and write the audit record."""
    started = time.perf_counter()
    error = None
    result = None
    try:
        result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
    except Exception as exc:  # API errors, JSON parse errors, etc.
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    record_audit(question, result, error, elapsed)
    if result is None:
        result = make_error_result(question, error)
    result['elapsed_seconds'] = elapsed
    return result


def render_result(result, key_prefix):
    """Show an answer: narrative, confidence, SQL used, raw data, validation details."""
    if result.get('error'):
        st.error(f"The pipeline failed before it could produce an answer: {result['error']}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(result['route']).title())
    c2.metric("Verified query", result['query_id'] or "—")
    c3.metric("Confidence", format_confidence(result['confidence']))
    c4.metric("Rows returned", "—" if result['row_count'] is None else result['row_count'])
    if result.get('match_reason'):
        st.caption(f"Routing reason: {result['match_reason']}")
    if result.get('retry_used'):
        st.info("The first generated SQL failed validation, so the query was regenerated once.")

    if result['escalated']:
        st.error("⚠️ This question could not be reliably resolved and has been escalated to a human analyst.")
        st.markdown(escape_dollars(result['narrative']))
        st.markdown("**Candidate SQL (for analyst review)**")
        st.code(result['candidate_sql'], language="sql")
    else:
        st.markdown("#### Answer")
        st.markdown(escape_dollars(result['narrative']))

        for w in result.get('warnings') or []:
            st.warning(w)

        st.markdown("#### SQL used")
        source = ("Pre-approved verified template " + str(result['query_id'])
                  if result['route'] == 'verified' else "Generated SQL (passed validation)")
        st.caption(source)
        st.code(result['executed_sql'], language="sql")

        st.markdown("#### Raw data")
        st.dataframe(result['dataframe'])
        st.download_button(
            "Download result data (CSV)",
            data=result['dataframe'].to_csv(index=False),
            file_name=f"{key_prefix}_result.csv",
            mime="text/csv",
            key=f"{key_prefix}_download",
        )

    with st.expander("Validation gate details"):
        st.json(result.get('gate_result') or {})
    with st.expander("Pipeline trace"):
        st.code("\n".join(result.get('pipeline_trace') or []) or "(no trace recorded)")
    if result.get('elapsed_seconds') is not None:
        st.caption(f"Completed in {result['elapsed_seconds']:.1f} seconds")


def compute_evaluation(ground_truth, test_results):
    """Evaluation against ground truth (same calculations as the notebook)."""
    evaluation_rows = []

    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        tr = test_results[i]

        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count']
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

    # Convert Confidence to numeric, coercing strings like 'ESCALATED' to NaN
    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# ======================================================================================
# SESSION STATE
# ======================================================================================

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "eval_ground_truth" not in st.session_state:
    st.session_state.eval_ground_truth = None

gt_path = find_ground_truth_path()


def set_question(q):
    st.session_state.question_input = q


# ======================================================================================
# SIDEBAR
# ======================================================================================

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Route** – your question is matched to a pre-approved SQL template, or sent to SQL generation.\n"
        "2. **Validate** – read-only, schema, parse/plan, relevance and template-integrity checks.\n"
        "3. **Retry once** – generated SQL gets one corrected attempt, then goes to a human analyst.\n"
        "4. **Answer** – narrative, SQL used, raw data and a confidence score; every run is logged."
    )
    st.divider()
    st.markdown(
        f"**Model:** `{MODEL_NAME}`  \n"
        f"**Database:** `{DB_PATH.name}` (read-only)  \n"
        "**Latest reporting / rating date:** 2025-09-30"
    )
    st.divider()
    st.subheader("Example questions")
    for i, example in enumerate(DEFAULT_EXAMPLE_QUESTIONS):
        st.button(example, key=f"example_{i}", on_click=set_question, args=(example,))


# ======================================================================================
# MAIN TABS
# ======================================================================================

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["💬 Ask a question", "📚 Verified query library", "✅ Evaluation", "🧾 Audit trail"]
)

# ---------------------------------------------------------------- Ask a question
with tab_ask:
    st.markdown(
        "Ask a routine commercial lending portfolio question in plain English. "
        "Complex or ambiguous questions are escalated to a human analyst."
    )
    st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="e.g. What is the total outstanding and NPA exposure for the Real Estate sector?",
    )
    run_clicked = st.button("Run query", type="primary")

    if run_clicked:
        question = st.session_state.question_input.strip()
        if not question:
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Routing, validating and running your query..."):
                st.session_state.last_result = execute_pipeline(question)

    if st.session_state.last_result is not None:
        st.divider()
        st.caption(f"Question: {st.session_state.last_result['user_question']}")
        render_result(st.session_state.last_result, "ask")

# ---------------------------------------------------------------- Verified library
with tab_library:
    st.markdown(
        f"The library contains **{len(verified_query_library)}** pre-approved SQL templates. "
        "Each runs without modification; the LLM only writes the narrative from the full result."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'], language="sql")
    with st.expander("Database schema provided to the LLM"):
        st.code(database_schema)

# ---------------------------------------------------------------- Evaluation
with tab_eval:
    st.markdown(
        "Runs every test case in the ground-truth CSV through the pipeline and compares the selected "
        "route and verified query with the expected values."
    )

    ground_truth = None
    if gt_path is not None:
        ground_truth = pd.read_csv(gt_path)
        st.caption(f"Ground truth loaded from `{gt_path.name}` ({len(ground_truth)} test cases)")
    else:
        uploaded = st.file_uploader("Upload the ground-truth CSV (test_queries.csv)", type="csv")
        if uploaded is not None:
            ground_truth = pd.read_csv(uploaded)

    required_cols = ['Test Case', 'User Query', 'Expected Route', 'Expected Query ID']
    if ground_truth is not None:
        missing_cols = [c for c in required_cols if c not in ground_truth.columns]
        if missing_cols:
            st.error(f"The ground-truth CSV is missing required columns: {missing_cols}")
            ground_truth = None

    if ground_truth is not None:
        st.dataframe(ground_truth)
        if st.button("Run all test cases", type="primary"):
            test_results = []
            progress = st.progress(0.0, text="Starting...")
            total = len(ground_truth)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                progress.progress(i / total, text=f"Running {gt['Test Case']} ({i + 1}/{total})")
                test_results.append(execute_pipeline(gt['User Query']))
            progress.progress(1.0, text="Done")
            st.session_state.eval_results = test_results
            st.session_state.eval_ground_truth = ground_truth

    if st.session_state.eval_results is not None and st.session_state.eval_ground_truth is not None:
        eval_gt = st.session_state.eval_ground_truth
        eval_results = st.session_state.eval_results
        evaluation_df, path_accuracy, query_accuracy, average_confidence = compute_evaluation(eval_gt, eval_results)

        m1, m2, m3 = st.columns(3)
        m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
        m2.metric("Selected Query Accuracy", "N/A" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%")
        m3.metric("Average Confidence Score", "N/A" if pd.isna(average_confidence) else f"{average_confidence:.2f}")

        display_df = evaluation_df.copy()
        display_df['Confidence'] = display_df['Confidence'].astype(str)
        st.dataframe(display_df)

        st.markdown("#### Test case outputs")
        for i, (_, gt) in enumerate(eval_gt.iterrows()):
            with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                if 'Expected Answer' in eval_gt.columns:
                    st.markdown("**Expected answer (ground truth)**")
                    st.markdown(escape_dollars(gt['Expected Answer']))
                render_result(eval_results[i], f"eval_{i}")

# ---------------------------------------------------------------- Audit trail
with tab_audit:
    st.markdown(
        "Every analytical output produced in this session is logged with its route, SQL, "
        "validation outcome, confidence and narrative."
    )
    if AUDIT_LOG_PATH:
        st.caption(f"Each run is also appended to `{AUDIT_LOG_PATH}`.")
    if st.session_state.audit_log:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        audit_df['confidence'] = audit_df['confidence'].astype(str)
        st.dataframe(audit_df)
        a1, a2, a3 = st.columns(3)
        a1.download_button("Download audit trail (CSV)", data=audit_df.to_csv(index=False),
                           file_name="audit_trail.csv", mime="text/csv")
        a2.download_button("Download audit trail (JSON)",
                           data=json.dumps(st.session_state.audit_log, indent=2, default=str),
                           file_name="audit_trail.json", mime="application/json")
        if a3.button("Clear session audit trail"):
            st.session_state.audit_log = []
            st.rerun()
    else:
        st.info("No queries have been run in this session yet.")
