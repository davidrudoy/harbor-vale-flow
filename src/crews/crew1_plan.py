"""Crew 1, Agent 2: Preprocessing Planner.

The LLM only PROPOSES a plan (JSON). Plain Python checks it against a closed
list of operations. The LLM never writes or runs code.
"""
import json

import pandas as pd
from crewai import Agent, Task
from crewai.tools import tool

from src.crews.crew1_ingest import RAW_CSV, ROOT

PROFILE_JSON = ROOT / "artifacts" / "crew1" / "profile.json"
PLAN_JSON = ROOT / "artifacts" / "crew1" / "cleaning_plan.json"

# The closed list: operation -> arguments it must have ("column" is checked against the real columns).
OPS = {
    "parse_date": {"column"},
    "cast_type": {"column", "dtype"},
    "drop_duplicates": set(),  # optional "column" = the key; without it, whole rows are compared
    "fill_missing": {"column", "strategy"},  # strategy "constant" also needs "value"
    "drop_rows_with_missing": {"column"},
    "strip_whitespace": {"column"},
    "standardize_category_case": {"column", "case"},
}
# Allowed values for arguments that are not column names.
CHOICES = {
    "dtype": {"int", "float", "str"},
    "strategy": {"mean", "median", "mode", "constant"},
    "case": {"lower", "upper", "title"},
}


def check_plan(plan, columns) -> list[str]:
    """Return a list of problems with the plan; an empty list means it is valid."""
    # Why plain code and not the LLM: a rule that decides what runs on the data must be deterministic.
    if not isinstance(plan, list):
        return ["plan must be a JSON list of steps"]
    errors = []
    for i, step in enumerate(plan):
        if not isinstance(step, dict) or step.get("op") not in OPS:
            errors.append(f"step {i}: unknown or missing op {step!r}")
            continue
        op = step["op"]
        for key in OPS[op] - step.keys():
            errors.append(f"step {i} ({op}): missing '{key}'")
        if "column" in step and step["column"] not in columns:
            errors.append(f"step {i} ({op}): column '{step['column']}' does not exist")
        for key, allowed in CHOICES.items():
            if key in step and step[key] not in allowed:
                errors.append(f"step {i} ({op}): '{key}' must be one of {sorted(allowed)}")
        if step.get("strategy") == "constant" and "value" not in step:
            errors.append(f"step {i} ({op}): strategy 'constant' needs 'value'")
    return errors


@tool("profile_data")
def profile_data() -> str:
    """Profile the raw CSV (types, missing values, ranges, samples), save it, and return it as JSON."""
    df = pd.read_csv(RAW_CSV)
    profile = {"rows": len(df), "duplicate_rows": int(df.duplicated().sum()), "columns": {}}
    for c in df.columns:
        info = {
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique()),
            "samples": [str(v) for v in df[c].dropna().unique()[:5]],
        }
        if pd.api.types.is_numeric_dtype(df[c]):
            info["min"], info["max"] = float(df[c].min()), float(df[c].max())
        profile["columns"][c] = info
    text = json.dumps(profile, indent=2)
    PROFILE_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_JSON.write_text(text, encoding="utf-8")
    return text


@tool("validate_plan")
def validate_plan(plan_json: str) -> str:
    """Check a cleaning plan (JSON list of steps) against the allowed operations. Returns 'VALID' or the problems."""
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as e:
        return f"INVALID: not valid JSON ({e})"
    errors = check_plan(plan, pd.read_csv(RAW_CSV, nrows=0).columns)
    return "VALID" if not errors else "INVALID:\n" + "\n".join(errors)


planner_agent = Agent(
    role="Preprocessing Planner",
    goal=(
        "Read the data profile and decide which preprocessing operations, "
        "from the allowed list only, are needed. Output a JSON plan."
    ),
    backstory=(
        "You are a cautious data engineer. You only propose operations that the "
        "profile clearly justifies. You never change money values or guess units; "
        "if something looks wrong, you flag it instead."
    ),
    tools=[profile_data, validate_plan],
    llm="gemini/gemini-3.8-flash",
    allow_delegation=False,  # fixed pipeline: Planner -> Cleaner, no side conversations
)

plan_task = Task(
    description=(
        "1. Call profile_data. 2. Decide which preprocessing is needed, using ONLY these "
        f"operations: {json.dumps({k: sorted(v) for k, v in OPS.items()})}. "
        f"Allowed values: {json.dumps({k: sorted(v) for k, v in CHOICES.items()})}. "
        "A date column stored as text needs parse_date. Add nothing the profile does not justify. "
        "3. Call validate_plan with your plan and fix it until it says VALID."
    ),
    expected_output=(
        'Only the final plan as a raw JSON list, no code fences, no text around it. '
        'Example: [{"op": "parse_date", "column": "Date"}]'
    ),
    agent=planner_agent,
    output_file=str(PLAN_JSON),
)
