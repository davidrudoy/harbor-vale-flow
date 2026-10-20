"""Crew 1, Agent 1: Data Ingestion.

Design rule: numbers are computed by plain Python (the tool),
judgement is left to the LLM (the agent).
"""
from pathlib import Path

import pandas as pd
from crewai import Agent, Task
from crewai.tools import tool

ROOT = Path(__file__).resolve().parents[2]  # project root (two folders up from this file)
RAW_CSV = ROOT / "data" / "raw" / "retail_sales.csv"
SUMMARY_MD = ROOT / "artifacts" / "crew1" / "ingest_summary.md"


@tool("read_csv_summary")
def read_csv_summary() -> str:
    """Load the raw retail sales CSV, write a factual summary file, and return it."""
    # Why a tool: an LLM cannot open files. It can only call functions we give it.
    df = pd.read_csv(RAW_CSV)
    lines = [
        "# Ingest summary",
        f"- File: data/raw/{RAW_CSV.name}",
        f"- Rows: {len(df)}, Columns: {df.shape[1]}",
        "",
        "| Column | Type | Missing |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {df[c].dtype} | {df[c].isna().sum()} |" for c in df.columns]
    text = "\n".join(lines)
    # Why write here and not via the Task: the facts must never pass through the LLM.
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(text, encoding="utf-8")
    return text


ingest_agent = Agent(
    role="Data Ingestion Specialist",
    goal=(
        "Load data/raw/retail_sales.csv and document its structure exactly "
        "as it is, without changing anything"
    ),
    backstory=(
        "You are a careful data engineer. You report facts about the file "
        "and never guess or fix values."
    ),
    tools=[read_csv_summary],
    llm="gemini/gemini-3.8-flash",
    allow_delegation=False,  # keeps the flow predictable: no handing work to other agents
)

ingest_task = Task(
    description=(
        "Call the read_csv_summary tool. Then, based only on its output, "
        "point out anything in the structure that looks suspicious or worth a "
        "second look (types, missing values, column names)."
    ),
    expected_output="A short 'Observations' section: 2-4 bullet points, no invented numbers.",
    agent=ingest_agent,
    output_file=str(ROOT / "artifacts" / "crew1" / "ingest_observations.md"),
)
