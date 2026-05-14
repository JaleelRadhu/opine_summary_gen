"""Save and load evaluator responses.

Primary backend: Google Sheets (when credentials are configured in st.secrets).
Fallback backend: local CSV file (for local runs without Sheets configured).
"""
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd

from .config import RESPONSES_CSV

COLUMNS = [
    "evaluator_id", "evaluator_name", "eval_id", "course",
    "node_name", "node_type", "node_level",
    "task1_guess", "task1_skipped",
    "task2_faithfulness", "task2_coverage", "task2_coherence",
    "task3_pos_alignment", "task3_neg_alignment",
    "task4_propagation",
    "comments", "timestamp",
]


def _use_sheets() -> bool:
    try:
        from .sheets_client import sheets_configured
        return sheets_configured()
    except Exception:
        return False


# ── Google Sheets backend ──────────────────────────────────────────────────

def _save_response_sheets(response: dict) -> None:
    from .sheets_client import get_worksheet
    ws = get_worksheet("responses")
    if not ws.get_all_values():  # empty sheet — write header first
        ws.append_row(COLUMNS)
    row = [str(response.get(col, "")) for col in COLUMNS]
    ws.append_row(row)


def _load_responses_sheets() -> pd.DataFrame:
    from .sheets_client import get_worksheet
    ws = get_worksheet("responses")
    rows = ws.get_all_values()
    if len(rows) < 2:
        return pd.DataFrame(columns=COLUMNS)
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


# ── Local CSV backend ──────────────────────────────────────────────────────

def _save_response_csv(response: dict) -> None:
    RESPONSES_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESPONSES_CSV.exists()
    with open(RESPONSES_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(response)


def _load_responses_csv() -> pd.DataFrame:
    if not RESPONSES_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RESPONSES_CSV)


# ── Public API ─────────────────────────────────────────────────────────────

def save_response(response: Dict[str, Any]) -> None:
    response = dict(response)
    response["timestamp"] = datetime.now().isoformat()
    if _use_sheets():
        _save_response_sheets(response)
    else:
        _save_response_csv(response)


def load_responses() -> pd.DataFrame:
    if _use_sheets():
        return _load_responses_sheets()
    return _load_responses_csv()


def completed_ids_for(evaluator_id: str) -> Set[str]:
    df = load_responses()
    if df.empty:
        return set()
    return set(df[df["evaluator_id"] == evaluator_id]["eval_id"].tolist())
