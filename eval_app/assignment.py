"""Assign eval records to evaluators with a fixed core overlap set.

Primary backend: Google Sheets (when credentials are configured in st.secrets).
Fallback backend: local JSON registry file.
"""
import hashlib
import json
import random
from typing import Any, Dict, List

from .config import REGISTRY_FILE, N_RECORDS_PER_EVALUATOR, N_CORE_RECORDS


def _use_sheets() -> bool:
    try:
        from .sheets_client import sheets_configured
        return sheets_configured()
    except Exception:
        return False


def _compute_assignment(
    evaluator_id: str,
    all_records: List[Dict],
    n_per_eval: int,
    n_core: int,
) -> List[Dict]:
    sorted_records = sorted(all_records, key=lambda r: r["eval_id"])
    n = len(sorted_records)

    step = max(1, n // max(n_core, 1))
    core_indices = set(range(0, n, step)[:n_core])
    core_records = [sorted_records[i] for i in sorted(core_indices)]
    non_core = [r for i, r in enumerate(sorted_records) if i not in core_indices]

    seed = int(hashlib.md5(evaluator_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    pool = non_core[:]
    rng.shuffle(pool)

    n_extra = min(n_per_eval - len(core_records), len(pool))
    assigned = core_records + pool[:n_extra]
    rng.shuffle(assigned)
    return assigned


# ── Google Sheets backend ──────────────────────────────────────────────────

def _read_registry_sheets() -> Dict[str, List[str]]:
    from .sheets_client import get_worksheet
    ws = get_worksheet("registry")
    records = ws.get_all_records()
    return {row["evaluator_id"]: json.loads(row["assigned_ids"]) for row in records}


def _write_registry_sheets(evaluator_id: str, assigned_ids: List[str]) -> None:
    from .sheets_client import get_worksheet
    ws = get_worksheet("registry")
    if ws.row_count == 0 or not ws.get("A1"):
        ws.append_row(["evaluator_id", "assigned_ids"])
    ws.append_row([evaluator_id, json.dumps(assigned_ids)])


# ── Local JSON backend ─────────────────────────────────────────────────────

def _read_registry_local() -> Dict[str, List[str]]:
    if not REGISTRY_FILE.exists():
        return {}
    return json.loads(REGISTRY_FILE.read_text())


def _write_registry_local(evaluator_id: str, assigned_ids: List[str]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry = _read_registry_local()
    registry[evaluator_id] = assigned_ids
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


# ── Public API ─────────────────────────────────────────────────────────────

def get_assignment(
    evaluator_id: str,
    all_records: List[Dict[str, Any]],
    n_per_eval: int = N_RECORDS_PER_EVALUATOR,
    n_core: int = N_CORE_RECORDS,
) -> List[Dict[str, Any]]:
    use_sheets = _use_sheets()
    registry = _read_registry_sheets() if use_sheets else _read_registry_local()

    if evaluator_id in registry:
        assigned_ids = set(registry[evaluator_id])
        return [r for r in all_records if r["eval_id"] in assigned_ids]

    assigned = _compute_assignment(evaluator_id, all_records, n_per_eval, n_core)
    assigned_ids = [r["eval_id"] for r in assigned]

    if use_sheets:
        _write_registry_sheets(evaluator_id, assigned_ids)
    else:
        _write_registry_local(evaluator_id, assigned_ids)

    return assigned
