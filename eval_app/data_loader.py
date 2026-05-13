"""Load and enrich eval records for the Streamlit app."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import EVAL_DIR, DATASET, MODEL_SLUG


def _load_all_summaries() -> Dict[Tuple[str, str], Dict]:
    """
    Load leaf + internal summaries for all courses into a dict
    keyed by (course_label, node_name).
    """
    summaries: Dict[Tuple[str, str], Dict] = {}
    for course_num in range(1, 4):
        course_key = f"Course-{course_num}"
        course_dir = EVAL_DIR / f"course_{course_num}"
        for fname in ("leaf_summaries.json", "internal_summaries.json"):
            fpath = course_dir / fname
            if not fpath.exists():
                continue
            data = json.loads(fpath.read_text())
            for node_name, node_data in data.items():
                summaries[(course_key, node_name)] = node_data
    return summaries


def _enrich_record(record: Dict, summaries: Dict[Tuple[str, str], Dict]) -> Dict:
    """Add _parent_name and _parent_summary fields to a record."""
    path = record.get("node_path", [])
    course = record.get("course", "")
    if len(path) >= 2:
        parent_name = path[-2]
        parent_data = summaries.get((course, parent_name))
        if parent_data:
            record["_parent_name"] = parent_name
            record["_parent_summary"] = parent_data.get("type1_consolidated")
    return record


def load_eval_records() -> List[Dict[str, Any]]:
    """Load all_eval.json, enrich with parent summaries, return list of records."""
    all_eval_path = EVAL_DIR / "human_eval" / "all_eval.json"
    if not all_eval_path.exists():
        raise FileNotFoundError(
            f"No eval records found at {all_eval_path}.\n"
            f"Run: python run_pipeline.py --dataset {DATASET} "
            f"--model ... --skip-split"
        )
    records: List[Dict] = json.loads(all_eval_path.read_text())
    summaries = _load_all_summaries()
    return [_enrich_record(r, summaries) for r in records]


def is_suggestions_node(node_path: List[str]) -> bool:
    """True if this node is the Suggestions subtree (skip sentiment guess task)."""
    return "Suggestions" in node_path


def actual_positive_score(tuple_counts: Dict) -> int:
    """Convert actual positive proportion to 0-10 scale."""
    total = tuple_counts.get("total", 0)
    if total == 0:
        return 5
    pos = tuple_counts.get("positive", 0)
    return round(pos / total * 10)


def extract_section(summary_a: str, section: str) -> str:
    """Pull a named section (Appraisals/Criticisms/Suggestions) from Summary A."""
    sections = ["## Appraisals", "## Criticisms", "## Suggestions"]
    if section not in sections:
        return ""
    try:
        start_marker = section
        start = summary_a.index(start_marker) + len(start_marker)
        # Find the next section header
        remaining = summary_a[start:]
        next_pos = len(remaining)
        for other in sections:
            if other != section and other in remaining:
                next_pos = min(next_pos, remaining.index(other))
        return remaining[:next_pos].strip()
    except ValueError:
        return ""
