"""Steps 8-9: Generate human evaluation packages and the markdown evaluation sheet."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

N_REGULAR_LEAVES = 3
N_NEG_SIBLING_LEAVES = 1
MAX_LEAF_TUPLES = 20  # leaf nodes with >= this many tuples are excluded from eval selection


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def _child_summaries_for_record(
    data: Dict, all_summaries: Dict[str, Dict]
) -> Optional[Dict[str, Dict]]:
    children_used = data.get("children_used")
    if not children_used:
        return None
    return {
        child: {
            "type2_positive": all_summaries.get(child, {}).get("type2_positive"),
            "type3_negative": all_summaries.get(child, {}).get("type3_negative"),
            "type4_suggestion": all_summaries.get(child, {}).get("type4_suggestion"),
        }
        for child in children_used
    }


def _make_record(
    name: str,
    data: Dict,
    node_type: str,
    course_num: int,
    all_summaries: Dict[str, Dict],
) -> Dict:
    is_leaf = node_type in ("regular_leaf", "negative_sibling_leaf")
    return {
        "eval_id": f"course{course_num}_{_slugify(name)}",
        "course": f"Course-{course_num}",
        "node_name": name,
        "node_path": data["node_path"],
        "node_level": data["node_level"],
        "node_type": node_type,
        "source_tuples": data.get("source_tuples") if is_leaf else None,
        "child_summaries_used": (
            _child_summaries_for_record(data, all_summaries) if not is_leaf else None
        ),
        "tuple_counts": {
            k: data["tuple_counts"].get(k, 0)
            for k in ("positive", "negative", "suggestion")
        },
        "summary_A": data.get("type1_consolidated"),
        "summary_B": data.get("type2_positive"),
        "summary_C": data.get("type3_negative"),
        "summary_D": data.get("type4_suggestion"),
    }


def select_eval_nodes(
    leaf_results: Dict[str, Any],
    internal_results: Dict[str, Any],
    all_summaries: Dict[str, Any],
    course_num: int,
) -> List[Dict]:
    records: List[Dict] = []

    # --- Leaf nodes (only those with < MAX_LEAF_TUPLES, for manageable source docs) ---
    leaves = [
        (name, data) for name, data in leaf_results.items()
        if data.get("type1_consolidated") is not None
        and data.get("tuple_counts", {}).get("total", 0) < MAX_LEAF_TUPLES
    ]

    # Sort by total tuples descending so we pick the richest small nodes
    regular = sorted(
        [(n, d) for n, d in leaves if not d["is_negative_sibling_style"]],
        key=lambda x: -x[1]["tuple_counts"]["total"],
    )
    neg_sibling = sorted(
        [(n, d) for n, d in leaves if d["is_negative_sibling_style"]],
        key=lambda x: -x[1]["tuple_counts"]["total"],
    )

    for name, data in regular[:N_REGULAR_LEAVES]:
        records.append(_make_record(name, data, "regular_leaf", course_num, all_summaries))

    for name, data in neg_sibling[:N_NEG_SIBLING_LEAVES]:
        records.append(_make_record(name, data, "negative_sibling_leaf", course_num, all_summaries))

    # --- Internal nodes: one at level 2–3, one at level 1 ---
    internals = [
        (name, data) for name, data in internal_results.items()
        if data.get("type1_consolidated") is not None
    ]

    for level_range in [(2, 3), (1, 1)]:
        lo, hi = level_range
        candidates = sorted(
            [(n, d) for n, d in internals if lo <= d["node_level"] <= hi],
            key=lambda x: -x[1]["tuple_counts"]["total"],
        )
        if candidates:
            name, data = candidates[0]
            records.append(_make_record(name, data, "internal", course_num, all_summaries))

    # --- Root node ---
    root_data = all_summaries.get("root")
    if root_data and root_data.get("type1_consolidated") is not None:
        records.append(_make_record("root", root_data, "internal", course_num, all_summaries))

    return records


_NODE_TYPE_LABEL = {
    "regular_leaf": "Regular Leaf Node",
    "negative_sibling_leaf": "Negative-Sibling Leaf Node",
    "internal": "Internal Node",
}

_NA_POS = "Not generated — no positive feedback for this topic"
_NA_NEG = "Not generated — no negative feedback for this topic"
_NA_SUG = "Not generated — no suggestions for this topic"


def render_markdown(records: List[Dict]) -> str:
    sep = "=" * 45
    lines: List[str] = []

    for rec in records:
        lines += [
            sep,
            f"EVALUATION TASK: {rec['eval_id']}",
            f"TOPIC: {rec['node_name']}",
            f"TOPIC PATH: {' > '.join(rec['node_path'])}",
            f"NODE TYPE: {_NODE_TYPE_LABEL.get(rec['node_type'], rec['node_type'])}",
            sep,
            "",
            "--- SOURCE MATERIAL ---",
        ]

        if rec.get("source_tuples"):
            for i, t in enumerate(rec["source_tuples"], 1):
                lines.append(
                    f"{i}. Aspect: {t['aspect']} | Opinion: {t['opinion']} | Sentiment: {t['sentiment']}"
                )
                lines.append(f'   Review: "{t["source_feedback"]}"')

        elif rec.get("child_summaries_used"):
            for child_name, child_data in rec["child_summaries_used"].items():
                lines += [
                    f"Child: {child_name}",
                    f"  Positive: {child_data.get('type2_positive') or 'None'}",
                    f"  Negative: {child_data.get('type3_negative') or 'None'}",
                    f"  Suggestions: {child_data.get('type4_suggestion') or 'None'}",
                    "  ---",
                ]

        lines += [
            "",
            "--- SUMMARY A (Consolidated) ---",
            rec.get("summary_A") or "[Not generated]",
            "",
            "--- SUMMARY B (Positive Aspects) ---",
            rec.get("summary_B") or _NA_POS,
            "",
            "--- SUMMARY C (Criticisms) ---",
            rec.get("summary_C") or _NA_NEG,
            "",
            "--- SUMMARY D (Suggestions) ---",
            rec.get("summary_D") or _NA_SUG,
            "",
            sep,
            "",
        ]

    return "\n".join(lines)


def run_human_eval(
    dataset_name: str,
    course_num: int,
    leaf_results: Dict[str, Any],
    internal_results: Dict[str, Any],
    all_summaries: Dict[str, Any],
    output_base: Path,
) -> List[Dict]:
    records = select_eval_nodes(
        leaf_results, internal_results, all_summaries, course_num
    )

    out_dir = output_base / "human_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    course_path = out_dir / f"course_{course_num}_eval.json"
    with open(course_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(records)} eval records → {course_path}")

    return records
