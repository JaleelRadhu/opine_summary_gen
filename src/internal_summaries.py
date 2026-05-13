"""Step 6: Bottom-up propagation of summaries through internal nodes (and root)."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .taxonomy import get_processing_order
from .llm_client import LLMClient, load_prompt

log = logging.getLogger(__name__)


def _child_block(child_name: str, t2: Optional[str], t3: Optional[str], t4: Optional[str]) -> str:
    return (
        f"Child Topic: {child_name}\n"
        f"Appraisals: {t2 or 'None'}\n"
        f"Criticisms: {t3 or 'None'}\n"
        f"Suggestions: {t4 or 'None'}\n"
        "---"
    )


def _filtered_child_content(
    children_paths: List[tuple],
    nodes: Dict[tuple, dict],
    all_summaries: Dict[str, Dict],
    field: str,
) -> Optional[str]:
    blocks = []
    for cp in children_paths:
        child_name = nodes[cp]["name"]
        child_data = all_summaries.get(child_name)
        if child_data is None:
            continue
        val = child_data.get(field)
        if val:
            blocks.append(f"Child Topic: {child_name}\n{val}")
    return "\n\n---\n\n".join(blocks) if blocks else None


def _aggregate_counts(
    children_paths: List[tuple],
    nodes: Dict[tuple, dict],
    all_summaries: Dict[str, Dict],
) -> Dict[str, int]:
    counts = {"positive": 0, "negative": 0, "suggestion": 0, "total": 0}
    for cp in children_paths:
        child_name = nodes[cp]["name"]
        child_data = all_summaries.get(child_name)
        if child_data and "tuple_counts" in child_data:
            for k in counts:
                counts[k] += child_data["tuple_counts"].get(k, 0)
    return counts


def generate_internal_summaries(
    leaf_results: Dict[str, Any],
    nodes: Dict[tuple, dict],
    llm: LLMClient,
    course_label: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns (internal_results, all_summaries).
    all_summaries contains both leaf and internal entries keyed by node name.
    """
    prompts = {
        "consolidated": load_prompt("internal_consolidated.txt"),
        "positive":     load_prompt("internal_positive.txt"),
        "negative":     load_prompt("internal_negative.txt"),
        "suggestion":   load_prompt("internal_suggestion.txt"),
    }

    all_summaries: Dict[str, Dict] = dict(leaf_results)
    internal_results: Dict[str, Any] = {}

    for node_path in get_processing_order(nodes):
        node = nodes[node_path]
        node_name = node["name"]
        children_paths = node["children"]

        if not children_paths:
            continue  # leaf — already in all_summaries

        # Build combined child input (all three columns)
        children_used = []
        child_blocks = []
        for cp in children_paths:
            child_name = nodes[cp]["name"]
            child_data = all_summaries.get(child_name)
            if child_data is None:
                log.info("Skipping child %r for %r — no summary", child_name, node_name)
                continue
            children_used.append(child_name)
            child_blocks.append(_child_block(
                child_name,
                child_data.get("type2_positive"),
                child_data.get("type3_negative"),
                child_data.get("type4_suggestion"),
            ))

        if not child_blocks:
            log.info("No children with summaries for %r — skipping", node_name)
            continue

        all_content = "\n\n".join(child_blocks)
        print(f"  [{course_label}] Internal: {node_name!r}  ({len(children_used)} children)")

        sys1 = prompts["consolidated"].replace("{node_name}", node_name)
        type1 = llm.generate(sys1, all_content, max_tokens=800)

        pos_content = _filtered_child_content(children_paths, nodes, all_summaries, "type2_positive")
        sys2 = prompts["positive"].replace("{node_name}", node_name)
        type2 = llm.generate(sys2, pos_content, max_tokens=512) if pos_content else None

        neg_content = _filtered_child_content(children_paths, nodes, all_summaries, "type3_negative")
        sys3 = prompts["negative"].replace("{node_name}", node_name)
        type3 = llm.generate(sys3, neg_content, max_tokens=512) if neg_content else None

        sug_content = _filtered_child_content(children_paths, nodes, all_summaries, "type4_suggestion")
        sys4 = prompts["suggestion"].replace("{node_name}", node_name)
        type4 = llm.generate(sys4, sug_content, max_tokens=512) if sug_content else None

        result = {
            "node_path": node["path"],
            "node_level": node["level"],
            "is_negative_sibling_style": node["is_negative_sibling_style"],
            "type1_consolidated": type1,
            "type2_positive": type2,
            "type3_negative": type3,
            "type4_suggestion": type4,
            "tuple_counts": _aggregate_counts(children_paths, nodes, all_summaries),
            "children_used": children_used,
        }

        internal_results[node_name] = result
        all_summaries[node_name] = result

    return internal_results, all_summaries


def run_internal_summaries(
    leaf_results: Dict[str, Any],
    nodes: Dict[tuple, dict],
    llm: LLMClient,
    course_num: int,
    output_base: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out_dir = output_base / f"course_{course_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"course_{course_num}"

    internal_results, all_summaries = generate_internal_summaries(leaf_results, nodes, llm, label)

    out_path = out_dir / "internal_summaries.json"
    with open(out_path, "w") as f:
        json.dump(internal_results, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(internal_results)} internal summaries → {out_path}")

    root_data = all_summaries.get("root")
    if root_data:
        course_path = out_dir / "course_summary.json"
        # Exclude source_tuples if somehow present
        save_root = {k: v for k, v in root_data.items() if k != "source_tuples"}
        with open(course_path, "w") as f:
            json.dump(save_root, f, indent=2, ensure_ascii=False)
        print(f"  Saved course summary → {course_path}")

    return internal_results, all_summaries
