"""Step 5: Generate leaf-node summaries for each course partition."""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .taxonomy import get_leaf_nodes, load_taxonomy, path_to_key
from .llm_client import LLMClient, load_prompt

log = logging.getLogger(__name__)

_SENTIMENT_LABEL = {"positive": "Positive", "negative": "Negative", "suggestion": "Suggestion"}


def _format_tuple(aspect: str, opinion: str, sentiment: str, sentence: str) -> str:
    label = _SENTIMENT_LABEL.get(sentiment, sentiment)
    return f"Aspect: {aspect} | Opinion: {opinion} | Sentiment: {label}\nReview: {sentence}"


def _build_content(tuples: List[Dict]) -> str:
    return "\n\n".join(
        _format_tuple(t["aspect"], t["opinion"], t["sentiment"], t["source_feedback"])
        for t in tuples
    )


def _build_filtered_content(tuples: List[Dict], sentiment: str) -> Optional[str]:
    filtered = [t for t in tuples if t["sentiment"] == sentiment]
    return _build_content(filtered) if filtered else None


def group_tuples_by_leaf(reviews: List[Dict]) -> Dict[tuple, List[Dict]]:
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for review in reviews:
        sentence = review.get("sentence", "")
        question = review.get("question", "")
        for aspect, opinion, sentiment, path in zip(
            review.get("aspects", []),
            review.get("opinions", []),
            review.get("sentiments", []),
            review.get("new_category_paths", []),
        ):
            key = path_to_key(path)
            grouped[key].append({
                "aspect": aspect,
                "opinion": opinion,
                "sentiment": sentiment,
                "source_feedback": sentence,
                "question": question,
            })
    return grouped


def generate_leaf_summaries(
    reviews: List[Dict],
    llm: LLMClient,
    nodes: Dict[tuple, dict],
    course_label: str,
) -> Dict[str, Any]:
    prompts = {
        "consolidated": load_prompt("leaf_consolidated.txt"),
        "positive":     load_prompt("leaf_positive.txt"),
        "negative":     load_prompt("leaf_negative.txt"),
        "suggestion":   load_prompt("leaf_suggestion.txt"),
    }

    leaf_nodes = {path_to_key(n["path"]): n for n in get_leaf_nodes(nodes)}
    grouped = group_tuples_by_leaf(reviews)

    # Warn about data paths that don't land on a taxonomy leaf
    for path_key in grouped:
        if path_key not in leaf_nodes:
            log.warning("Path %s from data is not a leaf in taxonomy — skipping", path_key)

    results: Dict[str, Any] = {}

    for leaf_path, node in leaf_nodes.items():
        tuples = grouped.get(leaf_path, [])
        if not tuples:
            continue

        node_name = node["name"]
        counts = {"positive": 0, "negative": 0, "suggestion": 0}
        for t in tuples:
            s = t["sentiment"]
            if s in counts:
                counts[s] += 1
        counts["total"] = sum(counts.values())

        print(f"  [{course_label}] Leaf: {node_name!r}  ({counts['total']} tuples)")

        sys1 = prompts["consolidated"].replace("{node_name}", node_name)
        type1 = llm.generate(sys1, _build_content(tuples), max_tokens=600)

        pos_text = _build_filtered_content(tuples, "positive")
        sys2 = prompts["positive"].replace("{node_name}", node_name)
        type2 = llm.generate(sys2, pos_text) if pos_text else None

        neg_text = _build_filtered_content(tuples, "negative")
        sys3 = prompts["negative"].replace("{node_name}", node_name)
        type3 = llm.generate(sys3, neg_text) if neg_text else None

        sug_text = _build_filtered_content(tuples, "suggestion")
        sys4 = prompts["suggestion"].replace("{node_name}", node_name)
        type4 = llm.generate(sys4, sug_text) if sug_text else None

        results[node_name] = {
            "node_path": node["path"],
            "node_level": node["level"],
            "is_negative_sibling_style": node["is_negative_sibling_style"],
            "type1_consolidated": type1,
            "type2_positive": type2,
            "type3_negative": type3,
            "type4_suggestion": type4,
            "tuple_counts": counts,
            "source_tuples": tuples,  # kept in memory for human eval; not written to disk
        }

    return results


def run_leaf_summaries(
    dataset_name: str,
    course_num: int,
    llm: LLMClient,
    nodes: Dict[tuple, dict],
    output_base: Path,
) -> Dict[str, Any]:
    data_path = (
        Path(__file__).parent.parent / "data" / dataset_name / f"course_{course_num}.jsonl"
    )
    reviews = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))

    out_dir = output_base / f"course_{course_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = generate_leaf_summaries(reviews, llm, nodes, f"course_{course_num}")

    # Write disk format (without source_tuples)
    disk = {
        name: {k: v for k, v in data.items() if k != "source_tuples"}
        for name, data in results.items()
    }
    out_path = out_dir / "leaf_summaries.json"
    with open(out_path, "w") as f:
        json.dump(disk, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(results)} leaf summaries → {out_path}")
    return results  # includes source_tuples for human eval downstream
