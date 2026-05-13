"""
Regenerate human eval packages from already-saved summaries.
Use after changing eval selection logic (e.g. MAX_LEAF_TUPLES) without
needing to re-run the full LLM pipeline.

Usage:
    conda run -n jaleel_btp python regenerate_eval.py \
        --dataset acad-real --model Qwen2-5-7B-Instruct
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.taxonomy import load_taxonomy, path_to_key
from src.leaf_summaries import group_tuples_by_leaf
from src.human_eval import run_human_eval, render_markdown

N_COURSES = 3


def _model_slug(model_name: str) -> str:
    name = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def load_reviews(dataset_name: str, course_num: int) -> list:
    data_path = Path("data") / dataset_name / f"course_{course_num}.jsonl"
    reviews = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    return reviews


def rebuild_leaf_results(
    leaf_summaries: dict,
    reviews: list,
    nodes: dict,
) -> dict:
    """Re-attach source_tuples to leaf summary data loaded from disk."""
    grouped = group_tuples_by_leaf(reviews)
    leaf_results = {}
    for node_name, node_data in leaf_summaries.items():
        path_key = path_to_key(node_data["node_path"])
        source_tuples = grouped.get(path_key, [])
        leaf_results[node_name] = {**node_data, "source_tuples": source_tuples}
    return leaf_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="acad-real")
    parser.add_argument("--model",   default="Qwen2-5-7B-Instruct",
                        help="Model slug (directory name under outputs/)")
    args = parser.parse_args()

    slug = _model_slug(args.model) if "/" in args.model else args.model
    output_base = Path("outputs") / args.dataset / slug

    print(f"Regenerating eval for {args.dataset} / {slug}")
    nodes = load_taxonomy()

    all_records = []

    for course_num in range(1, N_COURSES + 1):
        course_dir = output_base / f"course_{course_num}"

        leaf_path     = course_dir / "leaf_summaries.json"
        internal_path = course_dir / "internal_summaries.json"
        course_path   = course_dir / "course_summary.json"

        if not leaf_path.exists():
            print(f"  [SKIP] course_{course_num}: no leaf_summaries.json found")
            continue

        leaf_on_disk     = json.loads(leaf_path.read_text())
        internal_results = json.loads(internal_path.read_text()) if internal_path.exists() else {}
        root_data        = json.loads(course_path.read_text()) if course_path.exists() else None

        # Reload original reviews to recover source_tuples
        reviews = load_reviews(args.dataset, course_num)
        leaf_results = rebuild_leaf_results(leaf_on_disk, reviews, nodes)

        # Build full summaries lookup
        all_summaries = {**leaf_results, **internal_results}
        if root_data:
            all_summaries["root"] = root_data

        records = run_human_eval(
            args.dataset, course_num,
            leaf_results, internal_results, all_summaries,
            output_base,
        )
        all_records.extend(records)
        print(f"  course_{course_num}: {len(records)} eval records")

    # Save combined
    eval_dir = output_base / "human_eval"
    all_eval_path = eval_dir / "all_eval.json"
    with open(all_eval_path, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_records)} total records → {all_eval_path}")

    md_path = eval_dir / "evaluation_sheet.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(all_records))
    print(f"Saved evaluation sheet → {md_path}")


if __name__ == "__main__":
    main()
