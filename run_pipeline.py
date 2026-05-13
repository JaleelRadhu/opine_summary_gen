#!/usr/bin/env python3
"""
Academic Feedback Summarisation Pipeline
=========================================
Usage:
  python run_pipeline.py --dataset acad-real --model Qwen/Qwen2.5-3B-Instruct
  python run_pipeline.py --dataset acad-syn  --model Qwen/Qwen2.5-3B-Instruct --skip-split
  python run_pipeline.py --dataset acad-real --model meta-llama/Llama-3-8B-Instruct --courses 1,2

The pipeline is model-aware: outputs are stored under outputs/<dataset>/<model-slug>/.
Run it once per model; the dataset splits (data/) are shared across models.
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Allow `src.*` imports when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.taxonomy import load_taxonomy
from src.llm_client import LLMClient
from src.split_dataset import run_split, DATASETS, N_COURSES
from src.leaf_summaries import run_leaf_summaries
from src.internal_summaries import run_internal_summaries
from src.human_eval import run_human_eval, render_markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _model_slug(model_name: str) -> str:
    name = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Academic feedback summarisation pipeline")
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()), required=True,
        help="Dataset to process (acad-real or acad-syn)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name as served by vLLM (e.g. Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--host", default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--skip-split", action="store_true",
        help="Skip dataset splitting — reuse existing splits in data/"
    )
    parser.add_argument(
        "--courses", default=None,
        help="Comma-separated course numbers to process, e.g. 1,2,3 (default: all)"
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    slug = _model_slug(args.model)
    output_base = root / "outputs" / args.dataset / slug

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    print("Loading taxonomy …")
    nodes = load_taxonomy()
    print(f"  {sum(1 for n in nodes.values() if n['is_leaf'])} leaf nodes, "
          f"{sum(1 for n in nodes.values() if not n['is_leaf'])} internal nodes")

    # ── Step 1: Split ─────────────────────────────────────────────────────────
    if not args.skip_split:
        print(f"\n{'='*60}\nStep 1 — Splitting {args.dataset}\n{'='*60}")
        run_split(args.dataset)
    else:
        print("\nSkipping split (--skip-split flag set)")

    # ── LLM client ────────────────────────────────────────────────────────────
    print(f"\nModel : {args.model}")
    print(f"Host  : {args.host}")
    print(f"Output: {output_base}")
    llm = LLMClient(base_url=args.host, model=args.model)

    # ── Course numbers ────────────────────────────────────────────────────────
    if args.courses:
        course_nums = [int(c.strip()) for c in args.courses.split(",")]
    else:
        course_nums = list(range(1, N_COURSES + 1))

    all_eval_records = []

    for course_num in course_nums:
        banner = f" {args.dataset} / {slug} / Course {course_num} "
        print(f"\n{'='*60}\n{banner:^60}\n{'='*60}")

        # Step 5 ── Leaf summaries
        print(f"\n─── Step 5 · Leaf Summaries ───")
        leaf_results = run_leaf_summaries(
            args.dataset, course_num, llm, nodes, output_base
        )

        # Step 6 ── Internal + course-level summaries
        print(f"\n─── Step 6 · Internal & Course Summary ───")
        internal_results, all_summaries = run_internal_summaries(
            leaf_results, nodes, llm, course_num, output_base
        )

        # Steps 8-9 ── Human eval package
        print(f"\n─── Steps 8-9 · Human Eval Package ───")
        records = run_human_eval(
            args.dataset, course_num,
            leaf_results, internal_results, all_summaries,
            output_base,
        )
        all_eval_records.extend(records)

    # ── Combined outputs ──────────────────────────────────────────────────────
    eval_dir = output_base / "human_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_eval_path = eval_dir / "all_eval.json"
    with open(all_eval_path, "w") as f:
        json.dump(all_eval_records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved combined eval ({len(all_eval_records)} records) → {all_eval_path}")

    md_path = eval_dir / "evaluation_sheet.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(all_eval_records))
    print(f"Saved evaluation sheet → {md_path}")

    print(f"\n{'='*60}")
    print(f"Pipeline complete: {args.dataset} / {slug}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
