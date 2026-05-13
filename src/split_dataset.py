"""Combine train/val/test splits and partition each dataset into N balanced course files."""
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATASETS = {
    "acad-real": {
        "files": ["train.jsonl", "val.jsonl", "test.jsonl"],
        "base": Path("/home/abdullahm/jaleel/opine_summary_gen/acad-real"),
    },
    "acad-syn": {
        "files": ["synthetic_train.jsonl", "synthetic_val.jsonl", "synthetic_test.jsonl"],
        "base": Path("/home/abdullahm/jaleel/opine_summary_gen/acad-syn"),
    },
}

N_COURSES = 3
DATA_OUT = Path(__file__).parent.parent / "data"


def load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    cfg = DATASETS[dataset_name]
    reviews = []
    for fname in cfg["files"]:
        path = cfg["base"] / fname
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    reviews.append(json.loads(line))
    return reviews


def _extract_strata(review: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(sentiment, L1_category) for each tuple in a review."""
    pairs = []
    for sentiment, path in zip(review.get("sentiments", []), review.get("new_category_paths", [])):
        l1 = path[1] if len(path) > 1 else "Unknown"
        pairs.append((sentiment, l1))
    return pairs


def stratified_split(
    reviews: List[Dict[str, Any]], n: int, seed: int = 42
) -> List[List[Dict[str, Any]]]:
    """
    Greedily assign reviews to n partitions minimising imbalance across
    (sentiment, L1_category) strata. Reviews are never split.
    """
    rng = random.Random(seed)
    shuffled = reviews[:]
    rng.shuffle(shuffled)

    partitions: List[List[Dict]] = [[] for _ in range(n)]
    counts: List[Dict[Tuple, int]] = [defaultdict(int) for _ in range(n)]
    totals: List[int] = [0] * n

    for review in shuffled:
        strata = _extract_strata(review)
        # Assign to the partition with the smallest current total tuples
        # (tie-break: smallest per-stratum count for this review's strata)
        best = min(
            range(n),
            key=lambda i: (totals[i], sum(counts[i][s] for s in strata)),
        )
        partitions[best].append(review)
        for s in strata:
            counts[best][s] += 1
        totals[best] += len(strata)

    return partitions


def print_distribution(partitions: List[List[Dict]], dataset_name: str) -> None:
    n = len(partitions)
    all_strata: Dict[Tuple, List[int]] = defaultdict(lambda: [0] * n)

    for i, part in enumerate(partitions):
        for review in part:
            for s in _extract_strata(review):
                all_strata[s][i] += 1

    sentiments = sorted({s for s, _ in all_strata})
    l1cats = sorted({l for _, l in all_strata})

    col_w = 12
    label_w = 48
    header = f"\n{'=' * 60}\nDataset: {dataset_name}\n{'=' * 60}"
    print(header)
    hdr = f"{'Stratum':<{label_w}}" + "".join(f"{'Course-'+str(i+1):>{col_w}}" for i in range(n))
    print(hdr)
    print("-" * len(hdr))

    for sent in sentiments:
        for l1 in l1cats:
            key = (sent, l1)
            if any(v > 0 for v in all_strata[key]):
                label = f"{sent[:14]} / {l1[:30]}"
                row = f"{label:<{label_w}}" + "".join(f"{all_strata[key][i]:>{col_w}}" for i in range(n))
                print(row)

    print("-" * len(hdr))
    review_counts = [len(p) for p in partitions]
    tuple_counts = [sum(len(r.get("sentiments", [])) for r in p) for p in partitions]
    print(f"{'Reviews':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in review_counts))
    print(f"{'Tuples (total)':<{label_w}}" + "".join(f"{c:>{col_w}}" for c in tuple_counts))
    print()


def run_split(dataset_name: str) -> None:
    out_dir = DATA_OUT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    reviews = load_dataset(dataset_name)
    print(f"Loaded {len(reviews)} reviews from {dataset_name}")

    partitions = stratified_split(reviews, N_COURSES)
    print_distribution(partitions, dataset_name)

    for i, part in enumerate(partitions, 1):
        out_path = out_dir / f"course_{i}.jsonl"
        with open(out_path, "w") as f:
            for review in part:
                f.write(json.dumps(review) + "\n")
        print(f"  Saved {out_path.name}  ({len(part)} reviews)")


if __name__ == "__main__":
    for ds in DATASETS:
        run_split(ds)
