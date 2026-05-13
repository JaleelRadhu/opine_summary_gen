"""
Compute evaluation metrics and generate paper-ready tables + plots.

Usage (from project root):
    conda run -n jaleel_btp python eval_app/compute_report.py
    conda run -n jaleel_btp python eval_app/compute_report.py --model Qwen2-5-3B-Instruct
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_app.config import DATASET, MODEL_SLUG, EVAL_DIR, RESULTS_DIR, RESPONSES_CSV
from eval_app.storage import load_responses

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping plots")

REPORT_DIR = RESULTS_DIR / "report"


# ── Krippendorff alpha (ordinal, simplified) ───────────────────────────────
def krippendorff_alpha_ordinal(ratings_matrix: np.ndarray) -> float:
    """
    ratings_matrix: shape (n_items, n_raters), NaN for missing.
    Ordinal metric difference: (v1 - v2)^2 / scale_range^2
    """
    data = ratings_matrix.copy().astype(float)
    n_items, n_raters = data.shape
    values = data[~np.isnan(data)]
    if len(values) == 0:
        return float("nan")
    vmin, vmax = values.min(), values.max()
    scale_range = vmax - vmin if vmax != vmin else 1.0

    Do = 0.0
    n_obs = 0
    De_pairs = []

    for i in range(n_items):
        row = data[i][~np.isnan(data[i])]
        m = len(row)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                d = ((row[a] - row[b]) / scale_range) ** 2
                Do += d
                n_obs += 1

    if n_obs == 0:
        return float("nan")
    Do /= n_obs

    all_vals = values / scale_range
    for i in range(len(all_vals)):
        for j in range(i + 1, len(all_vals)):
            De_pairs.append((all_vals[i] - all_vals[j]) ** 2)

    De = np.mean(De_pairs) if De_pairs else 1.0
    if De == 0:
        return 1.0
    return 1.0 - Do / De


# ── Sentiment distribution from eval records ──────────────────────────────
def load_sentiment_distribution() -> pd.DataFrame:
    """
    Load sentiment counts per eval record from all_eval.json.
    'total' key is absent from saved tuple_counts, so we compute it from parts.
    Each row is keyed by eval_id (unique) to allow correct joins later.
    """
    all_eval_path = EVAL_DIR / "human_eval" / "all_eval.json"
    records = json.loads(all_eval_path.read_text())

    rows = []
    for r in records:
        tc  = r.get("tuple_counts", {})
        pos = tc.get("positive", 0)
        neg = tc.get("negative", 0)
        sug = tc.get("suggestion", 0)
        rows.append({
            "eval_id":    r["eval_id"],          # unique identifier — use for joins
            "node_name":  r["node_name"],
            "node_level": r["node_level"],
            "node_type":  r["node_type"],
            "positive":   pos,
            "negative":   neg,
            "suggestion": sug,
            "total":      pos + neg + sug,       # computed from parts, never 0-div
        })
    return pd.DataFrame(rows)


def _pct(n: int, tot: int) -> str:
    return f"{round(n / max(tot, 1) * 100, 1)}%"


def report_sentiment_distribution(df_sent: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("SENTIMENT DISTRIBUTION BY NODE LEVEL")
    print("=" * 60)

    summary_rows = []
    for level in sorted(df_sent["node_level"].unique()):
        subset = df_sent[df_sent["node_level"] == level]
        pos = int(subset["positive"].sum())
        neg = int(subset["negative"].sum())
        sug = int(subset["suggestion"].sum())
        tot = pos + neg + sug                   # always correct
        label = f"Level {level}" if level > 0 else "Level 0 (root)"
        row = {
            "Level": label, "Nodes": len(subset),
            "Positive": pos, "Negative": neg, "Suggestion": sug, "Total": tot,
            "% Positive":   round(pos / max(tot, 1) * 100, 1),
            "% Negative":   round(neg / max(tot, 1) * 100, 1),
            "% Suggestion": round(sug / max(tot, 1) * 100, 1),
        }
        summary_rows.append(row)
        print(f"  {label:20s}  +{pos:4d} / -{neg:4d} / >{sug:4d}  "
              f"({_pct(pos,tot)} / {_pct(neg,tot)} / {_pct(sug,tot)})")

    pd.DataFrame(summary_rows).to_csv(out_dir / "sentiment_distribution.csv", index=False)
    print(f"  → Saved {out_dir / 'sentiment_distribution.csv'}")

    print("\n  Leaf vs Internal:")
    for ntype in ["leaf", "internal"]:
        mask = df_sent["node_type"].str.contains("leaf") if ntype == "leaf" \
               else ~df_sent["node_type"].str.contains("leaf")
        s = df_sent[mask]
        pos = int(s["positive"].sum())
        neg = int(s["negative"].sum())
        sug = int(s["suggestion"].sum())
        tot = pos + neg + sug
        print(f"    {ntype.capitalize():10s}  +{pos} / -{neg} / >{sug}  "
              f"({_pct(pos,tot)} / {_pct(neg,tot)} / {_pct(sug,tot)})")


# ── Task 1: Sentiment Guess Accuracy ─────────────────────────────────────
def report_task1(df: pd.DataFrame, df_sent: pd.DataFrame, out_dir: Path) -> None:
    t1 = df[~df["task1_skipped"].fillna(False).astype(bool)].copy()
    t1 = t1.dropna(subset=["task1_guess"])
    if t1.empty:
        print("\n[Task 1] No data yet.")
        return

    # Build actual_score (0-10) keyed by eval_id — avoids duplicate node_name problem
    # (same node_name can appear in multiple courses with different tuple counts)
    actual = df_sent.set_index("eval_id").apply(
        lambda row: round(row["positive"] / max(row["total"], 1) * 10),
        axis=1,
    ).rename("actual_score")
    t1 = t1.join(actual, on="eval_id", how="left")
    t1 = t1.dropna(subset=["actual_score"])

    t1["error"] = t1["task1_guess"] - t1["actual_score"]
    t1["abs_error"] = t1["error"].abs()

    mae  = t1["abs_error"].mean()
    bias = t1["error"].mean()
    corr = t1["task1_guess"].corr(t1["actual_score"])

    print("\n" + "=" * 60)
    print("TASK 1 — SENTIMENT PREDICTION ACCURACY")
    print("=" * 60)
    print(f"  Responses  : {len(t1)}")
    print(f"  MAE (0-10) : {mae:.2f}")
    print(f"  Bias       : {bias:+.2f}  ({'over' if bias > 0 else 'under'}-estimates positivity)")
    print(f"  Pearson r  : {corr:.3f}")

    t1.to_csv(out_dir / "task1_detail.csv", index=False)

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(t1["actual_score"], t1["task1_guess"], alpha=0.6)
        ax.plot([0, 10], [0, 10], "k--", lw=1, label="Perfect")
        ax.set_xlabel("Actual score (0-10)")
        ax.set_ylabel("Evaluator guess (0-10)")
        ax.set_title("Sentiment Prediction: Guess vs Actual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "task1_scatter.png", dpi=150)
        plt.close(fig)
        print("  → Saved task1_scatter.png")


# ── Task 2: Summary Quality Ratings ──────────────────────────────────────
def report_task2(df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["task2_faithfulness", "task2_coverage", "task2_coherence"]
    t2 = df.dropna(subset=cols)

    if t2.empty:
        print("\n[Task 2] No data yet.")
        return

    print("\n" + "=" * 60)
    print("TASK 2 — SUMMARY QUALITY RATINGS (1-5)")
    print("=" * 60)
    print(f"  Total responses: {len(t2)}")

    rows = []
    for dim, col in zip(["Faithfulness", "Coverage", "Coherence"], cols):
        vals = t2[col].dropna()
        row = {
            "Dimension": dim,
            "N": len(vals),
            "Mean": round(vals.mean(), 3),
            "SD":   round(vals.std(), 3),
            "Min":  int(vals.min()),
            "Max":  int(vals.max()),
        }
        rows.append(row)
        print(f"  {dim:15s}  Mean={row['Mean']:.2f} ± {row['SD']:.2f}")

    # By node type
    print("\n  By node type:")
    for ntype in t2["node_type"].unique():
        subset = t2[t2["node_type"] == ntype]
        f_mean = subset["task2_faithfulness"].mean()
        c_mean = subset["task2_coverage"].mean()
        h_mean = subset["task2_coherence"].mean()
        print(f"    {ntype:30s}  F={f_mean:.2f}  C={c_mean:.2f}  H={h_mean:.2f}")

    # By level
    print("\n  By node level:")
    for lvl in sorted(t2["node_level"].dropna().unique()):
        subset = t2[t2["node_level"] == lvl]
        f_mean = subset["task2_faithfulness"].mean()
        c_mean = subset["task2_coverage"].mean()
        h_mean = subset["task2_coherence"].mean()
        print(f"    Level {int(lvl):2d}  F={f_mean:.2f}  C={c_mean:.2f}  H={h_mean:.2f}")

    pd.DataFrame(rows).to_csv(out_dir / "task2_summary.csv", index=False)
    t2.groupby("node_level")[cols].mean().round(3).to_csv(out_dir / "task2_by_level.csv")

    # IAA on evaluators who rated the same record
    print("\n  Inter-Annotator Agreement (Krippendorff α, ordinal):")
    for dim, col in zip(["Faithfulness", "Coverage", "Coherence"], cols):
        pivot = t2.pivot_table(index="eval_id", columns="evaluator_id", values=col, aggfunc="first")
        if pivot.shape[1] >= 2:
            alpha = krippendorff_alpha_ordinal(pivot.values)
            print(f"    {dim:15s}  α = {alpha:.3f}")
        else:
            print(f"    {dim:15s}  α = N/A (need ≥2 evaluators per item)")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        means = [t2[c].mean() for c in cols]
        stds  = [t2[c].std() for c in cols]
        labels = ["Faithfulness", "Coverage", "Coherence"]
        x = range(3)
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=["#4a90d9", "#e87040", "#50a850"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(1, 5)
        ax.set_ylabel("Mean rating (1-5)")
        ax.set_title("Summary Quality Ratings (mean ± SD)")
        fig.tight_layout()
        fig.savefig(out_dir / "task2_ratings.png", dpi=150)
        plt.close(fig)
        print("  → Saved task2_ratings.png")


# ── Task 3: Section Alignment ─────────────────────────────────────────────
def report_task3(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("TASK 3 — CONSOLIDATION ALIGNMENT")
    print("=" * 60)

    align_order = ["Yes, fully", "Mostly", "Partially", "No"]

    for label, col in [("Positive (A Appraisals vs B)", "task3_pos_alignment"),
                        ("Negative (A Criticisms vs C)",  "task3_neg_alignment")]:
        t3 = df.dropna(subset=[col])
        if t3.empty:
            print(f"  {label}: no data")
            continue
        counts = t3[col].value_counts()
        total  = len(t3)
        good   = counts.get("Yes, fully", 0) + counts.get("Mostly", 0)
        print(f"\n  {label}  (n={total})")
        for opt in align_order:
            n = counts.get(opt, 0)
            print(f"    {opt:15s}: {n:3d}  ({round(n/total*100)}%)")
        print(f"    ── Agreement rate (Yes+Mostly): {round(good/total*100)}%")

    t3_any = df.dropna(subset=["task3_pos_alignment", "task3_neg_alignment"], how="all")
    if not t3_any.empty:
        t3_any[["eval_id", "node_name", "task3_pos_alignment", "task3_neg_alignment"]]\
            .to_csv(out_dir / "task3_detail.csv", index=False)


# ── Task 4: Propagation Quality ───────────────────────────────────────────
def report_task4(df: pd.DataFrame, out_dir: Path) -> None:
    t4 = df.dropna(subset=["task4_propagation"])

    print("\n" + "=" * 60)
    print("TASK 4 — PROPAGATION QUALITY (1-5)")
    print("=" * 60)

    if t4.empty:
        print("  No data yet.")
        return

    print(f"  Responses: {len(t4)}")
    print(f"  Mean: {t4['task4_propagation'].mean():.2f} ± {t4['task4_propagation'].std():.2f}")

    print("\n  By source node level:")
    for lvl in sorted(t4["node_level"].dropna().unique()):
        subset = t4[t4["node_level"] == lvl]
        print(f"    Level {int(lvl):2d}  Mean={subset['task4_propagation'].mean():.2f}  "
              f"(n={len(subset)})")

    by_level = t4.groupby("node_level")["task4_propagation"].agg(["mean", "std", "count"])\
                 .round(3).rename(columns={"mean": "Mean", "std": "SD", "count": "N"})
    by_level.to_csv(out_dir / "task4_by_level.csv")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 4))
        levels = by_level.index.astype(int)
        means  = by_level["Mean"].values
        stds   = by_level["SD"].fillna(0).values
        ax.errorbar(levels, means, yerr=stds, marker="o", capsize=5, linewidth=2)
        ax.set_xlabel("Source node level")
        ax.set_ylabel("Propagation quality (1-5)")
        ax.set_ylim(1, 5)
        ax.set_title("Propagation Quality by Node Level")
        ax.invert_xaxis()   # root at right, leaves at left
        fig.tight_layout()
        fig.savefig(out_dir / "task4_propagation.png", dpi=150)
        plt.close(fig)
        print("  → Saved task4_propagation.png")


# ── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_SLUG,
                        help="Model slug to report on (default: config.MODEL_SLUG)")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset : {DATASET}")
    print(f"Model   : {args.model}")
    print(f"Responses file: {RESPONSES_CSV}")

    df = load_responses()
    if df.empty:
        print("\nNo responses yet. Run the evaluation app first.")
        return

    print(f"\nTotal responses loaded: {len(df)}")
    print(f"Unique evaluators: {df['evaluator_id'].nunique()}")
    print(f"Unique records evaluated: {df['eval_id'].nunique()}")

    df["node_level"] = pd.to_numeric(df["node_level"], errors="coerce")

    df_sent = load_sentiment_distribution()

    report_sentiment_distribution(df_sent, out_dir)
    report_task1(df, df_sent, out_dir)
    report_task2(df, out_dir)
    report_task3(df, out_dir)
    report_task4(df, out_dir)

    # Save full annotated responses
    df.to_csv(out_dir / "all_responses_annotated.csv", index=False)

    print(f"\n✅ Report saved to {out_dir}/")


if __name__ == "__main__":
    main()
