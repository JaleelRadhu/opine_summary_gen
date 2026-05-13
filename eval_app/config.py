"""Central configuration for the evaluation app.
Change MODEL_SLUG to switch which model's summaries are evaluated.
"""
from pathlib import Path

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET = "acad-real"

# ── Model — change this to switch models ─────────────────────────────────────
# Available: "Qwen2-5-7B-Instruct", "Qwen2-5-3B-Instruct"
MODEL_SLUG = "Qwen2-5-7B-Instruct"

# ── Assignment parameters ─────────────────────────────────────────────────────
N_RECORDS_PER_EVALUATOR = 7   # total records shown to each evaluator
N_CORE_RECORDS = 3            # records shown to ALL evaluators (for IAA)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
EVAL_DIR     = OUTPUTS_DIR / DATASET / MODEL_SLUG
RESULTS_DIR  = PROJECT_ROOT / "eval_results"
RESPONSES_CSV = RESULTS_DIR / f"responses_{DATASET}_{MODEL_SLUG}.csv"
REGISTRY_FILE = RESULTS_DIR / f"registry_{DATASET}_{MODEL_SLUG}.json"
