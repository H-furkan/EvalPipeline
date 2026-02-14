"""
utils/result_utils.py — Result persistence and aggregation helpers.

Every metric uses save_incremental() after each paper so that a partial run
can be resumed without re-doing completed papers.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C


# ── Persistence ───────────────────────────────────────────────────────────────

def save_incremental(path: Path | str, data: dict) -> None:
    """
    Atomically write *data* to *path* as JSON.

    Uses a .tmp sibling file and os.rename() so a crash never leaves a
    half-written result file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_existing(path: Path | str) -> dict:
    """
    Load a previously saved result file.
    Returns an empty dict if the file does not exist or is corrupt.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def result_path(metric_name: str) -> Path:
    """Return the standard output path for a given metric name."""
    return Path(C.RESULTS_DIR) / f"{metric_name}.json"


# ── Metadata helpers ──────────────────────────────────────────────────────────

def make_metadata(metric_name: str, model: str, extra: dict | None = None) -> dict:
    """Build a standard metadata block for a result file."""
    meta = {
        "metric": metric_name,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "backend": C.LLM_BACKEND,
        "ours_method": C.OURS_METHOD,
        "baseline_methods": C.BASELINE_METHODS,
        "seed": C.SEED,
    }
    if extra:
        meta.update(extra)
    return meta


# ── Aggregation helpers ───────────────────────────────────────────────────────

def aggregate_pairwise_wins(
    per_paper: dict[str, dict[str, Any]],
    baseline_methods: list[str],
) -> tuple[dict, dict]:
    """
    Aggregate per-paper pairwise win/loss counts into per-method and overall summaries.

    Args:
        per_paper:        {paper_name: {baseline: {"ours_wins": 0|1, ...}}}
        baseline_methods: List of baseline method names.

    Returns:
        (per_method_summary, overall_summary)
    """
    per_method: dict[str, dict] = {}
    total_comparisons = 0
    total_ours_wins = 0
    total_baseline_wins = 0
    total_unknown = 0

    for baseline in baseline_methods:
        ours_wins = 0
        baseline_wins = 0
        unknown = 0
        total = 0

        for paper_results in per_paper.values():
            if baseline not in paper_results:
                continue
            r = paper_results[baseline]
            total += 1
            w = r.get("winner", "unknown")
            if w == "ours":
                ours_wins += 1
            elif w == "baseline":
                baseline_wins += 1
            else:
                unknown += 1

        per_method[baseline] = {
            "total_comparisons": total,
            "ours_wins": ours_wins,
            "baseline_wins": baseline_wins,
            "unknown": unknown,
            "ours_win_rate": ours_wins / total if total > 0 else 0.0,
            "baseline_win_rate": baseline_wins / total if total > 0 else 0.0,
        }
        total_comparisons += total
        total_ours_wins += ours_wins
        total_baseline_wins += baseline_wins
        total_unknown += unknown

    overall = {
        "total_comparisons": total_comparisons,
        "ours_total_wins": total_ours_wins,
        "baseline_total_wins": total_baseline_wins,
        "unknown_total": total_unknown,
        "ours_overall_win_rate": (
            total_ours_wins / total_comparisons if total_comparisons > 0 else 0.0
        ),
    }
    return per_method, overall


# ── Console summary ───────────────────────────────────────────────────────────

def print_pairwise_summary(metric_name: str, result: dict) -> None:
    """Print a human-readable summary of a pairwise metric result."""
    overall = result.get("overall_summary", {})
    per_method = result.get("per_method_summary", {})
    ours = result.get("metadata", {}).get("ours_method", C.OURS_METHOD)

    print(f"\n{'='*60}")
    print(f"METRIC: {metric_name.upper()}")
    print(f"{'='*60}")
    print(f"Papers evaluated : {result.get('metadata', {}).get('total_papers', '?')}")
    print(f"Total comparisons: {overall.get('total_comparisons', 0)}")
    print(f"Overall win rate ({ours}): {overall.get('ours_overall_win_rate', 0):.1%}")
    print("\nPer-baseline breakdown:")
    for baseline, s in sorted(per_method.items()):
        print(
            f"  vs {baseline:30s}  "
            f"{s['ours_wins']:2d}/{s['total_comparisons']:2d} "
            f"({s['ours_win_rate']:.1%})"
        )


def print_scalar_summary(metric_name: str, result: dict) -> None:
    """Print a human-readable summary of a scalar metric result (ROUGE, PPL, etc.)."""
    per_method = result.get("per_method_summary", {})
    print(f"\n{'='*60}")
    print(f"METRIC: {metric_name.upper()}")
    print(f"{'='*60}")
    for method, stats in sorted(per_method.items()):
        scores = {k: v for k, v in stats.items() if isinstance(v, (int, float))}
        row = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"  {method:30s}  {row}")


# ── File path helpers ────────────────────────────────────────────────────────

def get_processed_text_path(paper_name: str, method_or_orig: str) -> Path | None:
    """
    Find the processed text file for a paper/method, supporting both .md and .txt.

    Looks for .md first (new format), falls back to .txt (legacy format).
    Returns the Path if found, None otherwise.
    """
    base = Path(C.PROCESSED_DATA_DIR) / paper_name / method_or_orig
    for ext in (".md", ".txt"):
        path = base.with_suffix(ext)
        if path.exists():
            return path
    return None


def read_processed_text(paper_name: str, method_or_orig: str) -> str | None:
    """
    Read the processed text for a paper/method.

    Supports both .md and .txt files (prefers .md).
    Returns the text content or None if not found.
    """
    path = get_processed_text_path(paper_name, method_or_orig)
    if path is not None:
        return path.read_text(encoding="utf-8")
    return None
