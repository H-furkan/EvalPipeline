"""
metrics/rouge_eval.py — ROUGE-L evaluation.

Computes ROUGE-L F1 between each method's slide text and the source paper text.
No LLM required — purely text overlap.

Dependencies: pip install evaluate rouge_score
Output: results/rouge_eval.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from utils.result_utils import (
    load_existing,
    make_metadata,
    read_processed_text,
    result_path,
    save_incremental,
)

METRIC_NAME = "rouge_eval"


def _compute_rouge_l(hypothesis: str, reference: str) -> float:
    """Return ROUGE-L F1 score between hypothesis and reference."""
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        scores = rouge.compute(
            predictions=[hypothesis],
            references=[reference],
            rouge_types=["rougeL"],
        )
        return float(scores["rougeL"])
    except ImportError:
        # Fallback: simple character-level LCS ratio
        def lcs_len(a: str, b: str) -> int:
            a_words = a.lower().split()
            b_words = b.lower().split()
            prev = [0] * (len(b_words) + 1)
            for w in a_words:
                curr = [0] * (len(b_words) + 1)
                for j, bw in enumerate(b_words, 1):
                    curr[j] = prev[j - 1] + 1 if w == bw else max(prev[j], curr[j - 1])
                prev = curr
            return prev[-1]

        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()
        lcs = lcs_len(hypothesis, reference)
        p = lcs / len(hyp_words) if hyp_words else 0.0
        r = lcs / len(ref_words) if ref_words else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return f1


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Compute ROUGE-L for all methods (ours + baselines) against each paper's source text.
    """
    out_path = result_path(METRIC_NAME)
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, "none (no LLM)")

    all_methods = [C.OURS_METHOD] + baseline_methods

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        reference = read_processed_text(paper, "orig")
        if reference is None:
            print(f"  Skipping: orig text not found")
            continue

        if paper not in per_paper:
            per_paper[paper] = {}

        for method in all_methods:
            if method in per_paper.get(paper, {}):
                print(f"  Skipping {method} (already done)")
                continue

            hyp = read_processed_text(paper, method)
            if hyp is None:
                print(f"  Skipping {method}: text not found")
                continue

            score = _compute_rouge_l(hyp, reference)
            per_paper[paper][method] = {"rouge_l": score}
            print(f"  {method}: ROUGE-L={score:.4f}")

        save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

    # Per-method summary
    per_method: dict[str, dict] = {}
    for method in all_methods:
        scores = [
            per_paper[p][method]["rouge_l"]
            for p in per_paper
            if method in per_paper[p]
        ]
        per_method[method] = {
            "mean_rouge_l": sum(scores) / len(scores) if scores else 0.0,
            "papers_evaluated": len(scores),
        }

    metadata["total_papers"] = len(per_paper)
    final = {
        "metadata": metadata,
        "per_method_summary": per_method,
        "per_paper": per_paper,
    }
    save_incremental(out_path, final)
    print(f"\n[{METRIC_NAME}] Done. Results → {out_path}")
    return final
