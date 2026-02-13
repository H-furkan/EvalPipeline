"""
metrics/narrative_flow.py — Narrative Flow pairwise evaluation.

Compares OURS_METHOD against each baseline on a single criterion:
does the presentation preserve the logical narrative structure of the source paper?

Model: text LLM (Qwen2.5-32B-Instruct by default)
Input: processed text files from PROCESSED_DATA_DIR
Prompt: prompts/narrative_flow.txt
Output: results/narrative_flow.json
"""

import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from llm.client import call_text
from utils.result_utils import (
    aggregate_pairwise_wins,
    load_existing,
    make_metadata,
    result_path,
    save_incremental,
)

METRIC_NAME = "narrative_flow"
PROMPT_FILE = Path(C.PROMPTS_DIR) / "narrative_flow.txt"


def _load_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def _parse_winner(response: str, a_source: str, b_source: str) -> tuple[str, str]:
    """
    Parse 'Answer: A' or 'Answer: B' from the model response.
    Returns (winner, reasoning).
    """
    if response is None:
        return "unknown", ""

    match = re.search(r"Answer:\s*([AB])", response, re.IGNORECASE)
    if match:
        chosen = match.group(1).upper()
        reasoning = response[: match.start()].strip()
    else:
        upper = response.upper().strip()
        reasoning = response.strip()
        has_a = "A" in upper
        has_b = "B" in upper
        if has_a and not has_b:
            chosen = "A"
        elif has_b and not has_a:
            chosen = "B"
        else:
            # scan first occurrence
            chosen = "UNKNOWN"
            for ch in upper:
                if ch in ("A", "B"):
                    chosen = ch
                    break

    winner = a_source if chosen == "A" else (b_source if chosen == "B" else "unknown")
    return winner, reasoning


def _compare_one(
    template: str,
    reference: str,
    ours_text: str,
    baseline_text: str,
) -> dict:
    """Run a single pairwise comparison with randomised A/B assignment."""
    if random.random() < 0.5:
        option_a, option_b = ours_text, baseline_text
        a_source, b_source = "ours", "baseline"
    else:
        option_a, option_b = baseline_text, ours_text
        a_source, b_source = "baseline", "ours"

    prompt = template.format(
        reference=reference, option_a=option_a, option_b=option_b
    )
    response = call_text(prompt, model=C.TEXT_MODEL)
    winner, reasoning = _parse_winner(response, a_source, b_source)

    return {
        "assignment": {"A": a_source, "B": b_source},
        "winner": winner,
        "reasoning": reasoning,
        "raw_response": response or "ERROR",
    }


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Run narrative flow pairwise evaluation.

    Args:
        papers:           List of paper names to evaluate.
        baseline_methods: List of baseline method names to compare against OURS.

    Returns:
        Result dict (also saved to results/narrative_flow.json).
    """
    random.seed(C.SEED)
    template = _load_template()
    data_dir = Path(C.PROCESSED_DATA_DIR)
    out_path = result_path(METRIC_NAME)

    # Resume support: load already-completed per_paper results
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})

    metadata = make_metadata(METRIC_NAME, C.TEXT_MODEL)

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")
        paper_dir = data_dir / paper

        orig_path = paper_dir / "orig.txt"
        ours_path = paper_dir / f"{C.OURS_METHOD}.txt"

        if not orig_path.exists():
            print(f"  Skipping: orig.txt not found")
            continue
        if not ours_path.exists():
            print(f"  Skipping: {C.OURS_METHOD}.txt not found")
            continue

        reference = orig_path.read_text(encoding="utf-8")
        if C.MAX_REFERENCE_CHARS and len(reference) > C.MAX_REFERENCE_CHARS:
            reference = reference[: C.MAX_REFERENCE_CHARS]

        ours_text = ours_path.read_text(encoding="utf-8")

        if paper not in per_paper:
            per_paper[paper] = {}

        for baseline in baseline_methods:
            if baseline in per_paper.get(paper, {}):
                print(f"  Skipping {baseline} (already done)")
                continue

            baseline_path = paper_dir / f"{baseline}.txt"
            if not baseline_path.exists():
                print(f"  Skipping {baseline}: text file not found")
                continue

            baseline_text = baseline_path.read_text(encoding="utf-8")
            print(f"  vs {baseline} ...", end=" ", flush=True)

            result = _compare_one(template, reference, ours_text, baseline_text)
            per_paper[paper][baseline] = result
            print(f"winner={result['winner']}")

            # Incremental save
            snapshot = {
                "metadata": metadata,
                "per_paper": per_paper,
            }
            save_incremental(out_path, snapshot)

            if C.SLEEP_BETWEEN_CALLS > 0:
                time.sleep(C.SLEEP_BETWEEN_CALLS)

    # Aggregate summaries
    per_method, overall = aggregate_pairwise_wins(per_paper, baseline_methods)
    total_papers = sum(1 for p in per_paper if per_paper[p])
    metadata["total_papers"] = total_papers

    final = {
        "metadata": metadata,
        "overall_summary": overall,
        "per_method_summary": per_method,
        "per_paper": per_paper,
    }
    save_incremental(out_path, final)
    print(f"\n[{METRIC_NAME}] Done. Results → {out_path}")
    return final
