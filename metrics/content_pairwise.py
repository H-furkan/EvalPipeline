"""
metrics/content_pairwise.py — Content quality pairwise evaluation (4 attributes).

Attributes:
  1. text_quality   — grammar, clarity, fluency               (prompts/content_text_quality.txt)
  2. structure      — titles, bullet hierarchy                 (prompts/content_structure.txt)
  3. organization   — within-slide logic, cross-slide flow     (prompts/content_organization.txt)
  4. conciseness    — conciseness, audience-appropriateness    (prompts/content_conciseness.txt)

Data type: text (processed markdown from PPTX/PDF presentations)
Model: VL model
Output: results/content_pairwise.json
"""

import random
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
    read_processed_text,
    result_path,
    save_incremental,
)

METRIC_NAME = "content_pairwise"

ATTRIBUTES = [
    {"name": "text_quality",  "prompt": "content_text_quality.txt"},
    {"name": "structure",     "prompt": "content_structure.txt"},
    {"name": "organization",  "prompt": "content_organization.txt"},
    {"name": "conciseness",   "prompt": "content_conciseness.txt"},
]


def _load_templates() -> dict[str, str]:
    templates = {}
    for attr in ATTRIBUTES:
        path = Path(C.PROMPTS_DIR) / attr["prompt"]
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        templates[attr["name"]] = path.read_text(encoding="utf-8")
    return templates


def _get_slide_text(method: str, paper_name: str) -> str | None:
    """Return slide text for a given method/paper from the processed data directory."""
    return read_processed_text(paper_name, method)


def _evaluate_attribute(template: str, ours_text: str, other_text: str) -> dict:
    """
    Run a single attribute evaluation with randomised method order.
    Expects the model to return JSON with a 'winner' key (1 or 2).
    """
    if random.random() < 0.5:
        rendered = template.replace("{{ method_1_content }}", ours_text).replace(
            "{{ method_2_content }}", other_text
        )
        method_order = "ours_first"
    else:
        rendered = template.replace("{{ method_1_content }}", other_text).replace(
            "{{ method_2_content }}", ours_text
        )
        method_order = "other_first"

    # Jinja2-style rendering fallback
    try:
        from jinja2 import Template as J2Template
        if "method_order" == "ours_first":
            rendered = J2Template(template).render(
                method_1_content=ours_text, method_2_content=other_text
            )
        else:
            rendered = J2Template(template).render(
                method_1_content=other_text, method_2_content=ours_text
            )
    except Exception:
        pass

    response = call_text(rendered, model=C.MODEL, return_json=True)

    if not isinstance(response, dict) or "winner" not in response:
        return {"ours_wins": False, "winner": "unknown", "method_order": method_order, "raw": response}

    raw_winner = response["winner"]
    if method_order == "ours_first":
        ours_wins = raw_winner == 1
    else:
        ours_wins = raw_winner == 2

    return {
        "ours_wins": bool(ours_wins),
        "winner": "ours" if ours_wins else "other",
        "method_order": method_order,
        "raw_winner": raw_winner,
        "reasoning": response.get("reasoning", ""),
        "method_1_analysis": response.get("method_1_analysis", ""),
        "method_2_analysis": response.get("method_2_analysis", ""),
    }


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Run content pairwise evaluation (4 attributes) for all papers and baselines.
    """
    random.seed(C.SEED)
    templates = _load_templates()
    out_path = result_path(METRIC_NAME)

    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, C.MODEL)

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        ours_text = _get_slide_text(C.OURS_METHOD, paper)
        if ours_text is None:
            print(f"  Skipping: text not found for {C.OURS_METHOD}")
            continue

        if paper not in per_paper:
            per_paper[paper] = {}

        for baseline in baseline_methods:
            if baseline in per_paper.get(paper, {}):
                print(f"  Skipping {baseline} (already done)")
                continue

            other_text = _get_slide_text(baseline, paper)
            if other_text is None:
                print(f"  Skipping {baseline}: text not found")
                continue

            print(f"  vs {baseline}")
            attr_results = []

            for attr in ATTRIBUTES:
                name = attr["name"]
                print(f"    [{name}] ...", end=" ", flush=True)
                r = _evaluate_attribute(templates[name], ours_text, other_text)
                r["attribute"] = name
                attr_results.append(r)
                print("ours" if r["ours_wins"] else "other")

                if C.SLEEP_BETWEEN_CALLS > 0:
                    time.sleep(C.SLEEP_BETWEEN_CALLS)

            # Compute aggregate for this baseline
            ours_wins_count = sum(1 for r in attr_results if r["ours_wins"])
            total_attrs = len(attr_results)
            winner = "ours" if ours_wins_count > total_attrs / 2 else "baseline"

            per_paper[paper][baseline] = {
                "winner": winner,
                "ours_wins": ours_wins_count,
                "total_attributes": total_attrs,
                "attributes": attr_results,
            }

            save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

    per_method, overall = aggregate_pairwise_wins(per_paper, baseline_methods)
    metadata["total_papers"] = sum(1 for p in per_paper if per_paper[p])

    final = {
        "metadata": metadata,
        "overall_summary": overall,
        "per_method_summary": per_method,
        "per_paper": per_paper,
    }
    save_incremental(out_path, final)
    print(f"\n[{METRIC_NAME}] Done. Results → {out_path}")
    return final
