"""
metrics/design_pairwise.py — Design quality pairwise evaluation (single-prompt, image-based).

Compares OURS_METHOD against each baseline on overall visual design quality
using a single unified prompt that covers layout, readability, text density,
aesthetics, and consistency.

Data type: images (PNG slides from IMAGES_CACHE_DIR or converted on the fly from PPTX/PDF)
Model: VL model
Prompt: prompts/design_pairwise.txt
Output: results/design_pairwise.json
"""

import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from llm.client import call_vision
from utils.image_utils import find_and_convert_images, resize_images_tmp, sample_slides
from utils.result_utils import (
    aggregate_pairwise_wins,
    load_existing,
    make_metadata,
    result_path,
    save_incremental,
)

METRIC_NAME = "design_pairwise"
PROMPT_FILE = Path(C.PROMPTS_DIR) / "design_pairwise.txt"


def _load_template() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def _get_images(method: str, paper_name: str) -> list[str]:
    """Return slide image paths, converting PPTX/PDF on-the-fly if needed."""
    return find_and_convert_images(method, paper_name)


def _render_template(template: str, n1: int, n2: int) -> str:
    """Render the design prompt template with slide counts."""
    try:
        from jinja2 import Template as J2Template
        return J2Template(template).render(method_1_count=n1, method_2_count=n2)
    except Exception:
        return template.replace("{{ method_1_count }}", str(n1)).replace(
            "{{ method_2_count }}", str(n2)
        )


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
            chosen = "UNKNOWN"
            for ch in upper:
                if ch in ("A", "B"):
                    chosen = ch
                    break

    winner = a_source if chosen == "A" else (b_source if chosen == "B" else "unknown")
    return winner, reasoning


def _compare_one(
    template: str,
    ours_images: list[str],
    other_images: list[str],
) -> dict:
    """
    Run a single design comparison with randomised A/B assignment.
    Images for both methods are concatenated: [Option A slides..., Option B slides...].
    Expects text response with 'Answer: A' or 'Answer: B'.
    """
    n = min(len(ours_images), len(other_images), C.MAX_SLIDES_FOR_VISUAL)
    if n == 0:
        return {"winner": "unknown", "method_order": "none", "reasoning": "no images"}

    ours_sampled = sample_slides(ours_images, n)
    other_sampled = sample_slides(other_images, n)

    if random.random() < 0.5:
        all_images = ours_sampled + other_sampled
        a_source, b_source = "ours", "baseline"
        prompt = _render_template(template, len(ours_sampled), len(other_sampled))
    else:
        all_images = other_sampled + ours_sampled
        a_source, b_source = "baseline", "ours"
        prompt = _render_template(template, len(other_sampled), len(ours_sampled))

    # Resize to reduce token usage
    resized = resize_images_tmp(all_images, C.IMAGE_TARGET_WIDTH)

    response = call_vision(prompt, resized, model=C.MODEL)

    winner, reasoning = _parse_winner(response, a_source, b_source)

    return {
        "assignment": {"A": a_source, "B": b_source},
        "winner": winner,
        "reasoning": reasoning,
        "raw_response": response or "ERROR",
    }


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Run design pairwise evaluation for all papers and baselines.
    Uses a single unified prompt covering all visual design criteria.
    """
    random.seed(C.SEED)
    template = _load_template()
    out_path = result_path(METRIC_NAME)

    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, C.MODEL)

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        ours_images = _get_images(C.OURS_METHOD, paper)
        if not ours_images:
            print(f"  Skipping: no images for {C.OURS_METHOD}")
            continue

        if paper not in per_paper:
            per_paper[paper] = {}

        for baseline in baseline_methods:
            if baseline in per_paper.get(paper, {}):
                print(f"  Skipping {baseline} (already done)")
                continue

            other_images = _get_images(baseline, paper)
            if not other_images:
                print(f"  Skipping {baseline}: no images")
                continue

            print(f"  vs {baseline} ...", end=" ", flush=True)
            result = _compare_one(template, ours_images, other_images)
            per_paper[paper][baseline] = result
            print(f"winner={result['winner']}")

            save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

            if C.SLEEP_BETWEEN_CALLS > 0:
                time.sleep(C.SLEEP_BETWEEN_CALLS)

    per_method, overall = aggregate_pairwise_wins(per_paper, baseline_methods)
    metadata["total_papers"] = sum(1 for p in per_paper if per_paper[p])

    final = {
        "metadata": metadata,
        "overall_summary": overall,
        "per_method_summary": per_method,
        "per_paper": per_paper,
    }
    save_incremental(out_path, final)
    print(f"\n[{METRIC_NAME}] Done. Results -> {out_path}")
    return final
