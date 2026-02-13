"""
metrics/design_pairwise.py — Design quality pairwise evaluation (5 attributes, image-based).

Attributes:
  1. layout        — alignment, spacing, visual hierarchy      (prompts/design_layout.txt)
  2. readability   — font sizes, contrast, crowding            (prompts/design_readability.txt)
  3. text_density  — whitespace, bullet vs paragraph balance   (prompts/design_text_density.txt)
  4. aesthetics    — color scheme, balance, polish             (prompts/design_aesthetics.txt)
  5. consistency   — template uniformity across slides         (prompts/design_consistency.txt)

Data type: images (PNG slides from IMAGES_CACHE_DIR or converted on the fly from PPTX)
Model: vision LLM
Output: results/design_pairwise.json
"""

import glob
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from llm.client import call_vision
from utils.image_utils import get_method_images, pptx_to_images, resize_images_tmp, sample_slides
from utils.result_utils import (
    aggregate_pairwise_wins,
    load_existing,
    make_metadata,
    result_path,
    save_incremental,
)

METRIC_NAME = "design_pairwise"

ATTRIBUTES = [
    {"name": "layout",       "prompt": "design_layout.txt"},
    {"name": "readability",  "prompt": "design_readability.txt"},
    {"name": "text_density", "prompt": "design_text_density.txt"},
    {"name": "aesthetics",   "prompt": "design_aesthetics.txt"},
    {"name": "consistency",  "prompt": "design_consistency.txt"},
]


def _load_templates() -> dict[str, str]:
    templates = {}
    for attr in ATTRIBUTES:
        path = Path(C.PROMPTS_DIR) / attr["prompt"]
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        templates[attr["name"]] = path.read_text(encoding="utf-8")
    return templates


def _get_images(method: str, paper_name: str) -> list[str]:
    """
    Return slide image paths, converting PPTX on-the-fly if needed.
    """
    # 1. Try cache
    cached = get_method_images(method, paper_name)
    if cached:
        return cached

    # 2. Try converting from PPTX
    pptx_files = glob.glob(
        str(Path(C.GENERATED_SAMPLES_DIR) / method / paper_name / "*.pptx")
    )
    if not pptx_files:
        return []

    out_dir = Path(C.IMAGES_CACHE_DIR) / method / paper_name
    images = pptx_to_images(pptx_files[0], out_dir)
    return images


def _render_template(template: str, n1: int, n2: int) -> str:
    """Render a design prompt template with slide counts."""
    try:
        from jinja2 import Template as J2Template
        return J2Template(template).render(method_1_count=n1, method_2_count=n2)
    except Exception:
        return template.replace("{{ method_1_count }}", str(n1)).replace(
            "{{ method_2_count }}", str(n2)
        )


def _evaluate_attribute(
    template: str,
    ours_images: list[str],
    other_images: list[str],
) -> dict:
    """
    Run one visual attribute evaluation with randomised method order.
    Images for both methods are interleaved: [method1_slides..., method2_slides...].
    Expects JSON response with 'winner' key (1 or 2).
    """
    n = min(len(ours_images), len(other_images), C.MAX_SLIDES_FOR_VISUAL)
    ours_sampled = sample_slides(ours_images, n)
    other_sampled = sample_slides(other_images, n)

    if random.random() < 0.5:
        all_images = ours_sampled + other_sampled
        method_order = "ours_first"
        prompt = _render_template(template, len(ours_sampled), len(other_sampled))
    else:
        all_images = other_sampled + ours_sampled
        method_order = "other_first"
        prompt = _render_template(template, len(other_sampled), len(ours_sampled))

    # Resize to reduce token usage
    resized = resize_images_tmp(all_images, C.IMAGE_TARGET_WIDTH)

    response = call_vision(prompt, resized, model=C.VISION_MODEL, return_json=True)

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
    Run design pairwise evaluation (5 visual attributes) for all papers and baselines.
    """
    random.seed(C.SEED)
    templates = _load_templates()
    out_path = result_path(METRIC_NAME)

    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, C.VISION_MODEL)

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

            print(f"  vs {baseline}")
            attr_results = []

            for attr in ATTRIBUTES:
                name = attr["name"]
                print(f"    [{name}] ...", end=" ", flush=True)
                r = _evaluate_attribute(templates[name], ours_images, other_images)
                r["attribute"] = name
                attr_results.append(r)
                print("ours" if r["ours_wins"] else "other")

                if C.SLEEP_BETWEEN_CALLS > 0:
                    time.sleep(C.SLEEP_BETWEEN_CALLS)

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
