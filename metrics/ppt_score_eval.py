"""
metrics/ppt_score_eval.py — VLM-as-judge per-slide scoring.

Scores each presentation on three dimensions (1-5 scale each):
  - content score  — information quality per slide   (prompts/ppteval_content.txt)
  - style score    — visual design per slide          (prompts/ppteval_style.txt)
  - logic score    — overall coherence of the deck   (prompts/ppteval_coherence.txt)

The logic score is computed at the presentation level using the processed slide text.
Content and style scores are computed per-slide using slide images
(converted from PPTX/PDF on-the-fly if needed).

Model: VL model (handles both vision and text tasks)
Output: results/ppt_score_eval.json
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from llm.client import call_text, call_vision
from utils.image_utils import find_and_convert_images, resize_images_tmp, sample_slides
from utils.result_utils import (
    load_existing,
    make_metadata,
    read_processed_text,
    result_path,
    save_incremental,
)

METRIC_NAME = "ppt_score_eval"


def _render(template: str, **kwargs) -> str:
    try:
        from jinja2 import Template as J2T
        return J2T(template).render(**kwargs)
    except Exception:
        text = template
        for k, v in kwargs.items():
            text = text.replace("{{ " + k + " }}", str(v)).replace("{{" + k + "}}", str(v))
        return text


def _load_prompt(filename: str) -> str:
    path = Path(C.PROMPTS_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


# ── Per-slide scoring ─────────────────────────────────────────────────────────

def _score_slides(images: list[str], content_tmpl: str, style_tmpl: str) -> dict:
    """
    Score a sample of slides for content and style.
    Returns {"content": mean_score, "style": mean_score}.
    """
    describe_content_tmpl = _load_prompt("ppteval_describe_content.txt")
    describe_style_tmpl = _load_prompt("ppteval_describe_style.txt")

    sampled = sample_slides(images, C.MAX_SLIDES_FOR_VISUAL)
    resized = resize_images_tmp(sampled, C.IMAGE_TARGET_WIDTH)

    content_scores = []
    style_scores = []

    for img_path in resized:
        # Describe content
        content_descr = call_vision(describe_content_tmpl, [img_path], model=C.MODEL)
        if content_descr:
            score_prompt = _render(content_tmpl, descr=content_descr)
            resp = call_text(score_prompt, model=C.MODEL, return_json=True)
            if isinstance(resp, dict) and "score" in resp:
                content_scores.append(float(resp["score"]))

        # Describe style
        style_descr = call_vision(describe_style_tmpl, [img_path], model=C.MODEL)
        if style_descr:
            score_prompt = _render(style_tmpl, descr=style_descr)
            resp = call_text(score_prompt, model=C.MODEL, return_json=True)
            if isinstance(resp, dict) and "score" in resp:
                style_scores.append(float(resp["score"]))

        if C.SLEEP_BETWEEN_CALLS > 0:
            time.sleep(C.SLEEP_BETWEEN_CALLS)

    return {
        "content": sum(content_scores) / len(content_scores) if content_scores else 0.0,
        "style": sum(style_scores) / len(style_scores) if style_scores else 0.0,
        "slides_scored": len(content_scores),
    }


# ── Presentation-level logic score ────────────────────────────────────────────

def _score_logic(slide_text: str, coherence_tmpl: str) -> dict:
    """Score overall presentation coherence using processed slide text."""
    prompt = _render(coherence_tmpl, presentation=slide_text)
    resp = call_text(prompt, model=C.MODEL, return_json=True)
    if isinstance(resp, dict) and "score" in resp:
        return resp
    return {"score": 0, "reason": str(resp)}


# ── Public entry point ────────────────────────────────────────────────────────

def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Run per-slide content / style scoring and presentation-level logic scoring
    for all methods (ours + baselines) across all papers.
    """
    out_path = result_path(METRIC_NAME)
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, C.MODEL)

    content_tmpl = _load_prompt("ppteval_content.txt")
    style_tmpl = _load_prompt("ppteval_style.txt")
    coherence_tmpl = _load_prompt("ppteval_coherence.txt")

    all_methods = [C.OURS_METHOD] + baseline_methods

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        if paper not in per_paper:
            per_paper[paper] = {}

        for method in all_methods:
            if method in per_paper.get(paper, {}):
                print(f"  Skipping {method} (already done)")
                continue

            print(f"  Evaluating {method} ...")
            result: dict = {}

            # Per-slide scoring (needs images)
            images = find_and_convert_images(method, paper)
            if images:
                slide_scores = _score_slides(images, content_tmpl, style_tmpl)
                result.update(slide_scores)
            else:
                print(f"    No images found; skipping slide scoring")
                result["content"] = 0.0
                result["style"] = 0.0
                result["slides_scored"] = 0

            # Logic score (uses processed slide text)
            slide_text = read_processed_text(paper, method)
            if slide_text:
                logic = _score_logic(slide_text, coherence_tmpl)
                result["logic"] = float(logic.get("score", 0)) if isinstance(logic, dict) else float(logic)
                result["logic_reason"] = logic.get("reason", "") if isinstance(logic, dict) else ""
            else:
                print(f"    No slide text found; skipping logic scoring")
                result["logic"] = 0.0
                result["logic_reason"] = "not available"

            per_paper[paper][method] = result
            print(
                f"    content={result['content']:.2f} "
                f"style={result['style']:.2f} "
                f"logic={result.get('logic', 0.0)}"
            )

            save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

    # Per-method summary
    per_method: dict[str, dict] = {}
    for method in all_methods:
        content_list, style_list, logic_list = [], [], []
        for paper_results in per_paper.values():
            if method in paper_results:
                r = paper_results[method]
                content_list.append(r.get("content", 0.0))
                style_list.append(r.get("style", 0.0))
                logic_score = r.get("logic", 0.0)
                if isinstance(logic_score, dict):
                    # Backward compat: handle old dict format from prior runs
                    logic_list.append(float(logic_score.get("score", 0)))
                else:
                    logic_list.append(float(logic_score))
        per_method[method] = {
            "mean_content": sum(content_list) / len(content_list) if content_list else 0.0,
            "mean_style": sum(style_list) / len(style_list) if style_list else 0.0,
            "mean_logic": sum(logic_list) / len(logic_list) if logic_list else 0.0,
            "papers_evaluated": len(content_list),
        }

    metadata["total_papers"] = len(per_paper)
    final = {
        "metadata": metadata,
        "per_method_summary": per_method,
        "per_paper": per_paper,
    }
    save_incremental(out_path, final)
    print(f"\n[{METRIC_NAME}] Done. Results -> {out_path}")
    return final
