"""
metrics/quiz_eval.py — Quiz-based information coverage evaluation.

Steps:
  1. Generate 50 simple + 50 detail multiple-choice questions from each source paper
     (skipped if quiz JSON files already exist in QUIZ_DATA_DIR).
  2. For each method/paper: answer both quizzes using slide text.
  3. Score: count correct answers out of 50.

Templates:
  • prompts/quiz_generate_simple.txt  — 50 high-level understanding questions
  • prompts/quiz_generate_detail.txt  — 50 detailed factual questions
  • prompts/quiz_taker_text.txt       — answer questions from slide text

Model: quiz generation + answering uses MODEL (VL model)
Output: results/quiz_eval.json
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from llm.client import call_text
from utils.result_utils import (
    get_processed_text_path,
    load_existing,
    make_metadata,
    read_processed_text,
    result_path,
    save_incremental,
)

METRIC_NAME = "quiz_eval"


def _quiz_model() -> str:
    """Return the model to use for quiz generation/answering."""
    return C.MODEL


def _load_template(name: str) -> str:
    path = Path(C.PROMPTS_DIR) / name
    if not path.exists():
        raise FileNotFoundError(f"Quiz prompt not found: {path}")
    return path.read_text(encoding="utf-8")


# ── Quiz generation ───────────────────────────────────────────────────────────

def _render_jinja(template: str, **kwargs) -> str:
    try:
        from jinja2 import Template as J2T
        return J2T(template).render(**kwargs)
    except Exception:
        text = template
        for k, v in kwargs.items():
            text = text.replace("{{ " + k + " }}", str(v))
        return text


def _clean_paper_text(text: str) -> str:
    """Remove References / Acknowledgements sections from paper markdown."""
    patterns = [
        r'^#+\s*(References|Bibliography|Works\s+Cited)',
        r'^#+\s*(Acknowledgments?|Acknowledgements?)',
    ]
    positions = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.MULTILINE | re.IGNORECASE):
            positions.append(m.start())
    if positions:
        text = text[: min(positions)]
    return text


def _generate_quiz(paper_name: str, force: bool = False) -> tuple[dict, dict] | None:
    """
    Generate (or load cached) quiz questions for a paper.
    Returns (simple_quiz, detail_quiz) or None on failure.
    """
    quiz_dir = Path(C.QUIZ_DATA_DIR) / paper_name
    quiz_dir.mkdir(parents=True, exist_ok=True)

    simple_path = quiz_dir / "quiz_simple.json"
    detail_path = quiz_dir / "quiz_detail.json"

    # Load cached
    if simple_path.exists() and detail_path.exists() and not force:
        simple = json.load(open(simple_path, encoding="utf-8"))
        detail = json.load(open(detail_path, encoding="utf-8"))
        return simple, detail

    # Need source paper text
    source_md = Path(C.BENCHMARK_DATA_DIR) / paper_name / "source.md"
    orig_path = get_processed_text_path(paper_name, "orig")

    if source_md.exists():
        paper_text = source_md.read_text(encoding="utf-8")
    elif orig_path is not None:
        paper_text = orig_path.read_text(encoding="utf-8")
    else:
        print(f"  [quiz] No source text found for {paper_name}")
        return None

    paper_text = _clean_paper_text(paper_text)

    simple_tmpl = _load_template("quiz_generate_simple.txt")
    detail_tmpl = _load_template("quiz_generate_detail.txt")

    if not simple_path.exists() or force:
        print(f"  [quiz] Generating simple quiz for {paper_name} ...")
        prompt = _render_jinja(simple_tmpl, document_markdown=paper_text)
        resp = call_text(prompt, model=_quiz_model(), max_tokens=4096, return_json=True)
        if isinstance(resp, dict):
            json.dump(resp, open(simple_path, "w", encoding="utf-8"), indent=4)
            simple = resp
        else:
            print(f"  [quiz] Warning: simple quiz generation failed for {paper_name}")
            return None
    else:
        simple = json.load(open(simple_path, encoding="utf-8"))

    if not detail_path.exists() or force:
        print(f"  [quiz] Generating detail quiz for {paper_name} ...")
        prompt = _render_jinja(detail_tmpl, document_markdown=paper_text)
        resp = call_text(prompt, model=_quiz_model(), max_tokens=4096, return_json=True)
        if isinstance(resp, dict):
            json.dump(resp, open(detail_path, "w", encoding="utf-8"), indent=4)
            detail = resp
        else:
            print(f"  [quiz] Warning: detail quiz generation failed for {paper_name}")
            return None
    else:
        detail = json.load(open(detail_path, encoding="utf-8"))

    return simple, detail


# ── Quiz taking ───────────────────────────────────────────────────────────────

def _take_quiz(slide_text: str, quiz: dict, quiz_type: str) -> tuple[int, int]:
    """
    Answer a quiz using slide_text.
    Returns (score, total_questions).
    """
    taker_tmpl = _load_template("quiz_taker_text.txt")

    # Strip answer/aspect before sending to model
    questions_for_model = {}
    solutions: dict[str, str] = {}
    for q, qdata in quiz.items():
        solutions[q] = str(qdata.get("answer", "A"))[:1].upper()
        clean = {k: v for k, v in qdata.items() if k not in ("answer", "aspect")}
        questions_for_model[q] = clean

    prompt = _render_jinja(
        taker_tmpl,
        slide_text=slide_text,
        Question_dict=questions_for_model,
    )
    resp = call_text(prompt, model=_quiz_model(), max_tokens=2048, return_json=True)

    if not isinstance(resp, dict):
        return 0, len(solutions)

    score = sum(
        1 for q, sol in solutions.items()
        if str(resp.get(q, "")).strip().upper()[:1] == sol
    )
    return score, len(solutions)


def _get_slide_text(method: str, paper_name: str) -> str | None:
    text = read_processed_text(paper_name, method)
    if text is not None:
        return text

    # Fallback: paper2slides uses slide_text.txt
    alt = Path(C.GENERATED_SAMPLES_DIR) / method / paper_name / "slide_text.txt"
    if alt.exists():
        return alt.read_text(encoding="utf-8")
    return None


# ── Public entry point ────────────────────────────────────────────────────────

def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Run quiz-based evaluation for all papers and methods (ours + baselines).
    """
    out_path = result_path(METRIC_NAME)
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, _quiz_model())

    all_methods = [C.OURS_METHOD] + baseline_methods

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        quiz_result = _generate_quiz(paper)
        if quiz_result is None:
            print(f"  Skipping: quiz generation failed")
            continue
        simple_quiz, detail_quiz = quiz_result

        if paper not in per_paper:
            per_paper[paper] = {}

        for method in all_methods:
            if method in per_paper.get(paper, {}):
                entry = per_paper[paper][method]
                if entry.get("simple_score", 0) > 0 or entry.get("detail_score", 0) > 0:
                    print(f"  Skipping {method} (already done)")
                    continue

            slide_text = _get_slide_text(method, paper)
            if slide_text is None:
                print(f"  Skipping {method}: slide text not found")
                continue

            print(f"  Evaluating {method} ...", end=" ", flush=True)

            simple_score, simple_total = _take_quiz(slide_text, simple_quiz, "simple")
            detail_score, detail_total = _take_quiz(slide_text, detail_quiz, "detail")

            per_paper[paper][method] = {
                "simple_score": simple_score,
                "simple_total": simple_total,
                "detail_score": detail_score,
                "detail_total": detail_total,
                "simple_pct": simple_score / simple_total if simple_total > 0 else 0.0,
                "detail_pct": detail_score / detail_total if detail_total > 0 else 0.0,
            }
            print(f"simple={simple_score}/{simple_total} detail={detail_score}/{detail_total}")

            save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

            if C.SLEEP_BETWEEN_CALLS > 0:
                time.sleep(C.SLEEP_BETWEEN_CALLS)

    # Per-method summary (mean scores across papers)
    per_method: dict[str, dict] = {}
    for method in all_methods:
        simple_scores, detail_scores = [], []
        for paper_results in per_paper.values():
            if method in paper_results:
                simple_scores.append(paper_results[method].get("simple_pct", 0))
                detail_scores.append(paper_results[method].get("detail_pct", 0))
        per_method[method] = {
            "mean_simple_pct": sum(simple_scores) / len(simple_scores) if simple_scores else 0.0,
            "mean_detail_pct": sum(detail_scores) / len(detail_scores) if detail_scores else 0.0,
            "papers_evaluated": len(simple_scores),
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
