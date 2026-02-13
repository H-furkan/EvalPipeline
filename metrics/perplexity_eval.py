"""
metrics/perplexity_eval.py — Perplexity (PPL) evaluation.

Measures language fluency of generated slide text using a causal language model.
Lower perplexity = more fluent / natural language.

Model: meta-llama/Meta-Llama-3-8B (or any causal LM set in constants.PPL_MODEL_PATH)
Dependencies: pip install transformers torch
Output: results/perplexity_eval.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C
from utils.result_utils import (
    load_existing,
    make_metadata,
    result_path,
    save_incremental,
)

METRIC_NAME = "perplexity_eval"


def _load_model():
    """Load and cache the PPL model + tokenizer. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = C.PPL_MODEL_PATH
    print(f"  [ppl] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype="auto",
    )
    model.eval()
    print(f"  [ppl] Model loaded on {device}")
    return model, tokenizer


def _compute_ppl(text: str, model, tokenizer) -> float:
    """Compute mean perplexity over all sentences/slides in text."""
    import torch

    # Split into paragraphs/slides to avoid exceeding max token length
    segments = [s.strip() for s in text.split("\n\n") if s.strip()]
    if not segments:
        segments = [text]

    ppls = []
    for segment in segments:
        try:
            inputs = tokenizer(segment, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                ppls.append(torch.exp(loss).item())
        except Exception as e:
            print(f"    [ppl] Segment error: {e}")

    return sum(ppls) / len(ppls) if ppls else float("inf")


def _get_text(paper_name: str, method: str) -> str | None:
    path = Path(C.PROCESSED_DATA_DIR) / paper_name / f"{method}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Compute perplexity for all methods (ours + baselines) on all papers.

    The model is loaded once and reused for all evaluations to avoid
    repeatedly loading a large model.
    """
    out_path = result_path(METRIC_NAME)
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, C.PPL_MODEL_PATH)

    all_methods = [C.OURS_METHOD] + baseline_methods

    # Check if there is any work to do before loading the heavy model
    needs_eval = False
    for paper in papers:
        for method in all_methods:
            if method not in per_paper.get(paper, {}):
                needs_eval = True
                break
        if needs_eval:
            break

    if not needs_eval:
        print(f"[{METRIC_NAME}] All results already computed. Skipping model load.")
        return existing

    # Load model once
    try:
        model, tokenizer = _load_model()
    except Exception as e:
        print(f"[{METRIC_NAME}] Failed to load PPL model: {e}")
        print("  Skipping perplexity evaluation.")
        return {}

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        if paper not in per_paper:
            per_paper[paper] = {}

        for method in all_methods:
            if method in per_paper.get(paper, {}):
                print(f"  Skipping {method} (already done)")
                continue

            text = _get_text(paper, method)
            if text is None:
                print(f"  Skipping {method}: text not found")
                continue

            print(f"  Computing PPL for {method} ...", end=" ", flush=True)
            ppl = _compute_ppl(text, model, tokenizer)
            per_paper[paper][method] = {"ppl": ppl}
            print(f"PPL={ppl:.2f}")

        save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

    # Per-method summary
    per_method: dict[str, dict] = {}
    for method in all_methods:
        scores = [
            per_paper[p][method]["ppl"]
            for p in per_paper
            if method in per_paper[p] and per_paper[p][method]["ppl"] != float("inf")
        ]
        per_method[method] = {
            "mean_ppl": sum(scores) / len(scores) if scores else float("inf"),
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
