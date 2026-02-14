#!/usr/bin/env python3
"""
pipeline.py — One-shot modular evaluation pipeline.

Usage:
  # Run all metrics on all papers (reads everything from constants.py)
  python pipeline.py

  # Skip the PDF/PPTX → text conversion step
  python pipeline.py --skip-data-prep

  # Run only specific metrics
  python pipeline.py --metrics narrative_flow rouge_eval

  # Evaluate only specific papers
  python pipeline.py --papers attn camera graph

  # Override the LLM backend at runtime
  python pipeline.py --backend openai

  # Re-convert data even if processed_data/ already exists
  python pipeline.py --force-conversion

Edit constants.py to permanently change any path, model, or threshold.
"""

import argparse
import importlib
import random
import sys
import time
from pathlib import Path

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import constants as C
from utils.result_utils import print_pairwise_summary, print_scalar_summary

# Metrics that produce pairwise win-rate results
PAIRWISE_METRICS = {"narrative_flow", "content_pairwise", "design_pairwise", "coherence_pairwise"}
# Metrics that produce scalar / per-method averages
SCALAR_METRICS = {"quiz_eval", "rouge_eval", "perplexity_eval", "ppt_score_eval", "fid_eval", "general_stats_eval"}


# ── Discovery ─────────────────────────────────────────────────────────────────

def _discover_papers() -> list[str]:
    """Return sorted list of papers available in GENERATED_SAMPLES_DIR / OURS_METHOD."""
    ours_dir = Path(C.GENERATED_SAMPLES_DIR) / C.OURS_METHOD
    if not ours_dir.exists():
        # Fallback: scan PROCESSED_DATA_DIR
        pd = Path(C.PROCESSED_DATA_DIR)
        if pd.exists():
            return sorted(d.name for d in pd.iterdir() if d.is_dir())
        print(f"[pipeline] Warning: cannot discover papers – {ours_dir} not found.")
        return []
    return sorted(d.name for d in ours_dir.iterdir() if d.is_dir())


def _select_papers(papers: list[str], num: int) -> list[str]:
    """Randomly sample *num* papers; return all if num <= 0."""
    if num <= 0 or num >= len(papers):
        return papers
    random.seed(C.SEED)
    return sorted(random.sample(papers, num))


# ── Data preparation ──────────────────────────────────────────────────────────

def run_data_prep(papers: list[str], force: bool = False) -> None:
    from data_prep.convert_to_text import run_conversion
    all_methods = [C.OURS_METHOD] + C.BASELINE_METHODS
    run_conversion(papers=papers, methods=all_methods, force=force)


# ── Metric runner ─────────────────────────────────────────────────────────────

def run_metric(name: str, papers: list[str], baseline_methods: list[str]) -> dict:
    """Dynamically import and run a metric module."""
    module = importlib.import_module(f"metrics.{name}")
    print(f"\n{'='*70}")
    print(f"RUNNING METRIC: {name}")
    print(f"{'='*70}")
    result = module.run(papers, baseline_methods)
    return result


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_final_summary(all_results: dict[str, dict]) -> None:
    print(f"\n{'#'*70}")
    print("FINAL EVALUATION SUMMARY")
    print(f"{'#'*70}")
    for metric_name, result in all_results.items():
        if not result:
            print(f"\n[{metric_name}] No results.")
            continue
        if metric_name in PAIRWISE_METRICS:
            print_pairwise_summary(metric_name, result)
        else:
            print_scalar_summary(metric_name, result)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot modular evaluation pipeline for presentation evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Metrics to run. If omitted, runs all metrics in constants.ENABLED_METRICS.\n"
            "Available: narrative_flow, content_pairwise, design_pairwise, "
            "coherence_pairwise, quiz_eval, rouge_eval, perplexity_eval, ppt_score_eval, "
            "fid_eval, general_stats_eval"
        ),
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        default=None,
        metavar="PAPER",
        help="Paper names to evaluate. Defaults to all discovered papers.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=None,
        metavar="METHOD",
        help=f"Baseline methods to compare against. Defaults to constants.BASELINE_METHODS.",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "openai"],
        default=None,
        help="LLM backend override. Defaults to constants.LLM_BACKEND.",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        default=False,
        help="Skip PDF/PPTX → text conversion (use if processed_data/ is up-to-date).",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        default=False,
        help="Re-convert all files even if they already exist in processed_data/.",
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N papers. -1 or omitted = all papers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="MODEL",
        help=(
            "VL model alias or full HuggingFace ID. "
            "Aliases: " + ", ".join(C.SUPPORTED_MODELS.keys())
        ),
    )
    parser.add_argument(
        "--vllm-api-base",
        type=str,
        default=None,
        metavar="URL",
        help="Override vLLM API base URL (e.g. http://localhost:7001/v1).",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Apply runtime overrides to constants (in-memory only)
    if args.backend:
        C.LLM_BACKEND = args.backend

    # Apply model override
    if args.model:
        C.MODEL = C.SUPPORTED_MODELS.get(args.model, args.model)
    if args.vllm_api_base:
        C.VLLM_API_BASE = args.vllm_api_base
        C.VLLM_HEALTH_URL = args.vllm_api_base.rsplit("/v1", 1)[0] + "/health"

    # Resolve metrics list
    metrics_to_run = args.metrics if args.metrics else list(C.ENABLED_METRICS)

    # Resolve baseline methods
    baseline_methods = args.baselines if args.baselines else list(C.BASELINE_METHODS)

    # Resolve papers
    if args.papers:
        papers = args.papers
    else:
        papers = _discover_papers()

    # Apply paper limit
    num_papers = args.num_papers if args.num_papers is not None else C.NUM_PAPERS
    papers = _select_papers(papers, num_papers)

    if not papers:
        print("[pipeline] No papers found. Check GENERATED_SAMPLES_DIR in constants.py.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("EVALUATION PIPELINE")
    print(f"{'='*70}")
    print(f"Backend       : {C.LLM_BACKEND}")
    print(f"Model         : {C.MODEL}")
    print(f"Ours method   : {C.OURS_METHOD}")
    print(f"Baselines     : {baseline_methods}")
    print(f"Papers        : {len(papers)}")
    print(f"Metrics       : {metrics_to_run}")
    print(f"Results dir   : {C.RESULTS_DIR}")
    Path(C.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data preparation ──────────────────────────────────────────────
    skip_prep = args.skip_data_prep or C.SKIP_DATA_PREP
    if not skip_prep:
        print(f"\n{'='*70}")
        print("STEP 1: Data Preparation (PDF/PPTX → text)")
        print(f"{'='*70}")
        run_data_prep(papers, force=args.force_conversion)
    else:
        print("\n[pipeline] Skipping data preparation.")

    # ── Step 2: Run metrics ───────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    t0 = time.time()

    for metric_name in metrics_to_run:
        try:
            result = run_metric(metric_name, papers, baseline_methods)
            all_results[metric_name] = result
        except FileNotFoundError as e:
            print(f"\n[pipeline] ERROR in {metric_name}: {e}")
            print("  Check that all prompt files exist in PROMPTS_DIR.")
            all_results[metric_name] = {}
        except Exception as e:
            print(f"\n[pipeline] ERROR in {metric_name}: {type(e).__name__}: {e}")
            all_results[metric_name] = {}

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_final_summary(all_results)

    print(f"\n{'='*70}")
    print(f"All done in {elapsed:.1f}s. Results saved to: {C.RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
