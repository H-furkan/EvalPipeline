#!/usr/bin/env python3
"""
make_tables.py — Generate summary tables from evaluation results.

Reads JSON results from the results/ directory and produces:
  1. Console-printed tables
  2. CSV files in results/tables/
  3. LaTeX tables in results/tables/

Usage:
  python make_tables.py
  python make_tables.py --results-dir results
"""

import argparse
import json
import csv
import os
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"

# Display names for methods
METHOD_NAMES = {
    "ours_rst-4o_newv2": "Ours",
    "paper2poster_orig-4o": "Paper2Poster",
    "paper2slides": "Paper2Slides",
    "slidegen": "SlideGen",
    "pptagent_template": "PPTAgent",
    "gt": "GT",
}

PAIRWISE_METRICS = ["narrative_flow", "design_pairwise"]
SCALAR_METRICS = ["quiz_eval", "rouge_eval", "perplexity_eval", "ppt_score_eval", "general_stats_eval"]
# Note: fid_eval and content_pairwise/coherence_pairwise have been removed


def load_json(name):
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals=2, pct=False):
    if val is None:
        return "-"
    if pct:
        return f"{val * 100:.{decimals}f}"
    return f"{val:.{decimals}f}"


def method_label(m):
    return METHOD_NAMES.get(m, m)


# ─── Table 1: Pairwise Win Rates ────────────────────────────────────────────

def build_pairwise_table():
    """Ours win rate (%) against each baseline, per pairwise metric."""
    baselines = ["paper2poster_orig-4o", "paper2slides", "slidegen", "pptagent_template", "gt"]
    metrics = []
    for name in PAIRWISE_METRICS:
        data = load_json(name)
        if data:
            metrics.append((name, data))

    if not metrics:
        return None, None, None

    header = ["Metric"] + [method_label(b) for b in baselines] + ["Overall"]
    rows = []
    for name, data in metrics:
        row = [name.replace("_", " ").title()]
        for b in baselines:
            info = data.get("per_method_summary", {}).get(b, {})
            wr = info.get("ours_win_rate")
            row.append(fmt(wr, 1, pct=True))
        overall = data.get("overall_summary", {}).get("ours_overall_win_rate")
        row.append(fmt(overall, 1, pct=True))
        rows.append(row)

    return header, rows, "pairwise_winrates"


# ─── Table 2: Quiz Eval ─────────────────────────────────────────────────────

def build_quiz_table():
    data = load_json("quiz_eval")
    if not data:
        return None, None, None

    methods_order = ["ours_rst-4o_newv2", "paper2poster_orig-4o", "paper2slides",
                     "slidegen", "pptagent_template", "gt"]
    header = ["Method", "Simple (%)", "Detail (%)", "Papers"]
    rows = []
    for m in methods_order:
        info = data.get("per_method_summary", {}).get(m)
        if not info:
            continue
        rows.append([
            method_label(m),
            fmt(info["mean_simple_pct"], 1, pct=True),
            fmt(info["mean_detail_pct"], 1, pct=True),
            str(info["papers_evaluated"]),
        ])
    return header, rows, "quiz_eval"


# ─── Table 3: ROUGE-L ───────────────────────────────────────────────────────

def build_rouge_table():
    data = load_json("rouge_eval")
    if not data:
        return None, None, None

    methods_order = ["ours_rst-4o_newv2", "paper2poster_orig-4o", "paper2slides",
                     "slidegen", "pptagent_template", "gt"]
    header = ["Method", "ROUGE-L", "Papers"]
    rows = []
    for m in methods_order:
        info = data.get("per_method_summary", {}).get(m)
        if not info:
            continue
        rows.append([
            method_label(m),
            fmt(info["mean_rouge_l"], 4),
            str(info["papers_evaluated"]),
        ])
    return header, rows, "rouge_eval"


# ─── Table 4: PPT Score ─────────────────────────────────────────────────────

def build_ppt_score_table():
    data = load_json("ppt_score_eval")
    if not data:
        return None, None, None

    methods_order = ["ours_rst-4o_newv2", "paper2poster_orig-4o", "paper2slides",
                     "slidegen", "pptagent_template", "gt"]
    header = ["Method", "Content", "Style", "Logic", "Avg", "Papers"]
    rows = []
    for m in methods_order:
        info = data.get("per_method_summary", {}).get(m)
        if not info:
            continue
        avg = (info["mean_content"] + info["mean_style"] + info["mean_logic"]) / 3
        rows.append([
            method_label(m),
            fmt(info["mean_content"]),
            fmt(info["mean_style"]),
            fmt(info["mean_logic"]),
            fmt(avg),
            str(info["papers_evaluated"]),
        ])
    return header, rows, "ppt_score_eval"


# ─── Table 5: General Stats ─────────────────────────────────────────────────

def build_general_stats_table():
    data = load_json("general_stats_eval")
    if not data:
        return None, None, None

    methods_order = ["ours_rst-4o_newv2", "paper2poster_orig-4o", "paper2slides",
                     "slidegen", "pptagent_template", "gt"]
    header = ["Method", "Pages", "Characters", "Figures", "Papers"]
    rows = []
    for m in methods_order:
        info = data.get("per_method_summary", {}).get(m)
        if not info:
            continue
        rows.append([
            method_label(m),
            fmt(info["mean_pages"], 1),
            fmt(info["mean_characters"], 0),
            fmt(info["mean_figures"], 1),
            str(info["papers_evaluated"]),
        ])
    return header, rows, "general_stats"


# ─── Table 6: Combined Summary ──────────────────────────────────────────────

def build_combined_table():
    """One row per method, columns for key metrics across all evals."""
    methods_order = ["ours_rst-4o_newv2", "paper2poster_orig-4o", "paper2slides",
                     "slidegen", "pptagent_template", "gt"]

    # Collect pairwise overall win rates (only for ours vs each)
    pairwise_data = {}
    for name in PAIRWISE_METRICS:
        data = load_json(name)
        if data:
            pairwise_data[name] = data.get("overall_summary", {}).get("ours_overall_win_rate")

    quiz = load_json("quiz_eval")
    rouge = load_json("rouge_eval")
    ppt = load_json("ppt_score_eval")

    header = ["Method",
              "Quiz Simple (%)", "Quiz Detail (%)",
              "ROUGE-L",
              "PPT Content", "PPT Style", "PPT Logic"]
    rows = []
    for m in methods_order:
        row = [method_label(m)]

        # Quiz
        qi = quiz.get("per_method_summary", {}).get(m, {}) if quiz else {}
        row.append(fmt(qi.get("mean_simple_pct"), 1, pct=True))
        row.append(fmt(qi.get("mean_detail_pct"), 1, pct=True))

        # ROUGE
        ri = rouge.get("per_method_summary", {}).get(m, {}) if rouge else {}
        row.append(fmt(ri.get("mean_rouge_l"), 4))

        # PPT Score
        pi = ppt.get("per_method_summary", {}).get(m, {}) if ppt else {}
        row.append(fmt(pi.get("mean_content")))
        row.append(fmt(pi.get("mean_style")))
        row.append(fmt(pi.get("mean_logic")))

        rows.append(row)

    return header, rows, "combined_summary"


# ─── Output helpers ──────────────────────────────────────────────────────────

def print_table(title, header, rows):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Calculate column widths
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Print header
    hdr = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(hdr)
    print("-" * len(hdr))

    # Print rows — bold the best value per column (skip first col = method name)
    for row in rows:
        line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
    print()


def save_csv(name, header, rows):
    path = TABLES_DIR / f"{name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Saved: {path}")


def save_latex(name, header, rows):
    path = TABLES_DIR / f"{name}.tex"
    ncols = len(header)
    col_spec = "l" + "c" * (ncols - 1)

    lines = []
    lines.append(f"\\begin{{table}}[ht]")
    lines.append(f"\\centering")
    lines.append(f"\\caption{{{name.replace('_', ' ').title()}}}")
    lines.append(f"\\label{{tab:{name}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"\\toprule")
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\")
    lines.append(f"\\midrule")

    # Find best values per column for bolding
    numeric_cols = list(range(1, ncols))
    best_vals = {}
    for ci in numeric_cols:
        vals = []
        for row in rows:
            try:
                vals.append((float(row[ci]), row))
            except (ValueError, IndexError):
                pass
        if vals:
            best_vals[ci] = max(vals, key=lambda x: x[0])[0]

    for row in rows:
        cells = [row[0]]
        for ci in range(1, ncols):
            cell = row[ci]
            try:
                if float(cell) == best_vals.get(ci):
                    cell = f"\\textbf{{{cell}}}"
            except (ValueError, KeyError):
                pass
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append(f"\\bottomrule")
    lines.append(f"\\end{{tabular}}")
    lines.append(f"\\end{{table}}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation result tables.")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    global RESULTS_DIR, TABLES_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        TABLES_DIR = RESULTS_DIR / "tables"

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    builders = [
        ("Pairwise Win Rates (Ours Win %)", build_pairwise_table),
        ("Quiz Evaluation", build_quiz_table),
        ("ROUGE-L Scores", build_rouge_table),
        ("PPT Score (VLM-as-Judge, 1-5)", build_ppt_score_table),
        ("General Statistics", build_general_stats_table),
        ("Combined Summary", build_combined_table),
    ]

    for title, builder in builders:
        result = builder()
        if result[0] is None:
            print(f"\n[skip] {title} — no data found")
            continue
        header, rows, name = result
        print_table(title, header, rows)
        save_csv(name, header, rows)
        save_latex(name, header, rows)

    print(f"\nAll tables saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
