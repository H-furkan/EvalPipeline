"""
metrics/general_stats_eval.py — General presentation statistics.

Computes structural statistics for each presentation:
  - pages:      Number of slides
  - characters: Total character count across all slide text
  - figures:    Number of images/figures in the presentation

These are absolute per-method metrics (not pairwise).
No LLM required — works directly from PPTX/PDF files.

Dependencies: pip install python-pptx pymupdf
Output: results/general_stats_eval.json
"""

import re
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

METRIC_NAME = "general_stats_eval"


def _count_pptx_stats(pptx_path: Path) -> dict | None:
    """
    Extract page count, character count, and figure count from a PPTX file.

    Returns dict with keys: pages, characters, figures
    """
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        prs = Presentation(str(pptx_path))
        pages = len(prs.slides)
        characters = 0
        figures = 0

        for slide in prs.slides:
            for shape in slide.shapes:
                # Count text characters
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            characters += len(run.text)

                # Count figures/images
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    figures += 1
                elif shape.shape_type == MSO_SHAPE_TYPE.PLACEHOLDER:
                    if hasattr(shape, "image"):
                        figures += 1

        return {"pages": pages, "characters": characters, "figures": figures}
    except ImportError:
        print(f"  [stats] python-pptx not installed. Run: pip install python-pptx")
        return None
    except Exception as e:
        print(f"  [stats] Error reading {pptx_path}: {e}")
        return None


def _count_pdf_stats(pdf_path: Path) -> dict | None:
    """
    Extract page count, character count, and figure count from a PDF file.

    Returns dict with keys: pages, characters, figures
    """
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf_path))
        pages = len(doc)
        characters = 0
        figures = 0

        for page in doc:
            text = page.get_text()
            characters += len(text)
            # Count images on each page
            figures += len(page.get_images(full=True))

        doc.close()
        return {"pages": pages, "characters": characters, "figures": figures}
    except ImportError:
        print(f"  [stats] pymupdf not installed. Run: pip install pymupdf")
        return None
    except Exception as e:
        print(f"  [stats] Error reading {pdf_path}: {e}")
        return None


def _count_text_stats(paper_name: str, method: str) -> dict | None:
    """
    Fallback: compute stats from converted text files when PPTX/PDF is not available.
    """
    text = read_processed_text(paper_name, method)
    if text is None:
        return None
    # Count slides by "## Slide N" headers
    slides = re.findall(r"^## Slide \d+", text, re.MULTILINE)
    pages = len(slides) if slides else 1

    return {
        "pages": pages,
        "characters": len(text),
        "figures": 0,  # Can't count from text
    }


def _find_presentation(method: str, paper_name: str) -> Path | None:
    """Find a PPTX or PDF file for a method/paper combination."""
    method_dir = Path(C.GENERATED_SAMPLES_DIR) / method / paper_name
    if not method_dir.exists():
        return None
    # Try PPTX first
    pptx_files = list(method_dir.glob("*.pptx"))
    if pptx_files:
        return pptx_files[0]
    # Try PDF
    pdf_files = list(method_dir.glob("*.pdf"))
    if pdf_files:
        return pdf_files[0]
    return None


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Compute general statistics for all methods (ours + baselines) on all papers.
    """
    out_path = result_path(METRIC_NAME)
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, "none (no LLM)")

    all_methods = [C.OURS_METHOD] + baseline_methods

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        if paper not in per_paper:
            per_paper[paper] = {}

        for method in all_methods:
            if method in per_paper.get(paper, {}):
                print(f"  Skipping {method} (already done)")
                continue

            # Try PPTX/PDF first
            pres_path = _find_presentation(method, paper)
            stats = None
            if pres_path:
                if pres_path.suffix.lower() == ".pptx":
                    stats = _count_pptx_stats(pres_path)
                else:  # .pdf
                    stats = _count_pdf_stats(pres_path)

            # Fallback to text stats
            if stats is None:
                stats = _count_text_stats(paper, method)

            if stats:
                per_paper[paper][method] = stats
                print(
                    f"  {method}: pages={stats['pages']}, "
                    f"chars={stats['characters']}, figures={stats['figures']}"
                )
            else:
                print(f"  Skipping {method}: no PPTX, PDF, or text found")

        save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

    # Per-method summary
    per_method: dict[str, dict] = {}
    for method in all_methods:
        pages_list = []
        chars_list = []
        figs_list = []
        for p in per_paper:
            if method in per_paper[p]:
                s = per_paper[p][method]
                pages_list.append(s["pages"])
                chars_list.append(s["characters"])
                figs_list.append(s["figures"])

        n = len(pages_list)
        per_method[method] = {
            "mean_pages": sum(pages_list) / n if n else 0.0,
            "mean_characters": sum(chars_list) / n if n else 0.0,
            "mean_figures": sum(figs_list) / n if n else 0.0,
            "papers_evaluated": n,
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
