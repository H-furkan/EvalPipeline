"""
data_prep/convert_to_text.py — Data preparation step.

Converts:
  • PPTX presentation JSON (extracted.json) → plain text
  • Original paper PDFs → markdown text

Output folder structure
-----------------------
  PROCESSED_DATA_DIR/
    {paper}/
      orig.txt          ← source paper (PDF → markdown)
      {method}.txt      ← slide text (extracted.json → plain text)
      ...

Files are skipped automatically if they already exist (pass force=True to re-convert).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None


# ── Conversion helpers ────────────────────────────────────────────────────────

def convert_extracted_json_to_text(json_path: Path) -> str | None:
    """
    Convert an extracted.json file (slide descriptions) to a plain text string.

    Expected JSON shape:
        {
          "slide_descriptions": ["Slide 1 content ...", ...],
          "background": {"title": "...", "speaker": "..."}
        }
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        slide_descriptions = data.get("slide_descriptions", [])
        background = data.get("background", {})

        parts = []
        if background.get("title"):
            parts.append(f"# {background['title']}\n")
        if background.get("speaker"):
            parts.append(f"**Speakers:** {background['speaker']}\n")

        for i, description in enumerate(slide_descriptions, 1):
            parts.append(f"\n## Slide {i}\n{description}")

        return "\n".join(parts)
    except Exception as e:
        print(f"  [convert] Error reading {json_path}: {e}")
        return None


def convert_pdf_to_markdown(pdf_path: Path) -> str | None:
    """Convert a PDF to markdown text using pymupdf4llm."""
    if pymupdf4llm is None:
        print("  [convert] pymupdf4llm is not installed. Run: pip install pymupdf4llm")
        return None
    try:
        return pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        print(f"  [convert] Error converting PDF {pdf_path}: {e}")
        return None


# ── Paper / method discovery ──────────────────────────────────────────────────

def discover_papers(papers_dir: Path) -> list[str]:
    """
    Return a sorted list of paper names found in papers_dir.
    A paper is valid if papers_dir/{paper}/original.pdf exists.
    """
    if not papers_dir.exists():
        print(f"  [convert] Papers directory not found: {papers_dir}")
        return []
    return sorted(
        p.name for p in papers_dir.iterdir()
        if p.is_dir() and (p / "original.pdf").exists()
    )


def discover_methods(generated_dir: Path) -> list[str]:
    """Return a sorted list of method sub-directories in generated_dir."""
    if not generated_dir.exists():
        print(f"  [convert] Generated samples directory not found: {generated_dir}")
        return []
    return sorted(
        d.name for d in generated_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


# ── Per-paper processing ──────────────────────────────────────────────────────

def process_paper(
    paper_name: str,
    papers_dir: Path,
    generated_dir: Path,
    output_dir: Path,
    methods: list[str],
    force: bool = False,
) -> dict:
    """
    Convert the source PDF and all method presentations for one paper.

    Returns a dict with conversion success flags.
    """
    stats: dict = {"paper": paper_name, "orig": False, "methods": {}}
    paper_out = output_dir / paper_name
    paper_out.mkdir(parents=True, exist_ok=True)

    # ── Source paper (orig.txt) ────────────────────────────────────────────
    orig_out = paper_out / "orig.txt"
    if orig_out.exists() and not force:
        print(f"  [convert] Skipping {paper_name}/orig.txt (already exists)")
        stats["orig"] = True
    else:
        pdf_path = papers_dir / paper_name / "original.pdf"
        if pdf_path.exists():
            print(f"  [convert] Converting PDF: {paper_name}")
            text = convert_pdf_to_markdown(pdf_path)
            if text:
                orig_out.write_text(text, encoding="utf-8")
                stats["orig"] = True
                print(f"    ✓ Written {orig_out}")
        else:
            print(f"  [convert] Warning: original.pdf not found for {paper_name}")

    # ── Method presentations ───────────────────────────────────────────────
    for method in methods:
        method_out = paper_out / f"{method}.txt"
        if method_out.exists() and not force:
            print(f"  [convert] Skipping {paper_name}/{method}.txt (already exists)")
            stats["methods"][method] = True
            continue

        json_path = generated_dir / method / paper_name / "extracted.json"
        if json_path.exists():
            print(f"  [convert] Converting {method} / {paper_name}")
            text = convert_extracted_json_to_text(json_path)
            if text:
                method_out.write_text(text, encoding="utf-8")
                stats["methods"][method] = True
                print(f"    ✓ Written {method_out}")
            else:
                stats["methods"][method] = False
        else:
            print(f"  [convert] Warning: extracted.json not found for {method}/{paper_name}")
            stats["methods"][method] = False

    return stats


# ── Public entry point ────────────────────────────────────────────────────────

def run_conversion(
    papers: list[str] | None = None,
    methods: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    """
    Convert all papers / methods to text files in PROCESSED_DATA_DIR.

    Args:
        papers:  List of paper names to process. None = auto-discover.
        methods: List of method names to process. None = auto-discover.
        force:   Re-convert even if the output file already exists.

    Returns:
        List of per-paper conversion stats dicts.
    """
    papers_dir = Path(C.BENCHMARK_DATA_DIR)
    generated_dir = Path(C.GENERATED_SAMPLES_DIR)
    output_dir = Path(C.PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = discover_methods(generated_dir)
    if papers is None:
        papers = discover_papers(papers_dir)

    print(f"[convert] Methods : {methods}")
    print(f"[convert] Papers  : {len(papers)} found")

    all_stats = []
    for i, paper in enumerate(papers, 1):
        print(f"\n[convert] [{i}/{len(papers)}] {paper}")
        stats = process_paper(paper, papers_dir, generated_dir, output_dir, methods, force)
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Papers with orig.txt : {sum(s['orig'] for s in all_stats)}/{len(all_stats)}")
    method_totals: dict[str, int] = {}
    for s in all_stats:
        for m, ok in s["methods"].items():
            method_totals[m] = method_totals.get(m, 0) + ok
    for m in sorted(method_totals):
        print(f"  {m}: {method_totals[m]}/{len(all_stats)}")
    print(f"Output: {output_dir.absolute()}")
    return all_stats
