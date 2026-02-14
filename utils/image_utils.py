"""
utils/image_utils.py — Slide image utilities.

Provides:
  • slides_to_images()  — convert PPTX or PDF to PNG slides via LibreOffice
  • pptx_to_images()    — convert a PPTX file to PNG slides (legacy alias)
  • pdf_to_images()     — convert a PDF file to PNG slides
  • resize_images()     — resize images in-place (reduces VLM token usage)
  • resize_images_tmp() — resize images to a temp directory (non-destructive)
  • sample_slides()     — evenly-distributed slide sampling
  • find_and_convert()  — find a PPTX/PDF and convert to images

Output folder structure
-----------------------
  IMAGES_CACHE_DIR/
    {method}/
      {paper}/
        slide_001.png
        slide_002.png
        ...
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C


# ── Presentation → PNG conversion ────────────────────────────────────────────

def _convert_pptx_to_pdf(pptx_path: Path, out_dir: Path) -> Path | None:
    """Convert a PPTX to PDF via LibreOffice headless. Returns PDF path or None."""
    lo_bin = C.LIBREOFFICE_PATH
    if not Path(lo_bin).exists():
        lo_bin = shutil.which("libreoffice") or shutil.which("soffice") or lo_bin

    cmd = [
        lo_bin,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", str(out_dir),
        str(pptx_path),
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [image] LibreOffice PPTX->PDF error: {result.stderr.decode()[:300]}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [image] LibreOffice PPTX->PDF failed: {e}")
        return None

    pdf_path = out_dir / (pptx_path.stem + ".pdf")
    if pdf_path.exists():
        return pdf_path
    # LibreOffice may use a different name
    pdfs = list(out_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def _convert_pdf_via_pymupdf(pdf_path: Path, out_dir: Path) -> list[str]:
    """
    Convert a PDF to PNG images using pymupdf (fallback when LibreOffice
    produces only a single image for a multi-page PDF).
    """
    try:
        import pymupdf
    except ImportError:
        print("  [image] pymupdf not available for PDF->PNG fallback")
        return []

    try:
        doc = pymupdf.open(str(pdf_path))
        renamed = []
        for i, page in enumerate(doc, 1):
            pix = page.get_pixmap(dpi=150)
            dest = out_dir / f"slide_{i:03d}.png"
            pix.save(str(dest))
            renamed.append(str(dest))
        doc.close()
        print(f"  [image] Converted {pdf_path.name} -> {len(renamed)} slides via pymupdf")
        return renamed
    except Exception as e:
        print(f"  [image] pymupdf PDF conversion failed: {e}")
        return []


def slides_to_images(file_path: str | Path, out_dir: str | Path) -> list[str]:
    """
    Convert a PPTX or PDF file to PNG slides.

    Slides are written to *out_dir* as slide_001.png, slide_002.png, ...
    Skips conversion entirely if slide_001.png already exists in *out_dir*.

    Returns a sorted list of absolute PNG file paths, or [] on failure.
    """
    file_path = Path(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already converted
    existing = sorted(out_dir.glob("slide_*.png"))
    if existing:
        print(f"  [image] Skipping {file_path.name} ({len(existing)} slides already in {out_dir})")
        return [str(p) for p in existing]

    if file_path.suffix.lower() == ".pdf":
        return _convert_pdf_via_pymupdf(file_path, out_dir)

    # PPTX: convert to PDF first via LibreOffice, then PDF → PNGs via pymupdf
    if file_path.suffix.lower() == ".pptx":
        tmp_pdf = _convert_pptx_to_pdf(file_path, out_dir)
        if tmp_pdf:
            result = _convert_pdf_via_pymupdf(tmp_pdf, out_dir)
            tmp_pdf.unlink(missing_ok=True)  # Clean up intermediate PDF
            return result
        print(f"  [image] Could not convert PPTX to PDF: {file_path.name}")
        return []

    print(f"  [image] Unsupported format: {file_path.suffix}")
    return []


def pptx_to_images(pptx_path: str | Path, out_dir: str | Path) -> list[str]:
    """Legacy alias for slides_to_images(). Works with both PPTX and PDF."""
    return slides_to_images(pptx_path, out_dir)


def pdf_to_images(pdf_path: str | Path, out_dir: str | Path) -> list[str]:
    """Convert a PDF file to PNG slides. Alias for slides_to_images()."""
    return slides_to_images(pdf_path, out_dir)


# ── Image resizing ────────────────────────────────────────────────────────────

def resize_images(image_paths: list[str], target_width: int = C.IMAGE_TARGET_WIDTH) -> None:
    """
    Resize images *in place* to *target_width* pixels wide, keeping aspect ratio.
    Only shrinks; never upscales.
    """
    for path in image_paths:
        try:
            img = Image.open(path)
            w, h = img.size
            if w > target_width:
                new_h = int(h * target_width / w)
                img = img.resize((target_width, new_h), Image.LANCZOS)
                img.save(path)
        except Exception as e:
            print(f"  [image] Could not resize {path}: {e}")


def resize_images_tmp(
    image_paths: list[str],
    target_width: int = C.IMAGE_TARGET_WIDTH,
    tmp_dir: str | Path | None = None,
) -> list[str]:
    """
    Resize images into a temporary directory (non-destructive).

    Returns a list of paths to the resized copies.
    If *tmp_dir* is None a system temp directory is used.
    """
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="evalp_imgs_"))
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    new_paths = []
    for path in image_paths:
        try:
            img = Image.open(path)
            w, h = img.size
            if w > target_width:
                new_h = int(h * target_width / w)
                img = img.resize((target_width, new_h), Image.LANCZOS)
            dest = tmp_dir / Path(path).name
            img.save(str(dest))
            new_paths.append(str(dest))
        except Exception as e:
            print(f"  [image] Could not resize {path}: {e}")
            new_paths.append(path)  # Use original on failure
    return new_paths


# ── Slide sampling ────────────────────────────────────────────────────────────

def sample_slides(paths: list[str], max_n: int = C.MAX_SLIDES_FOR_VISUAL) -> list[str]:
    """
    Select up to *max_n* slides from *paths* with even spacing.

    If len(paths) <= max_n, returns all paths (sorted).
    Otherwise picks evenly spaced indices so coverage spans the full deck.
    """
    paths = sorted(paths)
    n = len(paths)
    if n <= max_n:
        return paths
    indices = [int(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    return [paths[i] for i in indices]


def get_method_images(
    method: str,
    paper_name: str,
    images_cache_dir: str | Path | None = None,
) -> list[str]:
    """
    Return sorted slide image paths for a given method/paper combination.

    Looks in *images_cache_dir*/{method}/{paper_name}/ for *.jpg, *.png files.
    Falls back to constants.IMAGES_CACHE_DIR if not specified.
    """
    cache = Path(images_cache_dir or C.IMAGES_CACHE_DIR)
    paper_dir = cache / method / paper_name
    if not paper_dir.exists():
        return []
    images = (
        list(paper_dir.glob("*.jpg"))
        + list(paper_dir.glob("*.jpeg"))
        + list(paper_dir.glob("*.png"))
    )
    return sorted(str(p) for p in images)


def find_and_convert_images(method: str, paper_name: str) -> list[str]:
    """
    Find or convert slide images for a method/paper combination.

    1. Check the image cache for existing PNGs.
    2. Check the source method directory for existing images (PNG/JPG).
    3. If no images found, look for a PPTX or PDF file and convert it.
    4. Returns a sorted list of image file paths, or [] if nothing found.
    """
    # 1. Check cache
    cached = get_method_images(method, paper_name)
    if cached:
        return cached

    # 2. Check source directory for existing images
    method_dir = Path(C.GENERATED_SAMPLES_DIR) / method / paper_name
    if not method_dir.exists():
        return []

    existing = sorted(
        list(method_dir.glob("*.png"))
        + list(method_dir.glob("*.jpg"))
        + list(method_dir.glob("*.jpeg"))
    )
    if existing:
        return [str(p) for p in existing]

    # 3. Find a presentation file (PPTX or PDF) and convert
    pres_file = None
    pptx_files = list(method_dir.glob("*.pptx"))
    if pptx_files:
        pres_file = pptx_files[0]
    else:
        pdf_files = list(method_dir.glob("*.pdf"))
        if pdf_files:
            pres_file = pdf_files[0]

    if pres_file is None:
        return []

    out_dir = Path(C.IMAGES_CACHE_DIR) / method / paper_name
    return slides_to_images(pres_file, out_dir)
