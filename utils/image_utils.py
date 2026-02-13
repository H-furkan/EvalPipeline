"""
utils/image_utils.py — Slide image utilities.

Provides:
  • pptx_to_images()   — convert a PPTX file to PNG slides via LibreOffice
  • resize_images()    — resize images in-place (reduces VLM token usage)
  • resize_images_tmp()— resize images to a temp directory (non-destructive)
  • sample_slides()    — evenly-distributed slide sampling

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


# ── PPTX → PNG conversion ─────────────────────────────────────────────────────

def pptx_to_images(pptx_path: str | Path, out_dir: str | Path) -> list[str]:
    """
    Convert a PPTX file to PNG slides using LibreOffice headless.

    Slides are written to *out_dir* as slide_001.png, slide_002.png, …
    (zero-padded to 3 digits for reliable alphabetic sorting).

    Skips conversion entirely if slide_001.png already exists in *out_dir*.

    Returns a sorted list of absolute PNG file paths, or [] on failure.
    """
    pptx_path = Path(pptx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if already converted ────────────────────────────────────────────
    existing = sorted(out_dir.glob("slide_*.png"))
    if existing:
        print(f"  [image] Skipping {pptx_path.name} ({len(existing)} slides already in {out_dir})")
        return [str(p) for p in existing]

    # ── Locate LibreOffice binary ────────────────────────────────────────────
    lo_bin = C.LIBREOFFICE_PATH
    if not Path(lo_bin).exists():
        lo_bin = shutil.which("libreoffice") or shutil.which("soffice") or lo_bin

    # LibreOffice drops output files into the --outdir with its own naming.
    # Use a temp sub-directory so we can rename cleanly afterwards.
    tmp_dir = out_dir / "_lo_tmp"
    tmp_dir.mkdir(exist_ok=True)

    cmd = [
        lo_bin,
        "--headless",
        "--convert-to", "png",
        "--outdir", str(tmp_dir),
        str(pptx_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  [image] LibreOffice error: {result.stderr.decode()[:300]}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [image] LibreOffice conversion failed: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []

    # ── Rename to slide_001.png, slide_002.png, … ────────────────────────────
    raw_pngs = sorted(tmp_dir.glob("*.png"))
    if not raw_pngs:
        print(f"  [image] No PNGs produced for {pptx_path.name}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return []

    renamed = []
    for i, p in enumerate(raw_pngs, 1):
        dest = out_dir / f"slide_{i:03d}.png"
        p.rename(dest)
        renamed.append(str(dest))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  [image] Converted {pptx_path.name} → {len(renamed)} slides in {out_dir}")
    return renamed


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
