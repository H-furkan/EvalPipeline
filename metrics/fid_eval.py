"""
metrics/fid_eval.py — FID (Fréchet Inception Distance) evaluation.

Computes the FID score between generated slide images and reference (ground-truth)
slide images. Lower FID = more visually similar to the reference.

This metric requires:
  • Reference slide images in GENERATED_SAMPLES_DIR/gt/{paper}/ (as PNG/JPG)
  • Generated slide images (converted from PPTX via LibreOffice)

Dependencies: pip install torch torchvision scipy
Optional:     faster-pytorch-fid for GPU-accelerated computation

Output: results/fid_eval.json
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
from utils.image_utils import find_and_convert_images, get_method_images, slides_to_images

METRIC_NAME = "fid_eval"

# Minimum number of images required per set for FID to be meaningful
MIN_IMAGES_FOR_FID = 2


def _load_fid_modules():
    """Try to load FID computation libraries. Returns (compute_fid_func, InceptionV3) or raises."""
    try:
        import torch
        import numpy as np
        from scipy import linalg
        from torchvision import models, transforms
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image

        class ImagePathDataset(Dataset):
            def __init__(self, image_paths, transform=None):
                self.image_paths = image_paths
                self.transform = transform

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                img = Image.open(self.image_paths[idx]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img

        def get_inception_features(image_paths, model, device, batch_size=32):
            """Extract InceptionV3 features from a set of images."""
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            dataset = ImagePathDataset(image_paths, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            features = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    feat = model(batch)
                    # InceptionV3 output is [batch, 2048, 1, 1] with aux_logits=False
                    if feat.dim() == 4:
                        feat = feat.squeeze(-1).squeeze(-1)
                    features.append(feat.cpu().numpy())
            return np.concatenate(features, axis=0)

        def compute_fid(ref_paths, gen_paths, device="cpu"):
            """Compute FID between two sets of images."""
            # Load InceptionV3
            inception = models.inception_v3(pretrained=True, transform_input=False)
            # Remove final classification layer, keep up to avgpool
            inception.fc = torch.nn.Identity()
            inception.eval()
            inception.to(device)

            ref_feats = get_inception_features(ref_paths, inception, device)
            gen_feats = get_inception_features(gen_paths, inception, device)

            # Compute statistics
            mu1, sigma1 = ref_feats.mean(axis=0), np.cov(ref_feats, rowvar=False)
            mu2, sigma2 = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

            # Handle single-image case (cov returns scalar)
            if sigma1.ndim == 0:
                sigma1 = np.array([[sigma1]])
            if sigma2.ndim == 0:
                sigma2 = np.array([[sigma2]])

            # Compute FID
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fid_value = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid_value)

        return compute_fid

    except ImportError as e:
        raise ImportError(
            f"FID evaluation requires torch, torchvision, scipy. Missing: {e}"
        )


def _find_reference_images(paper_name: str) -> list[str]:
    """Find reference (ground-truth) slide images for a paper."""
    # Look in gt method directory for existing images
    gt_dir = Path(C.GENERATED_SAMPLES_DIR) / "gt" / paper_name
    if gt_dir.exists():
        images = (
            list(gt_dir.glob("*.png"))
            + list(gt_dir.glob("*.jpg"))
            + list(gt_dir.glob("*.jpeg"))
        )
        if images:
            return sorted(str(p) for p in images)

    # Try to find and convert from PPTX/PDF via cache
    return find_and_convert_images("gt", paper_name)


def run(papers: list[str], baseline_methods: list[str]) -> dict:
    """
    Compute FID for all methods (ours + baselines) against ground-truth slides.

    FID requires reference images (from 'gt' method). Papers without ground-truth
    images are skipped.
    """
    out_path = result_path(METRIC_NAME)
    existing = load_existing(out_path)
    per_paper: dict = existing.get("per_paper", {})
    metadata = make_metadata(METRIC_NAME, "InceptionV3")

    all_methods = [C.OURS_METHOD] + [m for m in baseline_methods if m != "gt"]

    # Try to load FID computation module
    try:
        compute_fid = _load_fid_modules()
    except ImportError as e:
        print(f"[{METRIC_NAME}] {e}")
        print(f"[{METRIC_NAME}] Skipping FID evaluation.")
        return {}

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{METRIC_NAME}] Using device: {device}")

    for i, paper in enumerate(papers, 1):
        print(f"\n[{METRIC_NAME}] [{i}/{len(papers)}] {paper}")

        # Get reference images
        ref_images = _find_reference_images(paper)
        if len(ref_images) < MIN_IMAGES_FOR_FID:
            print(f"  Skipping: fewer than {MIN_IMAGES_FOR_FID} reference images found")
            continue

        if paper not in per_paper:
            per_paper[paper] = {}

        for method in all_methods:
            if method in per_paper.get(paper, {}):
                print(f"  Skipping {method} (already done)")
                continue

            gen_images = find_and_convert_images(method, paper)
            if len(gen_images) < MIN_IMAGES_FOR_FID:
                print(f"  Skipping {method}: fewer than {MIN_IMAGES_FOR_FID} images")
                continue

            try:
                fid_score = compute_fid(ref_images, gen_images, device=device)
                per_paper[paper][method] = {"fid": fid_score}
                print(f"  {method}: FID={fid_score:.2f}")
            except Exception as e:
                print(f"  {method}: FID computation failed: {e}")
                per_paper[paper][method] = {"fid": None, "error": str(e)}

        save_incremental(out_path, {"metadata": metadata, "per_paper": per_paper})

    # Per-method summary
    per_method: dict[str, dict] = {}
    for method in all_methods:
        scores = [
            per_paper[p][method]["fid"]
            for p in per_paper
            if method in per_paper[p] and per_paper[p][method].get("fid") is not None
        ]
        per_method[method] = {
            "mean_fid": sum(scores) / len(scores) if scores else None,
            "papers_evaluated": len(scores),
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
