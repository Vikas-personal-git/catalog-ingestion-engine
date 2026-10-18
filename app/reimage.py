"""
Re-run only Stage 4 (crop + background removal) for a finished job — no AI calls, no cost.
Useful after improving image processing:

    python -m app.reimage output/<job_id> <original input PDF or folder>

Rewrites images/, then catalog_clean.json and catalog_clean.xlsx with the new image paths.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from app.config import settings
from app.image_processor import CropSource, crop_and_clean
from app.output_writer import export_excel
from app.pdf_renderer import enumerate_pages
from app.schemas import CROPPABLE_TYPES, ExtractedProduct, ProcessedProduct, ValidationResult
from app.verifier import bbox_problems

_EXTRACTED_FIELDS = ("product_name", "sku_or_model", "brand", "category", "description", "mrp", "b2b_price", "moq",
                     "color_variants", "specifications", "images", "price_slabs", "raw_text_snippet")


def record_to_product(rec: dict, passed: bool) -> ProcessedProduct:
    """Rebuild a ProcessedProduct from its exported JSON record (catalog_clean / exceptions)."""
    return ProcessedProduct(
        page_number=rec["page"], source=rec.get("source", ""), tier=rec.get("tier", ""),
        extracted=ExtractedProduct.model_validate({k: rec.get(k) for k in _EXTRACTED_FIELDS}),
        validation=ValidationResult(confidence_score=rec.get("confidence_score", 0), passed=passed,
                                    failure_reasons=rec.get("warnings" if passed else "failure_reasons", []),
                                    check_scores=rec.get("check_scores", {}), text_source=rec.get("text_source", "native")),
        image_paths=rec.get("image_paths", []), original_image_paths=rec.get("original_image_paths", []),
        image_warnings=rec.get("image_warnings", []), background_removed=rec.get("background_removed"),
        change_status=rec.get("change_status", "n/a"), changes=rec.get("changes", {}),
    )


def reimage(job_dir: Path, input_path: Path) -> dict:
    clean_path = job_dir / "catalog_clean.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    refs = {r.page_number: r for r in enumerate_pages(input_path)}
    page_counts: dict = {}
    for rec in clean["products"]:
        page_counts[rec["page"]] = page_counts.get(rec["page"], 0) + 1

    images_dir = job_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    stats = {"products": 0, "images": 0, "fallback_to_original": 0}
    used: dict = {}
    for i, rec in enumerate(clean["products"], 1):
        ref = refs[rec["page"]]
        source = CropSource(ref.source, ref.index, ref.is_image)
        rec["image_paths"], rec["original_image_paths"] = [], []
        rec["image_warnings"] = [w for w in rec.get("image_warnings", []) if ".png:" not in w]
        rec["background_removed"] = None
        ident = rec.get("sku_or_model") or rec["product_name"]
        base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in ident).strip("._-")[:60] or "item"
        for img in rec.get("images", []):
            if img["type"] not in CROPPABLE_TYPES or \
                    bbox_problems(img["box_2d"], page_counts[rec["page"]], img["type"] == "variant"):
                continue
            n = used.get(base.lower(), 0); used[base.lower()] = n + 1
            name = f"{base}_{n}.png"
            crop = crop_and_clean(source, img["box_2d"], images_dir / name, 0.03, True, settings.rembg_model,
                                  settings.min_image_px, 3, settings.max_image_px)
            if not crop.path:
                continue
            rec["image_paths"].append(f"images/{name}")
            if crop.original_path:
                rec["original_image_paths"].append(f"images/{crop.original_path.name}")
            rec["background_removed"] = crop.background_removed if rec["background_removed"] is None \
                else rec["background_removed"] and crop.background_removed
            if crop.upscale > 1.05:
                rec["image_warnings"].append(f"{name}: source photo is low resolution — upscaled {crop.upscale:.1f}×")
            if crop.warning:
                rec["image_warnings"].append(f"{name}: {crop.warning}")
                if "using the original photo" in crop.warning:
                    stats["fallback_to_original"] += 1
            stats["images"] += 1
        stats["products"] += 1
        print(f"  [{i}/{len(clean['products'])}] p{rec['page']} {rec['product_name'][:40]}: "
              f"{len(rec['image_paths'])} image(s)", flush=True)

    clean_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    exc = json.loads((job_dir / "exceptions.json").read_text(encoding="utf-8"))["products"]
    export_excel([record_to_product(r, True) for r in clean["products"]],
                 [record_to_product(r, False) for r in exc], job_dir / "catalog_clean.xlsx")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.reimage", description=__doc__.split("\n\n")[0])
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("input", type=Path, help="the PDF / folder the job was run on")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    stats = reimage(args.job_dir, args.input)
    print(f"\nDone: {stats['images']} images for {stats['products']} products "
          f"({stats['fallback_to_original']} kept as original photo — background not separable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
