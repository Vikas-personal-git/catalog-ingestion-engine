"""
Folder-structured export of a finished job — the hand-off package for teams that keep their
catalog in Excel + Google Drive / shared folders:

    catalog_<job>.zip
      products.xlsx            one row per validated product, image links relative to the ZIP
      products.csv             same data, flat (for portal bulk imports)
      needs_review.xlsx        quarantined products with reasons
      <Category>/<Brand>/<Product name (SKU)>/
          <sku>_1.png …        background-removed cut-outs
          original/<sku>_1.jpg …  the same crops before background removal

    python -m app.export_zip output/<job_id> [--brand "Fallback Brand"]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

from app.output_writer import export_excel
from app.reimage import record_to_product

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(text: str, limit: int = 80) -> str:
    """A folder/file name that is valid on Windows, macOS, Linux and Google Drive."""
    name = _INVALID.sub(" ", text or "").strip()
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:limit].rstrip(" .") or "Unnamed"


def _category_key(text: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    return re.sub(r"(es|s)$", "", key)


def normalize_categories(records: List[dict]) -> Dict[str, str]:
    """Map every printed category spelling ("LUNCH BOX", "Lunch Boxes") to one display name."""
    spellings: Dict[str, Counter] = {}
    for rec in records:
        cat = re.sub(r"\s+", " ", (rec.get("category") or "").strip())
        if cat:
            spellings.setdefault(_category_key(cat), Counter())[cat] += 1
    chosen = {}
    for key, counts in spellings.items():
        best = counts.most_common(1)[0][0]
        chosen[key] = best.title() if best.isupper() or best.islower() else best
    return {(rec.get("category") or ""): chosen.get(_category_key(rec.get("category") or ""), "Uncategorized")
            for rec in records}


def build_zip(job_dir: Path, fallback_brand: str = "") -> Path:
    job_dir = Path(job_dir)
    job_id = job_dir.name
    clean = json.loads((job_dir / "catalog_clean.json").read_text(encoding="utf-8"))["products"]
    exceptions = json.loads((job_dir / "exceptions.json").read_text(encoding="utf-8"))["products"]
    if not fallback_brand and (job_dir / "job.json").exists():
        fallback_brand = json.loads((job_dir / "job.json").read_text(encoding="utf-8")).get("brand", "")

    staging = job_dir / "_export"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    categories = normalize_categories(clean)
    used: set = set()
    exported: List[dict] = []

    for rec in clean:
        category = categories.get(rec.get("category") or "", "Uncategorized")
        brand = (rec.get("brand") or fallback_brand or "").strip()
        brand = brand.title() if brand.isupper() else brand
        sku = (rec.get("sku_or_model") or "").strip()
        product = rec["product_name"] + (f" ({sku})" if sku and sku.lower() not in rec["product_name"].lower() else "")
        rel = Path(safe_name(category)) / safe_name(brand or "Unknown Brand") / safe_name(product)
        base, n = rel, 2
        while str(rel).lower() in used:            # same name twice → keep both, numbered
            rel = base.with_name(f"{base.name} {n}"); n += 1
        used.add(str(rel).lower())
        folder = staging / rel
        folder.mkdir(parents=True)

        stem = safe_name(sku or rec["product_name"], 40).replace(" ", "_")
        images, originals = [], []
        for i, path in enumerate(rec.get("image_paths", []), 1):
            src = job_dir / path
            if src.exists():
                dst = folder / f"{stem}_{i}{src.suffix}"
                shutil.copy2(src, dst)
                images.append(dst.relative_to(staging).as_posix())
        for i, path in enumerate(rec.get("original_image_paths", []), 1):
            src = job_dir / path
            if src.exists():
                (folder / "original").mkdir(exist_ok=True)
                dst = folder / "original" / f"{stem}_{i}{src.suffix}"
                shutil.copy2(src, dst)
                originals.append(dst.relative_to(staging).as_posix())
        exported.append({**rec, "category": category, "brand": brand or None, "folder": rel.as_posix(),
                         "image_paths": images, "original_image_paths": originals})

    export_excel([record_to_product(r, True) for r in exported], [], staging / "products.xlsx")
    if exceptions:
        export_excel([], [record_to_product(r, False) for r in exceptions], staging / "needs_review.xlsx")
    _write_csv(exported, staging / "products.csv")

    zip_path = job_dir / f"catalog_{job_id if len(job_id) <= 24 else job_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging).as_posix())
    shutil.rmtree(staging)
    return zip_path


def _write_csv(records: List[dict], path: Path) -> None:
    cols = ["category", "brand", "product_name", "sku_or_model", "mrp", "b2b_price", "moq", "color_variants",
            "material", "capacity_or_size", "dimensions", "description", "price_slabs", "folder", "images"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM so Excel shows ₹ / accents correctly
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            spec = r.get("specifications") or {}
            w.writerow([
                r["category"], r.get("brand") or "", r["product_name"], r.get("sku_or_model") or "",
                r.get("mrp") if r.get("mrp") is not None else "", r.get("b2b_price") if r.get("b2b_price") is not None else "",
                r.get("moq") if r.get("moq") is not None else "", "; ".join(r.get("color_variants") or []),
                spec.get("material") or "", spec.get("capacity_or_size") or "", spec.get("dimensions") or "",
                r.get("description") or "",
                "; ".join(f"{s['min_qty']}+ pcs: {s['price']:g}" for s in r.get("price_slabs") or []),
                r["folder"], "; ".join(r["image_paths"]),
            ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.export_zip", description=__doc__.split("\n\n")[0])
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--brand", default="", help="brand to use when a page does not print one")
    args = ap.parse_args(argv)
    print(build_zip(args.job_dir, args.brand))
    return 0


if __name__ == "__main__":
    sys.exit(main())
