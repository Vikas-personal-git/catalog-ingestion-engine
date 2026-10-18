"""
Stage 5a – Catalog Diffing.
Persistent master catalog (JSON) that validated products are compared against across runs:
  - Brand-new key        → change_status "new", record created.
  - Existing, changed    → change_status "updated", per-field old/new (+ numeric delta) recorded.
  - Existing, unchanged  → change_status "unchanged", record left as-is.
Also resolves duplicate SKUs within one catalog run.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from rapidfuzz import fuzz

from app.schemas import ProcessedProduct

_lock = threading.Lock()

TRACKED_FIELDS = ("mrp", "b2b_price", "moq",
                  "material", "capacity_or_size", "dimensions", "color_variants", "price_slabs")
# Products without a SKU are identified by name; a re-extraction may word the name slightly
# differently, so an unambiguous fuzzy match within the same brand counts as the same product.
NAME_MATCH_THRESHOLD = 92
# Same SKU twice in one catalog only conflicts if the commercial terms differ; free-text fields
# (description, category) are VLM wording and legitimately vary between two mentions.
COMMERCIAL_FIELDS = ("mrp", "b2b_price", "moq", "price_slabs")
NUMERIC_FIELDS = {"mrp", "b2b_price", "moq"}


def _same(a: object, b: object) -> bool:
    """Equality that ignores case / spacing / punctuation differences in text values."""
    if isinstance(a, str) and isinstance(b, str):
        return re.sub(r"[\W_]+", "", a.lower()) == re.sub(r"[\W_]+", "", b.lower())
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(_same(x, y) for x, y in zip(a, b))
    return a == b


def product_key(p: ProcessedProduct, brand: str = "") -> str:
    ident = p.extracted.sku_or_model or p.extracted.product_name
    ident = re.sub(r"\s+", " ", ident).strip().lower()
    return f"{brand.strip().lower()}|{ident}" if brand.strip() else ident


def snapshot(p: ProcessedProduct) -> Dict[str, object]:
    e = p.extracted
    return {
        "product_name": e.product_name,
        "sku_or_model": e.sku_or_model,
        "category": e.category,
        "mrp": e.mrp,
        "b2b_price": e.b2b_price,
        "moq": e.moq,
        "material": e.specifications.material,
        "capacity_or_size": e.specifications.capacity_or_size,
        "dimensions": e.specifications.dimensions,
        "color_variants": sorted(e.color_variants),
        "price_slabs": sorted(([s.min_qty, s.max_qty, s.price] for s in e.price_slabs), key=lambda s: s[0]),
    }


def _diff(old: Dict[str, object], new: Dict[str, object],
          fields: tuple = TRACKED_FIELDS) -> Dict[str, Dict[str, object]]:
    changes: Dict[str, Dict[str, object]] = {}
    for f in fields:
        if not _same(old.get(f), new.get(f)):
            entry: Dict[str, object] = {"old": old.get(f), "new": new.get(f)}
            if f in NUMERIC_FIELDS and isinstance(old.get(f), (int, float)) and isinstance(new.get(f), (int, float)):
                entry["delta"] = round(float(new[f]) - float(old[f]), 4)  # type: ignore[arg-type]
            changes[f] = entry
    return changes


def resolve_duplicates(
    products: List[ProcessedProduct], brand: str = ""
) -> Tuple[List[ProcessedProduct], List[ProcessedProduct]]:
    """
    Within one run: repeats of a SKU with the same commercial terms are merged into the first
    (images combined); repeats with conflicting prices / MOQ / slabs are all quarantined.
    Returns (kept, quarantined).
    """
    groups: Dict[str, List[ProcessedProduct]] = {}
    for p in products:
        groups.setdefault(product_key(p, brand), []).append(p)

    kept: List[ProcessedProduct] = []
    quarantined: List[ProcessedProduct] = []
    for group in groups.values():
        first = group[0]
        if len(group) == 1:
            kept.append(first)
            continue
        snaps = [snapshot(p) for p in group]
        pages = ", ".join(str(p.page_number) for p in group)
        if all(_diff(snaps[0], s, COMMERCIAL_FIELDS) == {} for s in snaps[1:]):
            for dup in group[1:]:
                first.image_paths.extend(x for x in dup.image_paths if x not in first.image_paths)
            kept.append(first)
        else:
            for p in group:
                p.validation.passed = False
                p.validation.failure_reasons.append(
                    f"duplicate SKU/name in this catalog (pages {pages}) with conflicting values"
                )
                quarantined.append(p)
    return kept, quarantined


class CatalogStore:
    def __init__(self, path: Path):
        self.path = path
        self.records: Dict[str, dict] = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.records = json.load(f).get("products", {})

    def resolve_key(self, p: ProcessedProduct, brand: str = "") -> str:
        key = product_key(p, brand)
        if key in self.records or p.extracted.sku_or_model:
            return key
        name = re.sub(r"\s+", " ", p.extracted.product_name).strip().lower()
        scored = sorted(
            ((fuzz.token_set_ratio(name, (rec.get("data", {}).get("product_name") or "").lower()), k)
             for k, rec in self.records.items()
             if (rec.get("brand") or "").lower() == brand.strip().lower() and not rec.get("data", {}).get("sku_or_model")),
            reverse=True)
        if scored and scored[0][0] >= NAME_MATCH_THRESHOLD and (len(scored) == 1 or scored[1][0] < NAME_MATCH_THRESHOLD):
            return scored[0][1]
        return key

    def diff(self, products: List[ProcessedProduct], brand: str = "") -> None:
        """Set change_status / changes on each product (does not modify the store)."""
        for p in products:
            existing = self.records.get(self.resolve_key(p, brand))
            if existing is None:
                p.change_status, p.changes = "new", {}
            else:
                p.changes = _diff(existing.get("data", {}), snapshot(p))
                p.change_status = "updated" if p.changes else "unchanged"

    def apply(self, products: List[ProcessedProduct], job_id: str, source: str, brand: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for p in products:
            key = self.resolve_key(p, brand)
            rec = self.records.get(key)
            if p.change_status == "new" or rec is None:
                self.records[key] = {
                    "brand": brand or None,
                    "data": snapshot(p),
                    "image_paths": [f"{job_id}/{x}" for x in p.image_paths],
                    "first_seen": now, "last_seen": now, "last_job": job_id, "source": source,
                    "history": [],
                }
            else:
                rec["last_seen"], rec["last_job"] = now, job_id
                if p.change_status == "updated":
                    rec["data"] = snapshot(p)
                    rec["history"].append({"job_id": job_id, "at": now, "changes": p.changes})
                    if p.image_paths:
                        rec["image_paths"] = [f"{job_id}/{x}" for x in p.image_paths]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"products": self.records}, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, self.path)


def diff_and_update(
    products: List[ProcessedProduct], path: Path, job_id: str, source: str,
    brand: str = "", update: bool = True,
) -> None:
    """Diff validated products against the master catalog and (optionally) persist them."""
    with _lock:
        store = CatalogStore(path)
        store.diff(products, brand)
        if update:
            store.apply(products, job_id, source, brand)
            store.save()
