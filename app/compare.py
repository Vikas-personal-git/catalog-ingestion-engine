"""
Accuracy comparison between two extraction runs of the same pages.

    python -m app.compare output/<reference_job> output/<candidate_job>

Typical use: reference = a premium-model run (or a human-reviewed gold run), candidate = the
cheap cascade. Products are matched per page by SKU or fuzzy name. The report shows:
  - value disagreements between products BOTH runs validated (a published value is wrong on
    one side — these must be zero for the cheaper run to be safe),
  - products only one run validated (coverage differences),
  - field-level agreement rates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

FIELDS = ("mrp", "b2b_price", "moq")


def _load(job_dir: Path) -> Tuple[List[dict], List[dict], Optional[set]]:
    clean = json.loads((job_dir / "catalog_clean.json").read_text(encoding="utf-8"))["products"]
    exc = json.loads((job_dir / "exceptions.json").read_text(encoding="utf-8"))
    pages = None
    if (job_dir / "job.json").exists():
        job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        pages = {p["page_number"] for p in job.get("pages", []) if p.get("tier_used")}
    else:  # older runs: every page that is not a page error was processed
        errors = {e["page_number"] for e in exc.get("page_errors", [])}
        seen = {p["page"] for p in clean + exc["products"]}
        pages = seen - errors
    for p in clean:
        p["_validated"] = True
    for p in exc["products"]:
        p["_validated"] = False
    return clean, exc["products"], pages


def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _match(a: dict, b: dict) -> float:
    if a.get("sku_or_model") and b.get("sku_or_model") and _norm(a["sku_or_model"]) == _norm(b["sku_or_model"]):
        return 100.0
    return fuzz.token_set_ratio(_norm(a["product_name"]), _norm(b["product_name"]))


def compare(ref_dir: Path, cand_dir: Path) -> dict:
    ref_clean, ref_exc, ref_pages = _load(ref_dir)
    cand_clean, cand_exc, cand_pages = _load(cand_dir)
    pages = (ref_pages or set()) & (cand_pages or set())

    by_page: Dict[int, Tuple[List[dict], List[dict]]] = {}
    for p in ref_clean + ref_exc:
        if p["page"] in pages:
            by_page.setdefault(p["page"], ([], []))[0].append(p)
    for p in cand_clean + cand_exc:
        if p["page"] in pages:
            by_page.setdefault(p["page"], ([], []))[1].append(p)

    both_validated: List[Tuple[dict, dict]] = []
    ref_only: List[Tuple[dict, Optional[dict]]] = []
    cand_only: List[Tuple[dict, Optional[dict]]] = []
    for page, (refs, cands) in sorted(by_page.items()):
        unused = list(cands)
        for r in refs:
            best = max(unused, key=lambda c: _match(r, c), default=None)
            m = best if best is not None and _match(r, best) >= 80 else None
            if m is not None:
                unused.remove(m)
            if r["_validated"] and m is not None and m["_validated"]:
                both_validated.append((r, m))
            elif r["_validated"]:
                ref_only.append((r, m))
            elif m is not None and m["_validated"]:
                cand_only.append((m, r))
        cand_only += [(c, None) for c in unused if c["_validated"]]

    disagreements = []
    agree = {f: 0 for f in FIELDS}
    name_agree = 0
    for r, c in both_validated:
        diffs = {f: (r.get(f), c.get(f)) for f in FIELDS if r.get(f) != c.get(f)}
        for f in FIELDS:
            agree[f] += f not in diffs
        if _norm(r["product_name"]) == _norm(c["product_name"]):
            name_agree += 1
        else:
            diffs["product_name"] = (r["product_name"], c["product_name"])
        if any(f in diffs for f in FIELDS):
            disagreements.append({"page": r["page"], "product": r["product_name"], "diffs": diffs})
        elif "product_name" in diffs:
            disagreements.append({"page": r["page"], "product": r["product_name"], "diffs": diffs,
                                  "name_only": True})

    n = len(both_validated)
    return {
        "pages_compared": len(pages),
        "reference_validated": sum(p["_validated"] for p in ref_clean if p["page"] in pages),
        "candidate_validated": sum(p["_validated"] for p in cand_clean if p["page"] in pages),
        "both_validated": n,
        "field_agreement": {f: (agree[f] / n if n else None) for f in FIELDS} | {
            "product_name": name_agree / n if n else None},
        "value_disagreements": [d for d in disagreements if not d.get("name_only")],
        "name_differences": [d for d in disagreements if d.get("name_only")],
        "validated_only_in_reference": [
            {"page": r["page"], "product": r["product_name"],
             "candidate": (m and {"validated": m["_validated"],
                                  "reasons": m.get("failure_reasons", [])[:2]}) or "not extracted"}
            for r, m in ref_only],
        "validated_only_in_candidate": [
            {"page": c["page"], "product": c["product_name"],
             "reference": (r and {"validated": r["_validated"],
                                  "reasons": r.get("failure_reasons", [])[:2]}) or "not extracted"}
            for c, r in cand_only],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    rep = compare(args.reference, args.candidate)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pages compared            : {rep['pages_compared']}")
    print(f"Validated  reference / candidate / both : "
          f"{rep['reference_validated']} / {rep['candidate_validated']} / {rep['both_validated']}")
    print("Agreement on products both validated:")
    for f, v in rep["field_agreement"].items():
        print(f"  {f:<13} {'—' if v is None else f'{v:.1%}'}")
    print(f"\nVALUE DISAGREEMENTS (both validated, different values): {len(rep['value_disagreements'])}")
    for d in rep["value_disagreements"]:
        print(f"  p{d['page']} {d['product']}: {d['diffs']}")
    print(f"\nName wording differences only: {len(rep['name_differences'])}")
    for d in rep["name_differences"]:
        print(f"  p{d['page']} {d['diffs']['product_name'][0]!r} vs {d['diffs']['product_name'][1]!r}")
    print(f"\nValidated only in reference: {len(rep['validated_only_in_reference'])}")
    for d in rep["validated_only_in_reference"]:
        print(f"  p{d['page']} {d['product']} → candidate: {d['candidate']}")
    print(f"\nValidated only in candidate: {len(rep['validated_only_in_candidate'])}")
    for d in rep["validated_only_in_candidate"]:
        print(f"  p{d['page']} {d['product']} → reference: {d['reference']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
