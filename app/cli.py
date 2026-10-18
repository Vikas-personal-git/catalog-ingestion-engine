"""
Command-line entry point.

    python -m app catalog.pdf
    python -m app pages_folder/ --pages 3-20 --brand Acme
    python -m app catalog.pdf --tiers claude:claude-opus-5      # single premium model, no cascade
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from app.config import settings
from app.pipeline import PipelineOptions, job_store, run_pipeline


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app", description="Catalog Ingestion Engine — ingest a supplier PDF catalog.")
    p.add_argument("input", help="PDF file, page image, or folder of PDFs / page images")
    p.add_argument("--tiers", default=settings.tiers,
                   help="model escalation ladder, cheapest first (default: %(default)s)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=settings.claude_effort,
                   help="Claude effort level (default: %(default)s)")
    p.add_argument("--pages", help="page selection, e.g. '1-20' or '3,5,9-12' (1-indexed)")
    p.add_argument("--brand", default="", help="brand name; scopes SKU matching in the master catalog")
    p.add_argument("--output", type=Path, default=settings.output_dir, help="output root (default: %(default)s)")
    p.add_argument("--job-id", help="output sub-folder name (default: random UUID)")
    p.add_argument("--concurrency", type=int, default=settings.max_concurrent_pages)
    p.add_argument("--no-skip", action="store_true", help="process every page (disable cover/TOC skip heuristic)")
    p.add_argument("--no-bg-removal", action="store_true", help="keep crops opaque")
    p.add_argument("--no-ocr", action="store_true", help="do not OCR scanned pages")
    p.add_argument("--no-catalog-update", action="store_true",
                   help="diff against the master catalog without writing to it (dry run)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _page_spec(pages: list[int]) -> str:
    """[3,4,5,9] → '3-5,9'"""
    pages, parts = sorted(set(pages)), []
    i = 0
    while i < len(pages):
        j = i
        while j + 1 < len(pages) and pages[j + 1] == pages[j] + 1:
            j += 1
        parts.append(str(pages[i]) if i == j else f"{pages[i]}-{pages[j]}")
        i = j + 1
    return ",".join(parts)


async def _main(args: argparse.Namespace) -> int:
    opts = PipelineOptions(
        tiers=args.tiers,
        effort=args.effort,
        pages=args.pages,
        skip_non_product=not args.no_skip,
        remove_background=not args.no_bg_removal,
        use_ocr=not args.no_ocr,
        brand=args.brand,
        update_catalog=not args.no_catalog_update,
        max_concurrent=args.concurrency,
        output_root=args.output.resolve(),
    )

    job_id = args.job_id or str(uuid.uuid4())
    task = asyncio.create_task(run_pipeline(args.input, opts, job_id=job_id,
                                            on_progress=lambda j: print(f"» {j.stage}", flush=True)))
    last = None
    while not task.done():
        await asyncio.sleep(1.5)
        job = job_store.get(job_id)
        if job and job.total_pages and (job.processed_pages, job.total_pages) != last:
            last = (job.processed_pages, job.total_pages)
            print(f"  pages {job.processed_pages}/{job.total_pages}", flush=True)
    job = await task

    print()
    if job.status == "error":
        print(f"FAILED: {job.error_message}", file=sys.stderr)
        return 1

    print(f"Done in {job.elapsed_seconds}s")
    print(f"  document pages : {job.document_pages}  (processed {job.total_pages}, skipped {len(job.skipped_pages)})")
    print(f"  products       : {job.total_products}  validated {job.validated_products}, "
          f"quarantined {job.exception_products}")
    print(f"  catalog diff   : {job.new_products} new, {job.updated_products} updated, "
          f"{job.unchanged_products} unchanged")
    per_page = job.cost_inr / job.total_pages if job.total_pages else 0
    print(f"  AI cost        : ${job.cost_usd:.4f}  (₹{job.cost_inr:.2f}, ₹{per_page:.3f}/page)")
    for t in job.tier_stats:
        extra = f"  DISABLED: {t.disabled_reason}" if t.disabled_reason else ""
        print(f"    {t.tier:<34} attempted {t.pages_attempted:>3}  accepted {t.pages_accepted:>3}  "
              f"${t.cost_usd:.4f}{extra}")
    rerun = [p.page_number for p in job.pages if p.needs_rerun]
    missed = [w for w in job.page_warnings if w.reason.startswith("possible missed")]
    if missed:
        print(f"  page warnings  : {len(missed)} page(s) with possibly missed products (see exceptions.log)")
    if rerun:
        print(f"  needs re-run   : {len(rerun)} page(s) could not escalate (provider quota / credits):")
        print(f"                   python -m app {args.input} --pages {_page_spec(rerun)}"
              + (f" --brand {args.brand}" if args.brand else ""))
    if job.page_errors:
        print(f"  page errors    : {len(job.page_errors)} (see exceptions.log) — first: {job.page_errors[0].reason}")
        print(f"  re-run them    : python -m app {args.input} --pages {_page_spec([i.page_number for i in job.page_errors])}"
              + (f" --brand {args.brand}" if args.brand else ""))
    print(f"  outputs        : {Path(job.output_excel).parent}")
    for f in (job.output_excel, job.output_json, job.output_exceptions, job.output_log):
        print(f"                   {Path(f).name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows code pages can't encode "»" / "—"
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
