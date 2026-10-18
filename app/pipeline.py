"""
Pipeline Orchestrator – ties all stages together.

Stage 1 text extraction runs up-front (cheap, local) so non-product pages are skipped before any
model spend. Each remaining page then climbs a cost ladder of vision models ("tiers"):

    cheap model ──verify──► accepted
         │ verification failed / malformed output / unattributed prices / provider down
         ▼
    stronger model ──verify──► accepted
         │ …
         ▼
    premium model ──verify──► accepted or quarantined

Accuracy never depends on which tier answered: every record is gated by the same deterministic
verifier. Escalation stops early when two consecutive tiers return the same values and the same
failures (the problem is in the page — e.g. unreadable text — not the model), so premium spend
goes only where a stronger model can change the outcome.

Blocking work (rendering, OCR, background removal) runs in worker threads so the event loop —
and the web UI — stay responsive. Live progress is published into the shared `job_store` and
persisted to <job_dir>/job.json.
"""
from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from app.catalog_store import diff_and_update, resolve_duplicates
from app.config import cost_usd, settings
from app.image_processor import CropSource, crop_and_clean
from app.output_writer import (export_clean_json, export_excel, export_exceptions,
                               export_exceptions_log)
from app.pdf_renderer import (PRICE_SIGNAL, PageText, analyze_pages, enumerate_pages, ocr_region,
                              parse_page_range, render_page)
from app.schemas import (CROPPABLE_TYPES, ExtractedProduct, JobStatus, PageIssue, PageResult,
                         ProcessedProduct, TierAttempt, TierStats, ValidationResult)
from app.verifier import (bbox_problems, cross_product_checks, title_region,
                          unexplained_price_labels, verify_product)
from app.vlm_extractor import BaseExtractor, ExtractionError, FatalExtractionError, create_extractor

logger = logging.getLogger(__name__)

# ── In-memory job store (keyed by job_id) ─────────────────────────────────────
job_store: Dict[str, JobStatus] = {}

PREVIEW_LONG_EDGE = 1400


@dataclass
class PipelineOptions:
    tiers: Optional[str] = None            # "provider:model,…"; default settings.tiers
    api_keys: Dict[str, str] = field(default_factory=dict)   # per-provider overrides
    effort: str = field(default_factory=lambda: settings.claude_effort)
    thinking: str = field(default_factory=lambda: settings.gemini_thinking)
    pages: Optional[str] = None            # e.g. "3-20" (1-indexed); None = all
    skip_non_product: bool = True
    remove_background: bool = True
    use_ocr: bool = True
    brand: str = ""
    update_catalog: bool = True
    max_concurrent: int = field(default_factory=lambda: settings.max_concurrent_pages)
    dpi: int = field(default_factory=lambda: settings.render_dpi)
    pass_threshold: float = field(default_factory=lambda: settings.pass_threshold)
    min_image_px: int = field(default_factory=lambda: settings.min_image_px)
    max_cost_usd: float = field(default_factory=lambda: settings.max_job_cost_usd)
    output_root: Path = field(default_factory=lambda: settings.output_dir)
    catalog_path: Optional[Path] = None    # default: <output_root>/master_catalog.json


class Tier:
    """One rung of the cost ladder: an extractor plus its live usage statistics."""

    def __init__(self, extractor: BaseExtractor):
        self.extractor = extractor
        self.name = f"{extractor.provider}:{extractor.model}"
        self.stats = TierStats(tier=self.name)

    @property
    def available(self) -> bool:
        return self.stats.disabled_reason is None

    def extractor_cost(self) -> float:
        return cost_usd(self.extractor.model, self.extractor.input_tokens, self.extractor.output_tokens)

    def sync_usage(self) -> None:
        self.stats.input_tokens = self.extractor.input_tokens
        self.stats.output_tokens = self.extractor.output_tokens
        self.stats.cost_usd = round(cost_usd(self.extractor.model, self.stats.input_tokens,
                                             self.stats.output_tokens), 6)


@dataclass
class _Attempt:
    tier: Tier
    products: List[ExtractedProduct]
    results: List[ValidationResult]
    invalid: List[dict]
    escalation: Optional[str]

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.results)

    def signature(self) -> tuple:
        """What the model claimed — used to detect two tiers agreeing."""
        return tuple(sorted(
            (re.sub(r"\W+", "", p.product_name.lower()), p.mrp, p.b2b_price, p.moq) for p in self.products))


def build_tiers(opts: PipelineOptions) -> List[Tier]:
    tiers: List[Tier] = []
    missing = []
    for provider, model in settings.tier_list(opts.tiers):
        key = opts.api_keys.get(provider) or settings.api_key_for(provider)
        if not key:
            missing.append(f"{provider}:{model}")
            continue
        tiers.append(Tier(create_extractor(provider, key, model, opts.effort, opts.thinking)))
    if not tiers:
        raise ValueError("No usable model tiers — set GEMINI_API_KEY and/or ANTHROPIC_API_KEY in .env "
                         f"(tiers without a key: {', '.join(missing) or 'none configured'})")
    if missing:
        logger.warning("Skipping tiers without an API key: %s", ", ".join(missing))
    return tiers


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (s or "item")[:60]


class _ImageNamer:
    """Allocates `{sku_or_id}_{index}.png` names (PRD §5 step 4) without collisions."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}

    def next(self, ident: str) -> str:
        base = _slug(ident)
        n = self._counts.get(base.lower(), 0)
        self._counts[base.lower()] = n + 1
        return f"{base}_{n}.png"


def _escalation_reason(page: PageText, products: List[ExtractedProduct],
                       results: List[ValidationResult], invalid: List[dict]) -> Optional[str]:
    """Why this page's extraction should go to a stronger model (None = accept)."""
    if invalid:
        return f"{len(invalid)} malformed record(s)"
    if page.text_source == "none":
        return None  # nothing to verify against; a stronger model cannot change that
    failed = [r for r in results if not r.passed]
    if failed:
        return f"{len(failed)} of {len(results)} product(s) failed verification"
    missing = unexplained_price_labels(page.raw_text, products, page.text_source == "ocr")
    if missing:
        return "unattributed prices on page: " + ", ".join(f"{v:g}" for v in missing[:5])
    if not products and PRICE_SIGNAL.search(page.raw_text) and re.search(r"\d", page.raw_text):
        return "no products returned for a page with price indicators"
    return None


async def _verify_all(page: PageText, products: List[ExtractedProduct], pass_threshold: float,
                      ocr_cache: Dict[tuple, list]) -> List[ValidationResult]:
    n = len(products)
    results = [verify_product(p, page.raw_text, page.text_blocks, page.text_source, pass_threshold,
                              products_on_page=n, page_products=products) for p in products]
    # OCR pages: a title the full-page pass could not read gets a high-magnification second look.
    if page.text_source == "ocr":
        for i, (p, r) in enumerate(zip(products, results)):
            if r.passed or not r.text_only_failure:
                continue
            region = tuple(round(v, -1) for v in title_region(p, page.text_blocks, True))
            if region not in ocr_cache:
                ocr_cache[region] = await asyncio.to_thread(ocr_region, page.ref, region)
            extra = ocr_cache[region]
            if not extra:
                continue
            retry = verify_product(p, page.raw_text + "\n" + "\n".join(b.text for b in extra),
                                   list(page.text_blocks) + extra, page.text_source, pass_threshold,
                                   products_on_page=n, page_products=products)
            if retry.passed:
                retry.failure_reasons.append("warning: title confirmed by high-resolution OCR of the title area")
                results[i] = retry
    cross_product_checks(products, results)
    return results


def _accept_by_consensus(attempt: "_Attempt", previous: "_Attempt") -> bool:
    """Two different models returned identical products; if the only open issue is unreadable
    title text (every number verified), accept them with a visible warning."""
    if not settings.name_consensus or attempt.invalid:
        return False
    if attempt.signature() != previous.signature():
        return False
    if not all(r.passed or r.text_only_failure for r in attempt.results):
        return False
    for r in attempt.results:
        if not r.passed:
            r.passed = True
            r.consensus = f"{previous.tier.name} + {attempt.tier.name}"
            r.failure_reasons = [f if f.startswith("warning") else f"warning: {f}" for f in r.failure_reasons]
            r.failure_reasons.append(f"warning: title text not readable by OCR — name confirmed by agreement "
                                     f"of {previous.tier.name} and {attempt.tier.name}")
    return True


async def _extract_with_ladder(
    page: PageText, image_bytes: bytes, tiers: List[Tier], opts: PipelineOptions,
) -> Tuple[Optional[_Attempt], List[TierAttempt]]:
    attempts: List[_Attempt] = []
    log: List[TierAttempt] = []
    ocr_cache: Dict[tuple, list] = {}
    for tier in tiers:
        if not tier.available:
            if attempts and attempts[-1].escalation:  # this page needed it
                log.append(TierAttempt(tier=tier.name, outcome=f"skipped: tier disabled ({tier.stats.disabled_reason})"))
            continue
        spent = sum(t.extractor_cost() for t in tiers)
        if opts.max_cost_usd and spent >= opts.max_cost_usd:
            log.append(TierAttempt(tier=tier.name, outcome=f"skipped: job cost cap ${opts.max_cost_usd:g} reached"))
            break
        tier.stats.pages_attempted += 1
        try:
            extraction = await tier.extractor.extract_page(image_bytes, page.page_number,
                                                           page.raw_text, page.text_source)
        except FatalExtractionError as exc:
            tier.stats.disabled_reason = str(exc)
            log.append(TierAttempt(tier=tier.name, outcome=f"tier disabled: {exc}"))
            continue
        except ExtractionError as exc:
            log.append(TierAttempt(tier=tier.name, outcome=f"error: {exc}"))
            continue
        finally:
            tier.sync_usage()

        results = await _verify_all(page, extraction.products, opts.pass_threshold, ocr_cache)
        reason = _escalation_reason(page, extraction.products, results, extraction.invalid_records)
        attempt = _Attempt(tier, extraction.products, results, extraction.invalid_records, reason)
        attempts.append(attempt)
        if reason is None:
            log.append(TierAttempt(tier=tier.name, outcome="accepted"))
            break
        if len(attempts) >= 2 and _accept_by_consensus(attempt, attempts[-2]):
            attempt.escalation = reason = _escalation_reason(page, attempt.products, attempt.results, attempt.invalid)
            if reason is None:
                log.append(TierAttempt(tier=tier.name, outcome=f"accepted by agreement with {attempts[-2].tier.name} "
                                                               "(title text not readable by OCR)"))
                break
        if len(attempts) >= 2 and attempt.signature() == attempts[-2].signature():
            log.append(TierAttempt(tier=tier.name, outcome=f"{reason} — same result as previous tier, "
                                                           "not escalating further"))
            break
        log.append(TierAttempt(tier=tier.name, outcome=f"escalated: {reason}"))

    if not attempts:
        return None, log
    # Keep the attempt that validated most products (ties → fewer failures → stronger tier).
    best = max(enumerate(attempts),
               key=lambda ia: (ia[1].passed, -(len(ia[1].results) - ia[1].passed), ia[0]))[1]
    best.tier.stats.pages_accepted += 1
    return best, log


def _capacity_blocked(log: List[TierAttempt], best: _Attempt) -> bool:
    """The page still needed a stronger model, but every remaining tier was unavailable
    (quota, credits, overload) — a capacity problem, not a verification result."""
    if best.escalation is None:
        return False
    blocked = ("error:", "tier disabled:", "skipped:")
    last_answer = max((i for i, a in enumerate(log) if not a.outcome.startswith(blocked)), default=-1)
    if last_answer >= 0 and not log[last_answer].outcome.startswith("escalated:"):
        return False  # the ladder ended on a real verdict (accepted / agreeing tiers)
    return any(a.outcome.startswith(blocked) for a in log[last_answer + 1:])


async def _process_page(
    page: PageText,
    tiers: List[Tier],
    opts: PipelineOptions,
    job: JobStatus,
    job_dir: Path,
    namer: _ImageNamer,
    semaphore: asyncio.Semaphore,
    invalid_records: List[dict],
) -> List[ProcessedProduct]:
    label = page.ref.label
    page_result = PageResult(page_number=page.page_number, source=label)
    job.pages.append(page_result)
    try:
        async with semaphore:
            if not any(t.available for t in tiers):
                reasons = "; ".join(f"{t.name}: {t.stats.disabled_reason}" for t in tiers)
                raise ExtractionError(f"not attempted — all model tiers disabled ({reasons})")
            rendered = await asyncio.to_thread(render_page, page, opts.dpi)
            preview_rel = f"pages/p{page.page_number}.jpg"
            await asyncio.to_thread(_save_preview, rendered.image_bytes, job_dir / preview_rel)
            page_result.preview = preview_rel
            best, log = await _extract_with_ladder(page, rendered.image_bytes, tiers, opts)
            page_result.attempts = log

        if best is None:
            raise ExtractionError("; ".join(f"{a.tier}: {a.outcome}" for a in log) or "no tier produced a result")
        page_result.tier_used = best.tier.name
        if _capacity_blocked(log, best):
            page_result.needs_rerun = True
            blocked = next(a.outcome for a in reversed(log)
                           if a.outcome.startswith(("error:", "tier disabled:", "skipped:")))
            job.page_warnings.append(PageIssue(
                page_number=page.page_number, source=label,
                reason=f"needs re-run — escalation to a stronger model was blocked ({blocked[:160]})"))
            for r in best.results:
                if not r.passed:
                    r.failure_reasons.append("not escalated: stronger model unavailable (quota / credits) — re-run page")

        for bad in best.invalid:
            invalid_records.append({"page": page.page_number, "source": label, **bad})
        missing = unexplained_price_labels(page.raw_text, best.products, page.text_source == "ocr")
        if missing and page.text_source != "none":
            job.page_warnings.append(PageIssue(
                page_number=page.page_number, source=label,
                reason="possible missed product — price(s) on page not attributed to any extracted product: "
                       + ", ".join(f"{v:g}" for v in missing[:5])))

        source = CropSource(page.ref.source, page.ref.index, page.ref.is_image)
        products: List[ProcessedProduct] = []
        for product, validation in zip(best.products, best.results):
            pp = ProcessedProduct(page_number=page.page_number, source=label, tier=best.tier.name,
                                  extracted=product, validation=validation)
            if validation.passed:
                ident = product.sku_or_model or product.product_name
                for img in product.images:
                    if img.type not in CROPPABLE_TYPES or \
                            bbox_problems(img.box_2d, len(best.products), img.type == "variant"):
                        continue
                    name = namer.next(ident)
                    crop = await asyncio.to_thread(
                        crop_and_clean, source, img.box_2d, job_dir / "images" / name,
                        0.03, opts.remove_background, settings.rembg_model, opts.min_image_px,
                        3, settings.max_image_px,
                    )
                    if crop.path:
                        pp.image_paths.append(f"images/{name}")
                        if crop.original_path:
                            pp.original_image_paths.append(f"images/{crop.original_path.name}")
                        pp.background_removed = crop.background_removed if pp.background_removed is None \
                            else pp.background_removed and crop.background_removed
                        if crop.upscale > 1.05:
                            pp.image_warnings.append(
                                f"{name}: source photo is low resolution — upscaled {crop.upscale:.1f}×")
                    if crop.warning:
                        pp.image_warnings.append(f"{name}: {crop.warning}")
            products.append(pp)

        page_result.products = len(products)
        page_result.validated = sum(p.validation.passed for p in products)
        return products

    except ExtractionError as exc:
        job.page_errors.append(PageIssue(page_number=page.page_number, source=label, reason=str(exc)))
        logger.error("[%s] page %d: %s", job.job_id[:8], page.page_number, exc)
        return []
    except Exception as exc:  # never let one page take down the batch
        logger.exception("[%s] page %d failed", job.job_id[:8], page.page_number)
        job.page_errors.append(PageIssue(page_number=page.page_number, source=label,
                                         reason=f"{type(exc).__name__}: {exc}"))
        return []
    finally:
        job.processed_pages += 1
        _update_cost(job, tiers)


def _save_preview(png_bytes: bytes, path: Path) -> None:
    with Image.open(io.BytesIO(png_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE), Image.LANCZOS)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, "JPEG", quality=82, optimize=True)


def _update_cost(job: JobStatus, tiers: List[Tier]) -> None:
    for t in tiers:
        t.sync_usage()
    job.tier_stats = [t.stats.model_copy() for t in tiers]
    job.cost_usd = round(sum(t.stats.cost_usd for t in tiers), 6)
    job.cost_inr = round(job.cost_usd * settings.usd_to_inr, 2)


def save_job(job: JobStatus, output_root: Path) -> None:
    path = output_root / job.job_id / "job.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(job.model_dump_json(indent=2), encoding="utf-8")


def load_jobs(output_root: Path) -> None:
    """Restore finished jobs from disk so results survive a server restart."""
    for path in output_root.glob("*/job.json"):
        try:
            job = JobStatus.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.status in ("queued", "processing"):
            job.status, job.error_message = "error", "server restarted while this job was running"
        job.cost_inr = round(job.cost_usd * settings.usd_to_inr, 2)  # display at the current rate
        job_store.setdefault(job.job_id, job)


async def run_pipeline(
    input_path: str | Path,
    options: Optional[PipelineOptions] = None,
    job_id: Optional[str] = None,
    tiers: Optional[List[BaseExtractor]] = None,
    on_progress: Optional[Callable[[JobStatus], None]] = None,
) -> JobStatus:
    """
    Full pipeline entry point. Processes a PDF (or folder of pages) and writes outputs to
    <output_root>/<job_id>/. `tiers` injects extractors (tests); otherwise they are built from
    the options / settings. Returns the final JobStatus (also kept in job_store).
    """
    opts = options or PipelineOptions()
    job_id = job_id or str(uuid.uuid4())
    input_path = Path(input_path)
    started = time.monotonic()

    queued = job_store.get(job_id)
    job = JobStatus(job_id=job_id, status="processing", stage="Reading pages", brand=opts.brand,
                    filename=queued.filename if queued else input_path.name,
                    created_at=queued.created_at if queued and queued.created_at
                    else datetime.now(timezone.utc).isoformat(timespec="seconds"))
    job_store[job_id] = job
    job_dir = opts.output_root / job_id

    def progress(stage: str) -> None:
        job.stage = stage
        if on_progress:
            on_progress(job)

    try:
        ladder = [Tier(e) for e in tiers] if tiers else build_tiers(opts)
        job.tiers = " → ".join(t.name for t in ladder)

        # ── Stage 1: enumerate pages, extract text, skip non-product pages ──
        refs = await asyncio.to_thread(enumerate_pages, input_path)
        job.document_pages = len(refs)
        selected = parse_page_range(opts.pages, len(refs))
        if selected is not None:
            refs = [r for r in refs if r.page_number in selected]
        progress(f"Reading {len(refs)} page(s)")
        pages = await asyncio.to_thread(analyze_pages, refs, opts.use_ocr)

        active: List[PageText] = []
        for p in pages:
            if opts.skip_non_product and p.should_skip:
                job.skipped_pages.append(PageIssue(page_number=p.page_number, source=p.ref.label,
                                                   reason=p.skip_reason or ""))
            else:
                active.append(p)
        job.total_pages = len(active)
        logger.info("[%s] %d page(s) to process, %d skipped.", job_id[:8], len(active), len(job.skipped_pages))

        # ── Stage 2-4: extract (tier ladder), verify, crop — concurrently per page ──
        progress("Extracting and verifying products")
        semaphore = asyncio.Semaphore(max(1, opts.max_concurrent))
        namer = _ImageNamer()
        invalid_records: List[dict] = []
        results = await asyncio.gather(*[
            _process_page(p, ladder, opts, job, job_dir, namer, semaphore, invalid_records) for p in active
        ])
        all_products = [pp for page_products in results for pp in page_products]
        job.page_errors.sort(key=lambda i: i.page_number)
        job.pages.sort(key=lambda p: p.page_number)
        _update_cost(job, ladder)

        if active and len(job.page_errors) == len(active):
            raise RuntimeError(f"All {len(active)} page(s) failed. First error: {job.page_errors[0].reason}")

        # ── Stage 5: dedupe within the run, diff against the master catalog ─
        progress("Comparing with earlier uploads and building the Excel / image files")
        validated = [p for p in all_products if p.validation.passed]
        exceptions = [p for p in all_products if not p.validation.passed]
        validated, conflicting = resolve_duplicates(validated, opts.brand)
        exceptions.extend(conflicting)
        validated.sort(key=lambda p: p.page_number)
        exceptions.sort(key=lambda p: p.page_number)

        catalog_path = opts.catalog_path or (opts.output_root / "master_catalog.json")
        await asyncio.to_thread(diff_and_update, validated, catalog_path, job_id, input_path.name,
                                opts.brand, opts.update_catalog)

        job.total_products = len(validated) + len(exceptions)
        job.validated_products = len(validated)
        job.exception_products = len(exceptions)
        job.new_products = sum(p.change_status == "new" for p in validated)
        job.updated_products = sum(p.change_status == "updated" for p in validated)
        job.unchanged_products = sum(p.change_status == "unchanged" for p in validated)

        job_dir.mkdir(parents=True, exist_ok=True)
        job.output_excel = str(await asyncio.to_thread(
            export_excel, validated, exceptions, job_dir / "catalog_clean.xlsx"))
        job.output_json = str(export_clean_json(validated, job, job_dir / "catalog_clean.json"))
        job.output_exceptions = str(export_exceptions(exceptions, invalid_records, job, job_dir / "exceptions.json"))
        job.output_log = str(export_exceptions_log(exceptions, validated, invalid_records, job,
                                                   job_dir / "exceptions.log"))

        job.status = "done"
        job.elapsed_seconds = round(time.monotonic() - started, 1)
        progress("Complete")
        logger.info("[%s] Done in %.1fs. Validated: %d, Exceptions: %d, Page errors: %d, cost $%.4f",
                    job_id[:8], job.elapsed_seconds, len(validated), len(exceptions), len(job.page_errors),
                    job.cost_usd)

    except Exception as exc:
        logger.exception("[%s] Pipeline failed: %s", job_id[:8], exc)
        job.status = "error"
        job.error_message = str(exc)
        job.elapsed_seconds = round(time.monotonic() - started, 1)
        progress("Failed")

    try:
        save_job(job, opts.output_root)
    except Exception:
        logger.exception("[%s] could not persist job.json", job_id[:8])
    return job
