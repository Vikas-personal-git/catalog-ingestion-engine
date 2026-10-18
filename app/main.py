"""
FastAPI server for Catalog Ingestion Engine.
Provides REST API endpoints + SSR Jinja2 UI for testing.

Endpoints:
  GET  /                          → Upload form + recent jobs (SSR HTML)
  POST /jobs                      → Upload one or more PDFs, start pipeline, redirect to job page
  GET  /jobs/{job_id}             → Job progress / results page (SSR HTML, polls while running)
  GET  /api/jobs                  → List jobs (JSON)
  GET  /api/jobs/{id}             → Job status (JSON)
  GET  /api/jobs/{id}/results     → Validated products + exceptions (JSON)
  GET  /api/health                → Model tiers / OCR / background-removal readiness
  GET  /download/{id}/zip         → folder-structured hand-off package (see export_zip.py)
  GET  /download/{id}/{kind}      → excel | json | exceptions | log
  GET  /output/{job_id}/...       → Output files (images, page previews)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import MODEL_PRICING, settings
from app.export_zip import build_zip
from app.image_processor import rembg_status
from app.pdf_renderer import ocr_available
from app.pipeline import PipelineOptions, job_store, load_jobs, run_pipeline
from app.schemas import JobStatus

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Catalog Ingestion Engine", version="2.0.0")

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(settings.output_dir)), name="output")
load_jobs(settings.output_dir)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def customer_text(text: str) -> str:
    """Quality-check messages without internal model names (presentation mode)."""
    text = re.sub(r"agreement of \S+:\S+ and \S+:\S+", "agreement of two independent AI models", text)
    return re.sub(r"\b(?:gemini|claude):[\w.-]+", "an AI model", text)


def is_internal_note(text: str) -> bool:
    return text.startswith("not escalated")


templates.env.filters["customer_text"] = customer_text
templates.env.tests["internal_note"] = is_internal_note

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
_background_tasks: set[asyncio.Task] = set()

DOWNLOADS = {
    "excel": ("output_excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    "json": ("output_json", "application/json", "json"),
    "exceptions": ("output_exceptions", "application/json", "json"),
    "log": ("output_log", "text/plain; charset=utf-8", "log"),
}

def _processing_modes() -> dict:
    """Modes offered in the UI, derived from VLM_TIERS (cascade = production setting; others for comparison)."""
    tiers = [f"{provider}:{model}" for provider, model in settings.tier_list()]
    return {
        "cascade": ("Smart cascade — cheap model first, premium only when verification fails", None),
        "cheapest": (f"Cheapest model only ({tiers[0]})", tiers[0]),
        "premium": (f"Premium model only ({tiers[-1]})", tiers[-1]),
    }


MODES = _processing_modes()


def _safe_filename(name: Optional[str]) -> str:
    name = Path(name or "catalog.pdf").name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or "catalog.pdf"
    return name[:120]


def _file_kind(content: bytes) -> Optional[str]:
    """Extension for supported uploads, detected from the bytes (not the file name)."""
    if content.startswith(b"%PDF"):
        return ".pdf"
    if content.startswith(b"\x89PNG"):
        return ".png"
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    if content[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    return None


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _read_json(path: Optional[str]) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def _view_mode(request: Request) -> Tuple[bool, Optional[str]]:
    """(demo?, value to remember). ?view=demo / ?view=internal switches and is kept in a cookie."""
    q = request.query_params.get("view")
    if q in ("demo", "internal"):
        return q == "demo", q
    c = request.cookies.get("view")
    if c in ("demo", "internal"):
        return c == "demo", None
    return settings.demo_mode, None


def _render(request: Request, template: str, context: dict):
    demo, remember = _view_mode(request)
    response = templates.TemplateResponse(request, template, {**context, "demo": demo})
    if remember:
        response.set_cookie("view", remember, max_age=60 * 60 * 24 * 365, samesite="lax")
    return response


SESSION_JOBS_COOKIE = "session_jobs"


def _session_job_ids(request: Request) -> List[str]:
    """Jobs started from this browser session (presentation mode shows only these)."""
    raw = request.cookies.get(SESSION_JOBS_COOKIE, "")
    return [j for j in raw.split(".") if j][:20]


def _recent_jobs(only: Optional[List[str]] = None) -> List[JobStatus]:
    load_jobs(settings.output_dir)  # pick up jobs finished by the CLI since startup
    jobs = job_store.values() if only is None else [job_store[j] for j in only if j in job_store]
    return sorted(jobs, key=lambda j: j.created_at or "", reverse=True)[:20]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    demo, _ = _view_mode(request)
    fresh = request.query_params.get("fresh") == "1"   # start a clean demo for the next company
    session_ids = [] if fresh else _session_job_ids(request)
    response = _render(request, "index.html", {
        "modes": MODES,
        "default_tiers": settings.tiers.replace(",", " → "),
        "has_claude_key": bool(settings.anthropic_api_key),
        "has_gemini_key": bool(settings.gemini_api_key),
        "jobs": _recent_jobs(session_ids if demo else None),
        "usd_inr": settings.usd_to_inr,
    })
    if fresh:
        response.delete_cookie(SESSION_JOBS_COOKIE)
    return response


@app.post("/jobs")
async def create_job(
    request: Request,
    files: List[UploadFile] = File(...),
    mode: str = Form(default="cascade"),
    pages: str = Form(default=""),
    brand: str = Form(default=""),
    skip_non_product: bool = Form(default=False),
    remove_background: bool = Form(default=False),
    folder_name: str = Form(default=""),
):
    if mode not in MODES:
        mode = "cascade"   # presentation mode hides the selector
    job_id = str(uuid.uuid4())
    job_upload_dir = settings.upload_dir / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    # Folder uploads arrive as "folder/sub/page01.pdf" in arbitrary order: keep path order.
    files = sorted(files, key=lambda u: _natural_key(u.filename or ""))
    names: List[str] = []
    skipped: List[str] = []
    total = 0
    for upload in files:
        content = await upload.read()
        ext = _file_kind(content)
        if ext is None:
            skipped.append(upload.filename or "?")
            continue
        total += len(content)
        if total > MAX_UPLOAD_BYTES:
            return HTMLResponse("<h2>Upload too large</h2><a href='/'>Back</a>", status_code=413)
        stem = Path(_safe_filename(upload.filename)).stem
        name = f"{len(names) + 1:04d}_{stem}{ext}"
        async with aiofiles.open(job_upload_dir / name, "wb") as f:
            await f.write(content)
        names.append(_safe_filename(upload.filename))
    if not names:
        return HTMLResponse("<h2>No supported files</h2><p>Upload PDF catalogs or page images "
                            "(PNG, JPG, WEBP, TIFF).</p><a href='/'>Back</a>", status_code=400)
    if skipped:
        logging.getLogger(__name__).info("[%s] skipped unsupported files: %s", job_id[:8], ", ".join(skipped))

    display = names[0] if len(names) == 1 else f"{names[0]} + {len(names) - 1} more"
    if folder_name.strip():
        display = f"{_safe_filename(folder_name)}/ ({len(names)} file{'s' if len(names) != 1 else ''})"
    job_store[job_id] = JobStatus(job_id=job_id, status="queued", filename=display,
                                  created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    demo, _ = _view_mode(request)
    opts = PipelineOptions(
        tiers=MODES[mode][1],
        pages=pages.strip() or None,
        brand=brand.strip(),
        skip_non_product=skip_non_product,
        remove_background=remove_background,
        update_catalog=not demo,   # a prospect's catalog never enters the shared master catalog
    )
    task = asyncio.create_task(run_pipeline(job_upload_dir, opts, job_id=job_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    response = RedirectResponse(url=f"/jobs/{job_id}", status_code=303)
    ids = [job_id] + _session_job_ids(request)
    response.set_cookie(SESSION_JOBS_COOKIE, ".".join(ids[:20]), samesite="lax")   # browser-session cookie
    return response


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str):
    if job_id not in job_store:
        load_jobs(settings.output_dir)
    job = job_store.get(job_id)
    if not job:
        return HTMLResponse("<h2>Job not found.</h2><a href='/'>Back</a>", status_code=404)
    done = job.status == "done"
    clean = _read_json(job.output_json).get("products", []) if done else []
    exc = _read_json(job.output_exceptions) if done else {}
    previews = {p.page_number: p.preview for p in job.pages if p.preview}
    escalated = sum(1 for p in job.pages if len(p.attempts) > 1)
    return _render(request, "job.html", {
        "job": job,
        "validated": clean,
        "exceptions": exc.get("products", []),
        "invalid_records": exc.get("invalid_vlm_records", []),
        "previews": previews,
        "escalated": escalated,
        "cost_per_page_inr": job.cost_inr / job.total_pages if job.total_pages else 0,
        "cost_per_product_inr": job.cost_inr / job.validated_products if job.validated_products else 0,
        "image_count": sum(len(p.get("image_paths", [])) for p in clean),
    })


@app.get("/api/health")
async def health():
    rembg_ok, rembg_detail = await asyncio.to_thread(rembg_status)
    tiers = []
    for provider, model in settings.tier_list():
        tiers.append({"tier": f"{provider}:{model}", "key_configured": bool(settings.api_key_for(provider)),
                      "price_usd_per_mtok": MODEL_PRICING.get(model)})
    return {
        "tiers": tiers,
        "ocr_available": ocr_available(),
        "rembg_available": rembg_ok,
        "rembg_detail": rembg_detail,
    }


@app.get("/api/jobs")
async def list_jobs():
    return [j.model_dump(exclude={"pages"}) for j in _recent_jobs()]


INTERNAL_FIELDS = {"pages", "tier_stats", "tiers", "cost_usd", "cost_inr"}


@app.get("/api/jobs/{job_id}")
async def job_status(request: Request, job_id: str):
    job = job_store.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    demo, _ = _view_mode(request)
    data = job.model_dump(exclude=INTERNAL_FIELDS if demo else {"pages"})
    if demo:
        data.update(tier_stats=[], cost_inr=0.0)   # keys the page script expects, without the values
    return data


@app.get("/api/jobs/{job_id}/results")
async def job_results(job_id: str):
    job = job_store.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    if job.status != "done":
        return JSONResponse({"error": f"Job is {job.status}"}, status_code=409)
    return {"validated": _read_json(job.output_json), "exceptions": _read_json(job.output_exceptions)}


@app.get("/download/{job_id}/zip")
async def download_zip(job_id: str):
    job = job_store.get(job_id)
    if not job or job.status != "done" or not job.output_json:
        return JSONResponse({"error": "Job is not finished"}, status_code=404)
    job_dir = Path(job.output_json).parent
    zip_path = job_dir / f"catalog_{job_id if len(job_id) <= 24 else job_id[:8]}.zip"
    source = Path(job.output_json)
    if not zip_path.exists() or zip_path.stat().st_mtime < source.stat().st_mtime:
        zip_path = await asyncio.to_thread(build_zip, job_dir, job.brand)
    return FileResponse(path=str(zip_path), media_type="application/zip", filename=zip_path.name)


@app.get("/download/{job_id}/{kind}")
async def download(job_id: str, kind: str):
    job = job_store.get(job_id)
    if kind not in DOWNLOADS:
        return JSONResponse({"error": f"Unknown download '{kind}'"}, status_code=404)
    attr, media_type, ext = DOWNLOADS[kind]
    path = getattr(job, attr, None) if job else None
    if not path or not Path(path).exists():
        return JSONResponse({"error": "File not available"}, status_code=404)
    stem = Path(path).stem
    return FileResponse(path=path, media_type=media_type, filename=f"{stem}_{job_id[:8]}.{ext}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level=settings.log_level)
