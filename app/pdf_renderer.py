"""
Stage 1 – Page Renderer & Text Stream Extractor.
Uses PyMuPDF (fitz) to:
  - Enumerate pages from a PDF file, or a folder of PDFs / page images.
  - Extract native text blocks with bounding coordinates (cheap, done up-front).
  - Fall back to Tesseract OCR (via PyMuPDF) for scanned pages, when installed.
  - Apply a heuristic to detect and skip non-product pages (cover, TOC, index, …).
  - Render a page to a high-res PNG at 300 DPI (lazily, only for pages that are processed).
"""
from __future__ import annotations

import functools
import hashlib
import io
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pymupdf as fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

# A page carrying any of these is never skipped.
PRICE_SIGNAL = re.compile(r"[₹$€£¥]|\b(?:rs|inr|mrp|price|moq|usd|eur|gst|hsn)\b", re.IGNORECASE)
SKU_SIGNAL = re.compile(r"\b[A-Z]{1,6}[-/]?\d{2,}[A-Z0-9-]*\b")

# Titles that mark non-product pages (matched against the first lines only).
SKIP_TITLES = (
    "table of contents",
    "contents",
    "index",
    "about us",
    "our story",
    "contact us",
    "terms and conditions",
    "terms & conditions",
    "thank you",
)

# Native text shorter than this means the page has no usable digital text layer.
MIN_NATIVE_TEXT_CHARS = 50


@dataclass
class TextBlock:
    """A text block with its bbox normalized to the 0-1000 page space (same as VLM boxes)."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PageRef:
    page_number: int           # 1-indexed across the whole input
    source: Path               # file the page lives in
    index: int = 0             # page index inside `source` (0 for image files)

    @property
    def is_image(self) -> bool:
        return self.source.suffix.lower() in IMAGE_EXTENSIONS

    @property
    def label(self) -> str:
        return self.source.name if self.is_image else f"{self.source.name}#p{self.index + 1}"


@dataclass
class PageText:
    ref: PageRef
    text_blocks: List[TextBlock] = field(default_factory=list)
    raw_text: str = ""
    text_source: str = "native"      # native | ocr | none
    skip_reason: Optional[str] = None

    @property
    def page_number(self) -> int:
        return self.ref.page_number

    @property
    def is_scanned(self) -> bool:
        return self.text_source != "native"

    @property
    def should_skip(self) -> bool:
        return self.skip_reason is not None


@dataclass
class RenderedPage:
    text: PageText
    image_bytes: bytes         # PNG bytes at render DPI
    width_px: int
    height_px: int

    @property
    def page_number(self) -> int:
        return self.text.page_number


# ── Input enumeration ────────────────────────────────────────────────────────

def enumerate_pages(input_path: str | Path) -> List[PageRef]:
    """
    List every page of the input: a single PDF, a single page image,
    or a folder containing PDFs and/or page images (processed in name order).
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir()
             if p.is_file() and (p.suffix.lower() == ".pdf" or p.suffix.lower() in IMAGE_EXTENSIONS)),
            key=lambda p: _natural_key(p.name),
        )
        if not files:
            raise ValueError(f"No PDF or image files found in folder: {path}")
        files = _drop_duplicate_files(files)
    else:
        files = [path]

    refs: List[PageRef] = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            refs.append(PageRef(page_number=len(refs) + 1, source=f, index=0))
        elif suffix == ".pdf":
            with fitz.open(str(f)) as doc:
                if doc.needs_pass:
                    raise ValueError(f"PDF is password-protected: {f.name}")
                for i in range(doc.page_count):
                    refs.append(PageRef(page_number=len(refs) + 1, source=f, index=i))
        else:
            raise ValueError(f"Unsupported input type: {f.name} (expected .pdf or an image)")
    return refs


def _drop_duplicate_files(files: List[Path]) -> List[Path]:
    """Skip byte-identical copies (e.g. 'catalog (1).pdf') so they are not re-processed."""
    seen: dict[str, Path] = {}
    # Visit originals before "name (1).pdf"-style copies so the original is the one kept.
    for f in sorted(files, key=lambda f: bool(re.search(r"\(\d+\)$", f.stem))):
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        if digest in seen:
            logger.warning("Skipping %s: identical to %s", f.name, seen[digest].name)
            continue
        seen[digest] = f
    kept = set(seen.values())
    return [f for f in files if f in kept]


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def parse_page_range(spec: Optional[str], total: int) -> Optional[set[int]]:
    """Parse '1-5,8,10-12' into a set of 1-indexed page numbers (None = all pages)."""
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a or 1), int(b or total)
            pages.update(range(max(1, start), min(total, end) + 1))
        else:
            n = int(part)
            if 1 <= n <= total:
                pages.add(n)
    return pages


# ── OCR availability ─────────────────────────────────────────────────────────

_TESSDATA_CANDIDATES = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tessdata",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tessdata",
    Path("/usr/share/tesseract-ocr/5/tessdata"),
    Path("/usr/share/tesseract-ocr/4.00/tessdata"),
    Path("/opt/homebrew/share/tessdata"),
    Path("/usr/local/share/tessdata"),
)


_TESSERACT_CANDIDATES = (
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
)
OCR_DPI = 300
# Tesseract languages, "+"-joined (e.g. "eng+hin"); each needs its traineddata file installed.
OCR_LANGS = os.environ.get("OCR_LANGS", "").strip() or "eng"
# psm 3 = automatic page layout; psm 11 = sparse text. Catalog pages put text over photos,
# where layout analysis drops lines that sparse mode finds — so both passes are merged.
OCR_MODES = (3, 11)


@functools.lru_cache(maxsize=1)
def tesseract_cmd() -> Optional[str]:
    """Path of a working Tesseract binary (preferred OCR engine), or None."""
    candidates = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                  *(str(c) for c in _TESSERACT_CANDIDATES if c.exists())]
    try:
        import pytesseract
    except ImportError:
        return None
    for cmd in filter(None, candidates):
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd
            pytesseract.get_tesseract_version()
            return cmd
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=1)
def _mupdf_ocr_available() -> bool:
    """PyMuPDF bundles a Tesseract engine; it only needs the tessdata language files."""
    if not os.environ.get("TESSDATA_PREFIX"):
        for cand in _TESSDATA_CANDIDATES:
            if (cand / "eng.traineddata").exists():
                os.environ["TESSDATA_PREFIX"] = str(cand)
                break
    try:
        fitz.get_tessdata()
        return True
    except Exception:
        return False


def ocr_available() -> bool:
    return tesseract_cmd() is not None or _mupdf_ocr_available()


def _tesseract_lines(img: Image.Image, psm: int) -> List[TextBlock]:
    import pytesseract
    data = pytesseract.image_to_data(img, lang=OCR_LANGS, config=f"--psm {psm}",
                                     output_type=pytesseract.Output.DICT)
    lines: dict = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        ln = lines.setdefault(key, {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
        ln["words"].append(word)
        ln["x0"], ln["y0"] = min(ln["x0"], x), min(ln["y0"], y)
        ln["x1"], ln["y1"] = max(ln["x1"], x + w), max(ln["y1"], y + h)
    W, H = img.width or 1, img.height or 1
    return [TextBlock(text=" ".join(ln["words"]), x0=ln["x0"] / W * 1000, y0=ln["y0"] / H * 1000,
                      x1=ln["x1"] / W * 1000, y1=ln["y1"] / H * 1000) for ln in lines.values()]


def ocr_region(ref: PageRef, region: tuple, dpi: int = 400) -> List[TextBlock]:
    """
    High-magnification OCR of one region (ymin, xmin, ymax, xmax in 0-1000) — used to read
    stylised titles / logos that full-page OCR misses. Plain and inverted passes are merged.
    """
    if not tesseract_cmd():
        return []
    from PIL import ImageOps
    ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, v)) for v in region)
    with fitz.open(str(ref.source)) as doc:
        page = doc[ref.index] if not ref.is_image else fitz.open("pdf", doc.convert_to_pdf())[0]
        r = page.rect
        clip = fitz.Rect(r.x0 + xmin / 1000 * r.width, r.y0 + ymin / 1000 * r.height,
                         r.x0 + xmax / 1000 * r.width, r.y0 + ymax / 1000 * r.height)
        img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi, clip=clip, alpha=False).tobytes("png"))).convert("L")
    blocks: List[TextBlock] = []
    for variant in (img, ImageOps.invert(img)):
        for b in _tesseract_lines(variant, 6):
            # map region-relative coordinates back to page space
            blocks.append(TextBlock(text=b.text,
                                    x0=xmin + b.x0 / 1000 * (xmax - xmin), x1=xmin + b.x1 / 1000 * (xmax - xmin),
                                    y0=ymin + b.y0 / 1000 * (ymax - ymin), y1=ymin + b.y1 / 1000 * (ymax - ymin)))
    return blocks


def _ocr_blocks(page: fitz.Page) -> List[TextBlock]:
    if tesseract_cmd():
        pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        blocks: List[TextBlock] = []
        seen: set[str] = set()
        for psm in OCR_MODES:
            for b in _tesseract_lines(img, psm):
                key = re.sub(r"\W+", "", b.text.lower())
                if len(key) >= 2 and key not in seen:
                    seen.add(key)
                    blocks.append(b)
        return sorted(blocks, key=lambda b: (round(b.y0 / 10), b.x0))
    tp = page.get_textpage_ocr(dpi=OCR_DPI, full=True, language=OCR_LANGS)
    return _blocks_from(page, textpage=tp)


def _open_page(doc_cache: dict, ref: PageRef) -> fitz.Page:
    key = str(ref.source)
    if key not in doc_cache:
        doc = fitz.open(str(ref.source))
        if ref.is_image:
            # Wrap the image as a 1-page PDF so OCR / rendering share one code path.
            doc = fitz.open("pdf", doc.convert_to_pdf())
        doc_cache[key] = doc
    return doc_cache[key][ref.index]


def _blocks_from(page: fitz.Page, textpage=None) -> List[TextBlock]:
    w, h = page.rect.width or 1, page.rect.height or 1
    blocks: List[TextBlock] = []
    for b in page.get_text("blocks", textpage=textpage, sort=True):
        if len(b) >= 5 and isinstance(b[4], str) and b[4].strip():
            blocks.append(TextBlock(
                text=b[4].strip(),
                x0=b[0] / w * 1000, y0=b[1] / h * 1000,
                x1=b[2] / w * 1000, y1=b[3] / h * 1000,
            ))
    return blocks


# ── Text extraction & skip heuristic ─────────────────────────────────────────

def skip_reason_for(raw_text: str, page_number: int, text_source: str) -> Optional[str]:
    """
    Heuristic (PRD §5 step 1): skip cover / TOC / index pages and low-density pages
    with no price indicators. Pages with any price or SKU signal are never skipped,
    and pages with no readable text at all are never skipped (we cannot judge them).
    """
    if text_source == "none":
        return None
    text = raw_text.strip()
    if PRICE_SIGNAL.search(text) and re.search(r"\d", text):
        return None
    if SKU_SIGNAL.search(text):
        return None

    head_lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()][:3]
    for line in head_lines:
        for title in SKIP_TITLES:
            if line == title or line.startswith(title + " ") or line.startswith(title + ":"):
                return f"non-product page ('{title}' heading, no price/SKU indicators)"

    word_count = len(text.split())
    if page_number == 1 and word_count < 80:
        return "cover page (page 1, little text, no price/SKU indicators)"
    if word_count < 30:
        return f"low text density ({word_count} words) and no price/SKU indicators"
    return None


HYBRID_IMAGE_COVERAGE = 0.5   # share of the page covered by raster images
HYBRID_MAX_TEXT_CHARS = 400   # native text this short on an image page is a footer, not the content


def image_coverage(page: fitz.Page) -> float:
    """Approximate share of the page area covered by raster images (overlaps capped at 1)."""
    area = page.rect.width * page.rect.height or 1
    covered = 0.0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            covered += r.width * r.height
    return min(1.0, covered / area)


def _is_hybrid(page: fitz.Page, native_text: str) -> bool:
    """A mostly-image page whose native text carries no prices or is only a few lines."""
    if image_coverage(page) < HYBRID_IMAGE_COVERAGE:
        return False
    has_prices = bool(PRICE_SIGNAL.search(native_text) and re.search(r"\d", native_text))
    return not has_prices or len(native_text.strip()) < HYBRID_MAX_TEXT_CHARS


def extract_page_text(ref: PageRef, use_ocr: bool = True, doc_cache: Optional[dict] = None) -> PageText:
    """Extract native text (or OCR text for scanned pages) and decide whether to skip."""
    own_cache = doc_cache is None
    doc_cache = {} if own_cache else doc_cache
    try:
        page = _open_page(doc_cache, ref)
        blocks = [] if ref.is_image else _blocks_from(page)
        raw_text = "\n".join(b.text for b in blocks)
        text_source = "native"

        if len(raw_text.strip()) < MIN_NATIVE_TEXT_CHARS:
            text_source = "none"
            if use_ocr and ocr_available():
                try:
                    blocks = _ocr_blocks(page)
                    raw_text = "\n".join(b.text for b in blocks)
                    text_source = "ocr"
                except Exception as exc:
                    logger.warning("OCR failed on %s: %s", ref.label, exc)
            if text_source == "none":
                blocks, raw_text = [], ""
        elif use_ocr and _is_hybrid(page, raw_text) and ocr_available():
            # Flattened page carrying only a thin text layer (footer, page number, legal line):
            # the product text is in the image, so OCR it and keep the native lines too.
            try:
                blocks = blocks + _ocr_blocks(page)
                raw_text = "\n".join(b.text for b in blocks)
                text_source = "ocr"
            except Exception as exc:
                logger.warning("OCR failed on %s: %s", ref.label, exc)

        return PageText(
            ref=ref,
            text_blocks=blocks,
            raw_text=raw_text,
            text_source=text_source,
            skip_reason=skip_reason_for(raw_text, ref.page_number, text_source),
        )
    finally:
        if own_cache:
            for d in doc_cache.values():
                d.close()


def analyze_pages(refs: List[PageRef], use_ocr: bool = True, workers: Optional[int] = None) -> List[PageText]:
    """Extract text for all pages. OCR runs Tesseract subprocesses, so pages are done in parallel
    (each worker opens its own document handle — PyMuPDF objects are not shared across threads)."""
    workers = workers or int(os.environ.get("OCR_WORKERS", "0") or 0) or min(4, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda r: extract_page_text(r, use_ocr=use_ocr), refs))


def render_page(page_text: PageText, dpi: int = 300) -> RenderedPage:
    """Render one page to PNG at `dpi` (images are used at native resolution)."""
    ref = page_text.ref
    if ref.is_image:
        with Image.open(ref.source) as img:
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            return RenderedPage(page_text, buf.getvalue(), rgb.width, rgb.height)

    with fitz.open(str(ref.source)) as doc:
        pix = doc[ref.index].get_pixmap(dpi=dpi, alpha=False)
        return RenderedPage(page_text, pix.tobytes("png"), pix.width, pix.height)
