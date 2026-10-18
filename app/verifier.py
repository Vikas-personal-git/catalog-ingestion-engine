"""
Stage 3 – Deterministic Accuracy Verification Engine.
Runs programmatic validation against extracted product records before they are
promoted to the clean catalog. Nothing here trusts the VLM.

Checks performed:
  1. Verbatim text check – raw_text_snippet exists in the page's native (or OCR) text.
  2. Numeric cross-check – MRP / b2b_price / MOQ / slab values appear as exact number tokens
     in the text blocks surrounding the product (snippet + SKU text + image boxes).
  3. Price logic rules – prices positive, b2b_price <= mrp, slabs consistent with moq/b2b_price.
  4. Bounding box validation – geometry and 2%-80% page-area threshold.

Scoring:
  Each check is weighted into a 0.0-1.0 confidence score; PASS requires >= 0.95 (PRD).
  Hard gates fail a record regardless of score: any price not matched near the product,
  a price-logic violation, or a page with no verifiable text layer.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from app.pdf_renderer import TextBlock
from app.schemas import CROPPABLE_TYPES, ExtractedProduct, ValidationResult

PASS_THRESHOLD = 0.95

WEIGHTS = {
    "text_snippet": 0.30,
    "numeric_mrp":  0.20,
    "numeric_b2b":  0.20,
    "numeric_moq":  0.10,
    "price_logic":  0.10,
    "bbox":         0.10,
}

# How far (0-1000 page units) around the product's anchors numbers are searched.
NEIGHBOURHOOD_MARGIN = 100
MIN_AREA_RATIO = 0.02
MAX_AREA_RATIO = 0.80
QTY_KEYWORDS = re.compile(r"\b(?:moq|min(?:imum)?|qty|quantity|pcs|pieces|units|nos|order)\b", re.IGNORECASE)

# Exact number tokens: 450 | 450.00 | 1,450 | 1,00,000.50 — never a fragment of a longer number.
_NUMBER_TOKEN = re.compile(r"(?<!\d)(?<!\d[.,])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?(?!\d)")

# Tesseract has no ₹ glyph and usually reads it as a digit ("₹2599" → "22599"). On OCR text only,
# a number directly after a price label may have that one leading misread digit removed.
_OCR_RUPEE_MISREAD = re.compile(
    r"(?:mrp|price|rate|rs\.?|inr|[:=])[^\S\n]*[^\w\s]?[^\S\n]*[2378]"
    r"((?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)(?!\d)",
    re.IGNORECASE,
)
OCR_FUZZY_THRESHOLD = 85  # rapidfuzz partial_ratio for OCR snippet / anchor matching

Box = Tuple[float, float, float, float]  # (y0, x0, y1, x1) in 0-1000 space


# ── Text helpers ─────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-").replace(" ", " "))
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_numbers(text: str) -> List[float]:
    """All exact numeric tokens in `text`, with thousands separators removed."""
    out: List[float] = []
    for m in _NUMBER_TOKEN.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def number_in_text(value: float, text: str) -> bool:
    """True if `value` appears in `text` as a complete number token (450 matches 450.00, never 1450)."""
    return any(abs(n - value) < 1e-6 for n in parse_numbers(text))


_PRICE_WORD = re.compile(r"\b(?:mrp|price|rate)\b", re.IGNORECASE)
_STANDALONE_MISREAD = re.compile(r"[^\w]{0,3}[2378]((?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)[^\w]{0,4}")


def _under_label_hits(blocks: Sequence[TextBlock]) -> List[Tuple[float, TextBlock]]:
    labels = [b for b in blocks if _PRICE_WORD.search(b.text)]
    out: List[Tuple[float, TextBlock]] = []
    for b in blocks:
        m = _STANDALONE_MISREAD.fullmatch(b.text.strip())
        if m and any(0 < b.y0 - lab.y0 <= 45 and lab.x0 - 60 <= b.x0 <= lab.x1 + 60 for lab in labels):
            out.append((float(m.group(1).replace(",", "")), b))
    return out


def prices_under_labels(blocks: Sequence[TextBlock]) -> List[float]:
    """
    OCR layout where the price sits alone on its own line directly under a small "MRP" label
    ("MRP" / "₹1499" read as "21499"): the value with the misread ₹ digit removed.
    """
    return [v for v, _ in _under_label_hits(blocks)]


def find_number(value: float, text: str, ocr: bool = False,
                blocks: Optional[Sequence[TextBlock]] = None) -> Optional[str]:
    """'exact' if `value` is a number token in `text`; 'ocr_currency' if it only matches after
    removing a misread ₹ glyph next to a price label (OCR text only); else None."""
    if number_in_text(value, text):
        return "exact"
    if ocr:
        for m in _OCR_RUPEE_MISREAD.finditer(text or ""):
            if abs(float(m.group(1).replace(",", "")) - value) < 1e-6:
                return "ocr_currency"
        if blocks and any(abs(v - value) < 1e-6 for v in prices_under_labels(blocks)):
            return "ocr_currency"
    return None


def _alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def fuzzy_contains(needle: str, haystack: str) -> float:
    """Best partial-alignment similarity (0-100) of `needle` inside `haystack`, ignoring punctuation."""
    n, h = _alnum(needle), _alnum(haystack)
    if not n or not h:
        return 0.0
    return fuzz.partial_ratio(n, h)


_OCR_CONFUSABLE = str.maketrans({"l": "i", "1": "i", "|": "i", "!": "i", "0": "o"})


def ocr_word_coverage(needle: str, haystack: str) -> float:
    """
    Share (0-1) of `needle`'s words found in OCR text, independent of line order and tolerant
    of classic OCR confusions (I/l/1/|, O/0). Long words may differ by one or two characters.
    """
    words = _alnum(needle).translate(_OCR_CONFUSABLE).split()
    page_canon = _alnum(haystack).translate(_OCR_CONFUSABLE)
    page_words = set(page_canon.split())
    page_joined = page_canon.replace(" ", "")  # OCR often glues words: "MY BREWII"
    if not words or not page_words:
        return 0.0
    found = 0
    for w in words:
        if (w in page_words or (len(w) >= 3 and w in page_joined)
                or (len(w) >= 5 and any(fuzz.ratio(w, p) >= 85 for p in page_words))):
            found += 1
    return found / len(words)


# ── Product name ─────────────────────────────────────────────────────────────

TITLE_BLOCK_DEPTH = 120   # 0-1000 units below an anchor line that still count as its title block
TITLE_BLOCK_RISE = 80     # …and above it (the model name sits above the descriptor line)


def _word_found(word: str, words: set, joined: str, ocr: bool) -> bool:
    if word in words:
        return True
    if ocr and len(word) >= 3 and word in joined:      # OCR glues words: "BREWII"
        return True
    return ocr and len(word) >= 5 and any(fuzz.ratio(word, w) >= 85 for w in words)


def name_problem(name: str, page_raw_text: str, blocks: Sequence[TextBlock], ocr: bool,
                 snippet: str = "") -> Optional[str]:
    """
    The product name must be printed as a title block: every word (3+ characters) of the extracted
    name must appear in one run of lines — a line containing a name word plus the lines just below
    it. When the snippet (the product's title line) is located, only blocks around it count, so
    words scattered across feature text cannot make up a name.
    Known limit: when the catalog itself names the product differently in two places (title
    "Kettle & Cup", family strip "Kettle & Flask"), either printed name passes.
    """
    canon = (lambda s: _alnum(s).translate(_OCR_CONFUSABLE)) if ocr else _alnum
    words = [w for w in canon(name).split() if len(w) >= 3 or not ocr]
    if not words:
        return None
    if not blocks:  # no layout available: fall back to whole-page word presence
        page = canon(page_raw_text)
        missing = [w for w in words if not _word_found(w, set(page.split()), page.replace(" ", ""), ocr)]
        return f"product name words not found on the page: {', '.join(missing)}" if missing else None

    lines = [(b, canon(b.text)) for b in blocks]
    title_anchors = _snippet_blocks(snippet, blocks, ocr) if snippet else []
    best_missing: Optional[List[str]] = None
    for anchor, anchor_text in lines:
        if title_anchors and anchor not in title_anchors:
            continue
        anchor_words = set(anchor_text.split())
        if not any(_word_found(w, anchor_words, anchor_text.replace(" ", ""), ocr) for w in words):
            continue
        window = [t for b, t in lines
                  if anchor.y0 - TITLE_BLOCK_RISE <= b.y0 <= anchor.y0 + TITLE_BLOCK_DEPTH
                  and b.x0 < anchor.x1 + 150 and b.x1 > anchor.x0 - 150]
        text = " ".join(window)
        missing = [w for w in words if not _word_found(w, set(text.split()), text.replace(" ", ""), ocr)]
        if not missing:
            return None
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing
    if best_missing is None:
        return f"product name '{name}' not found on the page"
    return f"product name words not printed in its title block: {', '.join(best_missing)}"


def _snippet_blocks(snippet: str, blocks: Sequence[TextBlock], ocr: bool) -> List[TextBlock]:
    """Blocks holding (part of) the snippet — the product's own title line."""
    snip = normalize_text(snippet)
    whole, partial = [], []
    for b in blocks:
        bt = normalize_text(b.text)
        if len(_alnum(bt)) < 3:
            continue
        if snip in bt or (ocr and fuzzy_contains(snip, bt) >= OCR_FUZZY_THRESHOLD):
            whole.append(b)       # the line carries the entire snippet
        elif (len(bt) >= 6 and bt in snip) or \
                (ocr and len(_alnum(bt)) >= 6 and fuzzy_contains(bt, snip) >= OCR_FUZZY_THRESHOLD):
            partial.append(b)     # the line is a piece of a multi-line snippet
    return whole or partial


def title_region(product: ExtractedProduct, blocks: Sequence[TextBlock], ocr: bool) -> Tuple[float, float, float, float]:
    """Area to re-OCR at high magnification when the title text was not readable."""
    anchors = _snippet_blocks(product.raw_text_snippet or "", blocks, ocr) or \
        _anchor_blocks(product, blocks, ocr)
    if anchors:
        y0, x0 = min(b.y0 for b in anchors), min(b.x0 for b in anchors)
        y1, x1 = max(b.y1 for b in anchors), max(b.x1 for b in anchors)
        return (max(0, y0 - TITLE_BLOCK_RISE - 30), max(0, x0 - 60),
                min(1000, y1 + TITLE_BLOCK_DEPTH + 40), min(1000, x1 + 450))
    return (0, 0, 300, 1000)  # catalog titles usually sit in the top band


# ── Locality ─────────────────────────────────────────────────────────────────

def _block_box(b: TextBlock) -> Box:
    return (b.y0, b.x0, b.y1, b.x1)


def _intersects(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _anchor_blocks(product: ExtractedProduct, blocks: Sequence[TextBlock], ocr: bool = False) -> List[TextBlock]:
    """Blocks that belong to this product: those overlapping its snippet or carrying its SKU."""
    snippet = normalize_text(product.raw_text_snippet or "")
    sku = normalize_text(product.sku_or_model or "")
    anchors = []
    for b in blocks:
        bt = normalize_text(b.text)
        if not bt:
            continue
        if snippet and (snippet in bt or (len(bt) >= 6 and bt in snippet)):
            anchors.append(b)
        elif ocr and snippet and len(_alnum(bt)) >= 6 and fuzzy_contains(snippet, bt) >= OCR_FUZZY_THRESHOLD:
            anchors.append(b)
        elif sku and len(sku) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(sku) + r"(?![a-z0-9])", bt):
            anchors.append(b)
    return anchors


def product_region(product: ExtractedProduct, blocks: Sequence[TextBlock], ocr: bool = False,
                   margin: float = NEIGHBOURHOOD_MARGIN) -> Optional[Box]:
    boxes: List[Box] = [_block_box(b) for b in _anchor_blocks(product, blocks, ocr)]
    for img in product.images:
        if img.type != "swatch" and len(img.box_2d) == 4 and not bbox_problems(img.box_2d, 50):
            boxes.append(tuple(img.box_2d))  # type: ignore[arg-type]
    if not boxes:
        return None
    m = margin
    return (
        max(0.0, min(b[0] for b in boxes) - m),
        max(0.0, min(b[1] for b in boxes) - m),
        min(1000.0, max(b[2] for b in boxes) + m),
        min(1000.0, max(b[3] for b in boxes) + m),
    )


def neighbourhood_blocks(product: ExtractedProduct, blocks: Sequence[TextBlock], ocr: bool = False) -> List[TextBlock]:
    region = product_region(product, blocks, ocr)
    if region is None:
        return []
    return [b for b in blocks if _intersects(_block_box(b), region)]


def neighbourhood_text(product: ExtractedProduct, blocks: Sequence[TextBlock], ocr: bool = False) -> str:
    return "\n".join(b.text for b in neighbourhood_blocks(product, blocks, ocr))


def _gap(a: Box, b: Box) -> float:
    """Distance between two boxes (0 when they overlap)."""
    dy = max(0.0, a[0] - b[2], b[0] - a[2])
    dx = max(0.0, a[1] - b[3], b[1] - a[3])
    return (dx * dx + dy * dy) ** 0.5


def _blocks_with_value(value: float, blocks: Sequence[TextBlock], ocr: bool) -> List[Tuple[TextBlock, str]]:
    """Every block printing `value` (exactly, or via an OCR-misread ₹ next to a price label)."""
    hits: List[Tuple[TextBlock, str]] = []
    for b in blocks:
        how = find_number(value, b.text, ocr)
        if how:
            hits.append((b, how))
    if ocr:
        hits += [(b, "ocr_currency") for v, b in _under_label_hits(blocks) if abs(v - value) < 1e-6]
    return hits


def nearest_product_owns(value: float, product: ExtractedProduct, others: Sequence[ExtractedProduct],
                         blocks: Sequence[TextBlock], ocr: bool) -> Optional[str]:
    """
    Layout-independent locality: a printed price belongs to the product it is physically closest to.
    Passes when some block printing `value` is at least as close to this product as to every other
    product on the page — so two products' prices can never be swapped, on spreads, grids or tables.
    """
    mine = product_region(product, blocks, ocr, margin=0)
    if mine is None:
        return None
    theirs = [r for r in (product_region(o, blocks, ocr, margin=0) for o in others) if r is not None]
    for block, how in _blocks_with_value(value, blocks, ocr):
        box = _block_box(block)
        if all(_gap(box, mine) <= _gap(box, r) for r in theirs):
            return how
    return None


# ── Bounding boxes ───────────────────────────────────────────────────────────

def min_area_ratio(products_on_page: int = 1) -> float:
    """
    PRD minimum photo size is 2% of the page — right for 1-8 products per page. Dense grids
    (12, 24, 48 products) have legitimately smaller photos, so the floor scales with the count
    (never below 0.3%, which still rejects icons and colour swatches).
    """
    return max(0.003, min(MIN_AREA_RATIO, 0.25 / max(products_on_page, 1)))


SECONDARY_MIN_AREA_RATIO = 0.004   # variant / alternate views may be small insets


def bbox_problems(box_2d: Sequence[float], products_on_page: int = 1, secondary: bool = False) -> List[str]:
    """Empty list if the normalized [ymin, xmin, ymax, xmax] box is usable."""
    if len(box_2d) != 4:
        return ["bbox does not have exactly 4 coordinates"]
    ymin, xmin, ymax, xmax = box_2d
    problems: List[str] = []
    if any(v < -1 or v > 1001 for v in box_2d):
        problems.append(f"bbox {list(box_2d)} lies outside the 0-1000 page space")
    if ymax <= ymin:
        problems.append(f"invalid bbox: ymax ({ymax:g}) <= ymin ({ymin:g})")
    if xmax <= xmin:
        problems.append(f"invalid bbox: xmax ({xmax:g}) <= xmin ({xmin:g})")
    if not problems:
        ratio = (ymax - ymin) * (xmax - xmin) / 1_000_000
        floor = min_area_ratio(products_on_page)
        if secondary:
            floor = min(floor, SECONDARY_MIN_AREA_RATIO)
        if ratio < floor:
            problems.append(f"bbox too small: {ratio:.1%} of page (min {floor:.1%})")
        elif ratio > MAX_AREA_RATIO:
            problems.append(f"bbox too large: {ratio:.1%} of page (max {MAX_AREA_RATIO:.0%})")
    return problems


# ── Price slab normalization ─────────────────────────────────────────────────

def apply_price_slabs(product: ExtractedProduct) -> List[str]:
    """
    PRD §6: the smallest-quantity slab sets moq and the standard b2b_price.
    Fills missing values deterministically; reports conflicts instead of overwriting.
    """
    if not product.price_slabs:
        return []
    entry = min(product.price_slabs, key=lambda s: s.min_qty)
    issues: List[str] = []
    if product.moq is None:
        product.moq = entry.min_qty
    elif product.moq != entry.min_qty:
        issues.append(f"moq {product.moq} conflicts with smallest price slab min_qty {entry.min_qty}")
    if product.b2b_price is None:
        product.b2b_price = entry.price
    elif abs(product.b2b_price - entry.price) > 1e-6:
        issues.append(f"b2b_price {product.b2b_price:g} conflicts with smallest-slab price {entry.price:g}")
    return issues


# ── Page-level checks ────────────────────────────────────────────────────────

_PRICE_LABEL = re.compile(
    r"(?:mrp|price|rate|rs\.?|inr|₹)[^\S\n]*[:=\-]?[^\S\n]*[^\w\s]?[^\S\n]*"
    r"((?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)(?!\d)",
    re.IGNORECASE,
)


def unexplained_price_labels(page_raw_text: str, products: Sequence[ExtractedProduct], ocr: bool) -> List[float]:
    """
    Recall check: price-labelled numbers on the page ("MRP: 4,999") that no extracted product
    accounts for. Any such value suggests a product the model missed.
    """
    claimed: List[float] = []
    for p in products:
        claimed += [v for v in (p.mrp, p.b2b_price) if v is not None]
        claimed += [s.price for s in p.price_slabs]
    missing: List[float] = []
    for m in _PRICE_LABEL.finditer(page_raw_text or ""):
        raw = m.group(1)
        value = float(raw.replace(",", ""))
        candidates = {value}
        if ocr and raw[0] in "2378" and len(raw.replace(",", "")) > 2:
            candidates.add(float(raw[1:].lstrip(",").replace(",", "") or 0))  # misread ₹
        if value < 10 or any(abs(c - v) < 1e-6 for c in candidates for v in claimed):
            continue
        if not any(abs(value - x) < 1e-6 for x in missing):
            missing.append(value)
    return missing


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, y1 - y0) * max(0.0, x1 - x0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cross_product_checks(products: Sequence[ExtractedProduct], results: Sequence[ValidationResult]) -> None:
    """Fail products that claim the same primary photo as another product (misattributed images)."""
    boxes = [[img.box_2d for img in p.images if img.type == "primary" and len(img.box_2d) == 4] for p in products]
    for i in range(len(products)):
        for j in range(i + 1, len(products)):
            if any(_iou(a, b) > 0.7 for a in boxes[i] for b in boxes[j]):
                for k, other in ((i, j), (j, i)):
                    results[k].passed = False
                    results[k].failure_reasons.append(
                        f"same primary photo assigned to '{products[other].product_name}' — possible misattributed image")


# ── Main entry point ─────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f"{v:g}"


def verify_product(
    product: ExtractedProduct,
    page_raw_text: str,
    text_blocks: Optional[Iterable[TextBlock]] = None,
    text_source: str = "native",
    pass_threshold: float = PASS_THRESHOLD,
    products_on_page: int = 1,
    page_products: Optional[Sequence[ExtractedProduct]] = None,
) -> ValidationResult:
    """
    Run all deterministic checks on one product. Mutates `product` only to apply price slabs.
    Numbers must be found near the product when the page has several products; on a
    single-product page the whole page is that product's neighbourhood.
    """
    blocks = list(text_blocks or [])
    ocr = text_source == "ocr"
    scores = {k: 0.0 for k in WEIGHTS}
    failures: List[str] = []
    hard_fail = False

    slab_issues = apply_price_slabs(product)

    if not (product.product_name or "").strip():
        failures.append("product_name is empty")
        hard_fail = True

    has_text = text_source in ("native", "ocr") and bool(page_raw_text.strip())
    if not has_text:
        failures.append("page has no text layer and OCR is unavailable — values cannot be verified")
        hard_fail = True
    else:
        page_norm = normalize_text(page_raw_text)
        localized = products_on_page > 1 and bool(blocks)
        near_blocks = neighbourhood_blocks(product, blocks, ocr) if localized else blocks
        near_text = "\n".join(b.text for b in near_blocks) if localized else page_raw_text

        # ── Check 1: verbatim snippet (fuzzy alignment on OCR text) ─────────
        # An elided snippet ("NAME ... MRP: 450") is checked segment by segment; every
        # segment must be on the page, and the weakest one sets the score.
        snippet = normalize_text(product.raw_text_snippet or "")
        segments = [s.strip() for s in re.split(r"\.{3,}|…", snippet) if len(s.strip()) >= 3]
        if not segments:
            failures.append("raw_text_snippet is missing")
        else:
            seg_scores = []
            for seg in segments:
                if seg in page_norm:
                    seg_scores.append(1.0)
                    continue
                similarity = fuzzy_contains(seg, page_raw_text)
                if ocr and (similarity >= OCR_FUZZY_THRESHOLD or ocr_word_coverage(seg, page_raw_text) >= 0.85):
                    seg_scores.append(1.0)  # OCR recognition noise / OCR line order
                elif similarity >= 70:
                    seg_scores.append(0.5)
                    failures.append(f"snippet text {seg!r} not found verbatim in page text "
                                    f"(similarity {similarity:.0f}%)")
                else:
                    seg_scores.append(0.0)
                    failures.append(f"snippet text {seg!r} not found in page text (possible hallucination)")
            scores["text_snippet"] = min(seg_scores)

        # Every word of the product name must be printed in its title block.
        problem = name_problem(product.product_name or "", page_raw_text, blocks, ocr,
                               product.raw_text_snippet or "")
        if problem:
            scores["text_snippet"] = min(scores["text_snippet"], 0.5)
            failures.append(problem)

        # ── Check 2: numeric cross-check (prices are hard gates) ────────────
        def check_price(name: str, value: Optional[float]) -> float:
            nonlocal hard_fail
            if value is None:
                return 1.0  # null makes no claim
            how = find_number(value, near_text, ocr, near_blocks)
            if how is None and localized and page_products:
                others = [o for o in page_products if o is not product]
                how = nearest_product_owns(value, product, others, blocks, ocr)
            if how == "exact":
                return 1.0
            if how == "ocr_currency":
                failures.append(f"warning: {name} {_fmt(value)} matched after correcting an OCR-misread ₹ symbol")
                return 1.0
            hard_fail = True
            if localized and find_number(value, page_raw_text, ocr, blocks):
                failures.append(f"{name} {_fmt(value)} appears on the page but not near this product")
            else:
                failures.append(f"{name} {_fmt(value)} not found in page text stream")
            return 0.0

        scores["numeric_mrp"] = check_price("mrp", product.mrp)
        scores["numeric_b2b"] = check_price("b2b_price", product.b2b_price)
        for slab in product.price_slabs:
            check_price(f"price slab ({slab.min_qty}+ pcs) price", slab.price)

        def qty_found(qty: int) -> bool:
            if number_in_text(qty, near_text):
                return True
            # A catalog-wide statement such as "MOQ: 100 pcs" in a header/footer block.
            return any(QTY_KEYWORDS.search(b.text) and number_in_text(qty, b.text) for b in blocks) \
                if blocks else False

        if product.moq is None:
            scores["numeric_moq"] = 1.0
        elif qty_found(product.moq):
            scores["numeric_moq"] = 1.0
        else:
            failures.append(f"moq {product.moq} not found near this product in page text stream")
        for slab in product.price_slabs:
            if not qty_found(slab.min_qty):
                scores["numeric_moq"] = 0.0
                failures.append(f"price slab min_qty {slab.min_qty} not found in page text stream")

    # ── Check 3: price logic ────────────────────────────────────────────────
    price_ok = True
    for name, value in (("mrp", product.mrp), ("b2b_price", product.b2b_price)):
        if value is not None and value <= 0:
            price_ok = False
            failures.append(f"{name} must be a positive number (got {_fmt(value)})")
    if product.moq is not None and product.moq <= 0:
        price_ok = False
        failures.append(f"moq must be positive (got {product.moq})")
    if product.mrp is not None and product.b2b_price is not None and product.b2b_price > product.mrp:
        price_ok = False
        failures.append(f"b2b_price ({_fmt(product.b2b_price)}) > mrp ({_fmt(product.mrp)}) — price logic violation")
    for slab in product.price_slabs:
        if slab.price <= 0 or slab.min_qty <= 0:
            price_ok = False
            failures.append(f"price slab has non-positive values (min_qty={slab.min_qty}, price={_fmt(slab.price)})")
    if slab_issues:
        price_ok = False
        failures.extend(slab_issues)
    scores["price_logic"] = 1.0 if price_ok else 0.0
    if not price_ok:
        hard_fail = True

    # ── Check 4: bounding boxes ─────────────────────────────────────────────
    # Invalid boxes are never cropped. The score reflects whether the product has a usable
    # main photo: a valid primary/group shot → 1.0; no photo at all → 0.5 (data can still
    # publish, flagged); photos given but none usable → 0.0 (the extraction is confused).
    # Invalid secondary boxes (e.g. tiny feature thumbnails) are reported as warnings.
    n = products_on_page
    croppable = [img for img in product.images if img.type in CROPPABLE_TYPES]
    valid_main = [img for img in croppable if img.type != "variant" and not bbox_problems(img.box_2d, n)]
    valid_any = [img for img in croppable if not bbox_problems(img.box_2d, n, img.type == "variant")]
    for img in product.images:
        if img.type == "swatch":
            continue
        for problem in bbox_problems(img.box_2d, n, img.type in ("variant", "packaging")):
            prefix = "" if (img.type in ("primary", "group_shot") and not valid_main) else "warning: "
            failures.append(f"{prefix}{img.type} image skipped — {problem}")
    if valid_main:
        scores["bbox"] = 1.0
    elif not croppable:
        scores["bbox"] = 0.5
        failures.append("warning: no product photo bounding box extracted")
    elif valid_any:
        scores["bbox"] = 0.5
        failures.append("warning: no usable primary photo; only variant photos cropped")
    else:
        scores["bbox"] = 0.0

    confidence = round(sum(scores[k] * WEIGHTS[k] for k in WEIGHTS), 4)
    passed = (not hard_fail) and confidence >= pass_threshold
    # Failed only because the title/snippet text could not be confirmed (every number verified).
    text_only = (not passed and not hard_fail and
                 confidence + (1 - scores["text_snippet"]) * WEIGHTS["text_snippet"] >= pass_threshold)

    return ValidationResult(
        confidence_score=confidence,
        passed=passed,
        failure_reasons=failures,
        check_scores=scores,
        is_scanned_pdf=text_source != "native",
        text_source=text_source if has_text else "none",
        text_only_failure=text_only,
    )
