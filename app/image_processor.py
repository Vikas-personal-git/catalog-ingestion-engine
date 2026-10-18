"""
Stage 4 – Asset Processing.
Crops product images from rendered page PNGs using normalized bounding boxes,
then removes backgrounds to produce clean transparent PNGs.

Background removal backends:
  - rembg (ML segmentation, default model u2net — fastest on CPU with clean product cut-outs).
    onnxruntime is native code that can crash the interpreter outright on machines with an
    outdated Visual C++ runtime, so availability is probed once in a subprocess before
    importing it in-process.
  - flat-background fallback: flood-fills the uniform (white / solid) background connected
    to the crop border. Used when rembg is unavailable; patterned backgrounds are left intact
    and reported as a warning.
"""
from __future__ import annotations

import functools
import io
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

logger = logging.getLogger(__name__)

_rembg_lock = threading.Lock()
_rembg_sessions: dict = {}


@dataclass
class CropResult:
    path: Optional[Path]
    background_removed: bool = False
    method: str = "none"           # rembg | flat-fill | none
    warning: Optional[str] = None
    upscale: float = 1.0           # >1 when the source held fewer pixels than min_px
    size: Tuple[int, int] = (0, 0)
    original_path: Optional[Path] = None   # the crop before background removal


# ── Backend selection ────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def rembg_status() -> tuple[bool, str]:
    """(available, detail). Imports onnxruntime + rembg in a child process first."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import onnxruntime, rembg"],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:
        return False, f"probe failed: {exc}"
    if proc.returncode == 0:
        return True, "rembg available"
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    reason = detail[-1] if detail else f"exit code {proc.returncode}"
    if proc.returncode in (5, 0xC0000005, -1073741819, 3221225477):
        reason = ("onnxruntime crashed on import (access violation) — install the latest "
                  "Microsoft Visual C++ Redistributable x64")
    return False, reason


def _rembg_session(model: str):
    with _rembg_lock:
        if model not in _rembg_sessions:
            from rembg import new_session
            _rembg_sessions[model] = new_session(model)
            logger.info("rembg session initialised (%s).", model)
        return _rembg_sessions[model]


def _remove_bg_rembg(image: Image.Image, model: str) -> Image.Image:
    from rembg import remove
    session = _rembg_session(model)
    out = remove(image, session=session)
    return out.convert("RGBA") if isinstance(out, Image.Image) else Image.open(io.BytesIO(out)).convert("RGBA")


def _remove_bg_flat(image: Image.Image, tolerance: float = 28.0) -> Optional[Image.Image]:
    """
    Remove a uniform background connected to the image border.
    Returns None when the border is not uniform (photo / patterned background).
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.int16)
    h, w, _ = rgb.shape
    if h < 4 or w < 4:
        return None
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    border_dist = np.linalg.norm(border - bg, axis=1)
    if np.mean(border_dist < tolerance) < 0.85:
        return None  # border is not a flat colour

    close = np.linalg.norm(rgb - bg, axis=2) < tolerance
    labels, _ = ndimage.label(close)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    background = np.isin(labels, edge_labels)
    # Soften the cut edge by one pixel to avoid jaggies.
    alpha = np.where(background, 0, 255).astype(np.uint8)
    alpha = ndimage.uniform_filter(alpha.astype(np.float32), size=3).astype(np.uint8)
    alpha[~ndimage.binary_dilation(background, iterations=1)] = 255

    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(rgba, "RGBA")


def _trim_transparent(img: Image.Image, pad_ratio: float = 0.02) -> Image.Image:
    """Tighten the canvas to the opaque product, keeping a small transparent margin."""
    bbox = img.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if not bbox:
        return img
    l, t, r, b = bbox
    pad = int(max(r - l, b - t) * pad_ratio)
    return img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))


# ── Native-resolution crop source ────────────────────────────────────────────

VECTOR_DPI = 300      # pages drawn as vectors/text have no native pixel grid — render crisp
MAX_CROP_DPI = 600


class CropSource:
    """
    Produces crops of one page at the best resolution the source actually holds.
    For PDFs the DPI is chosen per region from the embedded images covering it, so a 115-DPI
    flattened page is cropped at ~115 DPI (no blurry upsampling) while a high-res photo placed
    on the page is cropped at its own resolution. Image files are cropped at native pixels.
    """

    def __init__(self, source: Path, index: int = 0, is_image: bool = False):
        self.source, self.index, self.is_image = Path(source), index, is_image

    def crop(self, box: Tuple[float, float, float, float]) -> Tuple[Image.Image, float]:
        """box = (ymin, xmin, ymax, xmax) in 0-1000 page space. Returns (RGB image, dpi used)."""
        ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, v)) for v in box)
        if self.is_image:
            with Image.open(self.source) as img:
                W, H = img.size
                return img.convert("RGB").crop((int(xmin / 1000 * W), int(ymin / 1000 * H),
                                                int(round(xmax / 1000 * W)), int(round(ymax / 1000 * H)))), 0.0
        import pymupdf as fitz
        with fitz.open(str(self.source)) as doc:
            page = doc[self.index]
            r = page.rect
            clip = fitz.Rect(r.x0 + xmin / 1000 * r.width, r.y0 + ymin / 1000 * r.height,
                             r.x0 + xmax / 1000 * r.width, r.y0 + ymax / 1000 * r.height)
            dpi = self._native_dpi(page, clip)
            pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"), dpi

    @staticmethod
    def _native_dpi(page, clip) -> float:
        best = 0.0
        clip_area = max(clip.width * clip.height, 1e-6)
        for info in page.get_image_info():
            import pymupdf as fitz
            bbox = fitz.Rect(info["bbox"])
            inter = bbox & clip
            if inter.is_empty or (inter.width * inter.height) / clip_area < 0.3 or bbox.width <= 0:
                continue
            best = max(best, info["width"] / bbox.width * 72)
        return int(round(min(MAX_CROP_DPI, best))) if best else VECTOR_DPI


def _touching_sides(img: Image.Image, threshold: float = 0.12) -> List[str]:
    """Sides of a cut-out where the product runs into the crop border (likely cut off)."""
    alpha = np.asarray(img.getchannel("A")) > 128
    sides = {"top": alpha[0], "bottom": alpha[-1], "left": alpha[:, 0], "right": alpha[:, -1]}
    return [s for s, edge in sides.items() if edge.mean() > threshold]


def _fit_size(img: Image.Image, min_px: int, max_px: int) -> Tuple[Image.Image, float]:
    """Bring a cut-out to listing size: downscale very large sources to `max_px`, upscale small ones."""
    long_edge = max(img.size)
    if max_px and long_edge > max_px:
        f = max_px / long_edge
        down = img.convert("RGBa").resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
        return down.convert("RGBA"), f
    return _upscale(img, min_px)


def _upscale(img: Image.Image, min_px: int) -> Tuple[Image.Image, float]:
    """Lanczos upscale (≤3×) of small cut-outs, premultiplied to avoid dark fringes, plus light sharpening."""
    long_edge = max(img.size)
    if not min_px or long_edge >= min_px:
        return img, 1.0
    factor = min(3.0, min_px / long_edge)
    size = (round(img.width * factor), round(img.height * factor))
    up = img.convert("RGBa").resize(size, Image.LANCZOS).convert("RGBA")
    rgb = up.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.6, percent=70, threshold=2))
    rgb.putalpha(up.getchannel("A"))
    return rgb, factor


# ── Public API ───────────────────────────────────────────────────────────────

def padded_box(box_2d: List[float], padding_pct: float = 0.03) -> Optional[Tuple[float, float, float, float]]:
    """Normalized [ymin, xmin, ymax, xmax] plus a safety margin, clamped to the page."""
    if len(box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = (float(v) for v in box_2d)
    pad_y, pad_x = (ymax - ymin) * padding_pct, (xmax - xmin) * padding_pct
    box = (max(0.0, ymin - pad_y), max(0.0, xmin - pad_x), min(1000.0, ymax + pad_y), min(1000.0, xmax + pad_x))
    return box if box[2] - box[0] > 1 and box[3] - box[1] > 1 else None


SECOND_OPINION_MODEL = "isnet-general-use"   # strong on light / metallic products where u2net washes out
# Share of foreground pixels with uncertain alpha in a trustworthy cut-out. Calibrated on real
# catalog photos: good cut-outs scored ≤0.16, washed-out ones (white/chrome erased) ≥0.77.
MAX_HALF_TRANSPARENT = 0.30
MIN_COVERAGE = 0.20           # a product should fill at least this much of its own crop


def cutout_quality(img: Image.Image) -> Tuple[float, float]:
    """(coverage, half_transparent): how much of the crop is product, and how much of that is
    semi-transparent. Washed-out cut-outs (white / chrome parts judged as background) score high."""
    a = np.asarray(img.getchannel("A"))
    fg = a > 25
    coverage = float(fg.mean())
    half = float(((a > 25) & (a < 230)).sum() / max(int(fg.sum()), 1))
    return coverage, half


def _trustworthy(img: Image.Image) -> bool:
    coverage, half = cutout_quality(img)
    return coverage >= MIN_COVERAGE and half <= MAX_HALF_TRANSPARENT


def _remove_background(img: Image.Image, rembg_model: str, name: str) -> Tuple[Optional[Image.Image], str, Optional[str]]:
    """
    Background removal with a quality gate:
      1. primary model; accept a confident, solid cut-out;
      2. otherwise ask a second model and keep, per pixel, whichever is surer it is product;
      3. otherwise return None → the caller keeps the original crop on white and flags it
         (e.g. a white product photographed on white cannot be separated reliably).
    """
    available, detail = rembg_status()
    if available:
        try:
            first = _remove_bg_rembg(img, rembg_model)
            if _trustworthy(first):
                return first, "rembg", None
            second = _remove_bg_rembg(img, SECOND_OPINION_MODEL) if rembg_model != SECOND_OPINION_MODEL else first
            merged = first.copy()
            merged.putalpha(Image.fromarray(np.maximum(np.asarray(first.getchannel("A")),
                                                       np.asarray(second.getchannel("A")))))
            if _trustworthy(merged):
                return merged, "rembg-ensemble", None
            return None, "none", ("background not removed — the product's colours are too close to its "
                                  "background to separate reliably; using the original photo (please review)")
        except Exception as exc:
            logger.warning("rembg failed for %s: %s — trying flat-background fallback", name, exc)
    cleaned = _remove_bg_flat(img)
    if cleaned is not None:
        return cleaned, "flat-fill", None
    reason = f"rembg unavailable: {detail}" if not available else "rembg failed"
    return None, "none", f"background not removed (non-uniform background; {reason})"


def crop_and_clean(
    source: CropSource,
    box_2d: List[float],
    output_path: Path,
    padding_pct: float = 0.03,
    remove_bg: bool = True,
    rembg_model: str = "u2net",
    min_px: int = 1000,
    max_expansions: int = 3,
    max_px: int = 2000,
) -> CropResult:
    """
    Crop a product at native resolution (3% safety margin), remove its background, and write an
    RGBA PNG. If the cut-out runs into a crop edge (product cut off by a tight box), that side is
    widened by 10% and the cut redone, up to `max_expansions` times.
    """
    box = padded_box(box_2d, padding_pct)
    if box is None:
        return CropResult(None, warning=f"degenerate crop box {box_2d}")

    warning: Optional[str] = None
    method = "none"
    result_img: Optional[Image.Image] = None
    cropped: Optional[Image.Image] = None
    for attempt in range(max_expansions + 1):
        cropped, _dpi = source.crop(box)
        if cropped.width < 2 or cropped.height < 2:
            return CropResult(None, warning=f"crop box {box_2d} is empty")
        if not remove_bg:
            result_img = cropped.convert("RGBA")
            break
        cleaned, method, warning = _remove_background(cropped, rembg_model, output_path.name)
        if cleaned is None:
            result_img = cropped.convert("RGBA")
            break
        if np.mean(np.asarray(cleaned.getchannel("A")) > 8) < 0.01:
            warning, method = "background removal erased the product — kept the plain crop", "none"
            result_img = cropped.convert("RGBA")
            break
        result_img = cleaned
        ymin, xmin, ymax, xmax = box
        h, w = ymax - ymin, xmax - xmin
        grow = {"top": ymin > 0, "bottom": ymax < 1000, "left": xmin > 0, "right": xmax < 1000}
        sides = [s for s in _touching_sides(cleaned) if grow[s]]
        if not sides or attempt == max_expansions:
            if sides:
                warning = f"product may extend beyond the extracted box ({', '.join(sides)})"
            break
        box = (max(0.0, ymin - 0.1 * h) if "top" in sides else ymin,
               max(0.0, xmin - 0.1 * w) if "left" in sides else xmin,
               min(1000.0, ymax + 0.1 * h) if "bottom" in sides else ymax,
               min(1000.0, xmax + 0.1 * w) if "right" in sides else xmax)

    if method != "none":
        result_img = _trim_transparent(result_img)
    result_img, factor = _fit_size(result_img, min_px, max_px)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(str(output_path), "PNG", optimize=True)

    # Keep the untouched crop next to the cut-out so a reviewer can always compare / fall back.
    original_path = output_path.with_name(output_path.stem + "_original.jpg")
    original, _ = _fit_size(cropped.convert("RGBA"), min_px, max_px)
    original.convert("RGB").save(str(original_path), "JPEG", quality=92)

    return CropResult(output_path, background_removed=method != "none", method=method, warning=warning,
                      upscale=factor, size=result_img.size, original_path=original_path)
