"""Catalog layouts unlike the Gifteeco samples: hybrid (image + thin text layer) and dense grids."""
import asyncio
import io
import json
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.pdf_renderer import enumerate_pages, extract_page_text, tesseract_cmd
from app.pipeline import PipelineOptions, run_pipeline
from app.vlm_extractor import BaseExtractor, parse_page_payload

FONT = Path(r"C:\Windows\Fonts\arial.ttf")


def _font(size):
    return ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()


@pytest.mark.skipif(not tesseract_cmd(), reason="Tesseract not installed")
def test_hybrid_page_with_native_footer_is_ocrd(tmp_path):
    # Product text exists only as pixels; the native layer is a long footer line.
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    d.text((100, 120), "AQUA FLASK 750", font=_font(64), fill="black")
    d.text((100, 220), "Stainless Steel Bottle", font=_font(44), fill="black")
    d.rectangle([700, 150, 1100, 700], fill=(40, 90, 160))
    d.text((100, 800), "MRP: Rs 1,299", font=_font(52), fill="black")
    buf = io.BytesIO(); img.save(buf, format="PNG")

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    page.insert_text((40, 830), "BRANDCO Corporate Gifting Catalogue 2026 | www.brandco.example | Page 7",
                     fontsize=7)
    path = tmp_path / "hybrid.pdf"; doc.save(str(path))

    pt = extract_page_text(enumerate_pages(path)[0])
    assert pt.text_source == "ocr"                     # not mistaken for a digital page
    assert "1,299" in pt.raw_text and "BRANDCO" in pt.raw_text


class GridExtractor(BaseExtractor):
    """Returns the 24 products of the generated grid with their (small) photo boxes."""
    provider = "fake"

    def __init__(self, cells):
        super().__init__("grid-vlm")
        self.cells = cells

    async def extract_page(self, image_bytes, page_number, native_text="", text_source="native"):
        return parse_page_payload({"products": [
            {"product_name": f"Pen Model {i}", "sku_or_model": f"PN-{100 + i}", "mrp": 150 + i * 10,
             "images": [{"box_2d": box, "type": "primary"}], "raw_text_snippet": f"Pen Model {i} PN-{100 + i}"}
            for i, box in self.cells]}, page_number)


def test_dense_grid_page_validates_small_photos(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    W, H = 595, 842
    swatch = io.BytesIO(); Image.new("RGB", (120, 120), (30, 120, 60)).save(swatch, format="PNG")
    cells = []
    for i in range(24):                                  # 4 columns × 6 rows
        col, row = i % 4, i // 4
        x0, y0 = 20 + col * 142, 20 + row * 135
        photo = fitz.Rect(x0 + 20, y0, x0 + 100, y0 + 80)  # ≈1.3% of the page
        page.insert_image(photo, stream=swatch.getvalue())
        page.insert_text((x0, y0 + 95), f"Pen Model {i} PN-{100 + i}", fontsize=8)
        page.insert_text((x0, y0 + 108), f"MRP Rs {150 + i * 10}", fontsize=8)
        cells.append((i, [photo.y0 / H * 1000, photo.x0 / W * 1000, photo.y1 / H * 1000, photo.x1 / W * 1000]))
    path = tmp_path / "grid.pdf"; doc.save(str(path))

    opts = PipelineOptions(output_root=tmp_path / "out", remove_background=False, skip_non_product=False)
    job = asyncio.run(run_pipeline(path, opts, job_id="grid", tiers=[GridExtractor(cells)]))
    assert job.status == "done", job.error_message
    assert job.validated_products == 24, json.loads((tmp_path / "out/grid/exceptions.json").read_text())["products"][:2]
    clean = json.loads((tmp_path / "out/grid/catalog_clean.json").read_text(encoding="utf-8"))["products"]
    assert all(p["image_paths"] for p in clean)           # every small grid photo was cropped
    assert {p["mrp"] for p in clean} == {150 + i * 10 for i in range(24)}
