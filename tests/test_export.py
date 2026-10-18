import asyncio
import csv
import io
import zipfile

import openpyxl

from app.export_zip import normalize_categories, safe_name, build_zip
from app.pipeline import PipelineOptions, run_pipeline
from tests.test_pipeline import FakeExtractor, catalog_pdf  # noqa: F401  (fixture)


def test_zip_has_category_brand_product_folders(catalog_pdf, tmp_path):
    opts = PipelineOptions(output_root=tmp_path / "out", brand="Acme Gifts")
    job = asyncio.run(run_pipeline(catalog_pdf, opts, job_id="zipjob", tiers=[FakeExtractor()]))
    assert job.status == "done"

    zip_path = build_zip(tmp_path / "out" / "zipjob")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # brand not printed on the page → falls back to the brand given at upload
        assert "Drinkware/Acme Gifts/Red Ceramic Mug (RM-330)/RM-330_1.png" in names
        assert "Drinkware/Acme Gifts/Red Ceramic Mug (RM-330)/original/RM-330_1.jpg" in names
        assert {"products.xlsx", "products.csv", "needs_review.xlsx"} <= set(names)

        rows = list(csv.DictReader(io.StringIO(zf.read("products.csv").decode("utf-8-sig"))))
        assert rows[0]["brand"] == "Acme Gifts" and rows[0]["mrp"] == "1450.0"
        assert rows[0]["images"] == "Drinkware/Acme Gifts/Red Ceramic Mug (RM-330)/RM-330_1.png"

        wb = openpyxl.load_workbook(io.BytesIO(zf.read("products.xlsx")))
        links = [c.hyperlink.target for row in wb.active.iter_rows(min_row=2) for c in row if c.hyperlink]
        assert links == ["Drinkware/Acme Gifts/Red Ceramic Mug (RM-330)/RM-330_1.png"]


def test_category_spellings_are_merged():
    recs = [{"category": "LUNCH BOX"}, {"category": "Lunch Boxes"}, {"category": "lunch box"}, {"category": None}]
    mapping = normalize_categories(recs)
    assert mapping["LUNCH BOX"] == mapping["Lunch Boxes"] == mapping["lunch box"] == "Lunch Box"
    assert mapping[""] == "Uncategorized"


def test_safe_names():
    assert safe_name('Pen: "Pro" / 2*') == "Pen Pro 2"
    assert safe_name("  ...  ") == "Unnamed"
