from app.pdf_renderer import TextBlock
from app.schemas import ExtractedProduct
from app.verifier import (bbox_problems, find_number, number_in_text, parse_numbers,
                          verify_product)

GOOD_BOX = [100, 100, 400, 400]  # 9% of page


def product(**kw) -> ExtractedProduct:
    base = dict(product_name="Steel Bottle", sku_or_model="SB-750", mrp=1450, b2b_price=999,
                raw_text_snippet="Steel Bottle SB-750", images=[{"box_2d": GOOD_BOX, "type": "primary"}])
    base.update(kw)
    return ExtractedProduct.model_validate(base)


PAGE = "Steel Bottle SB-750\n750 ml double wall\nMRP ₹1,450/-  Trade price: 999\nMOQ 50 pcs"


# ── number matching ──────────────────────────────────────────────────────────

def test_parse_numbers_handles_grouping_and_decimals():
    assert parse_numbers("₹1,450/- 1,00,000 450.00 Rs.75") == [1450, 100000, 450.0, 75]


def test_number_must_be_whole_token():
    assert number_in_text(450, "Price 450.00")
    assert not number_in_text(45, "Price 1450")        # substring of a longer number
    assert not number_in_text(450, "Price 1,450")
    assert not number_in_text(449.5, "Price 450")      # no rounding


def test_ocr_rupee_misread_only_after_price_label():
    assert find_number(2599, "MRP : 22599", ocr=True) == "ocr_currency"
    assert find_number(7999, "MRP: 27,999/-", ocr=True) == "ocr_currency"
    assert find_number(2599, "MRP : 22599", ocr=False) is None          # native text: strict
    assert find_number(2599, "Model 22599 cable", ocr=True) is None     # no price label


# ── verify_product ───────────────────────────────────────────────────────────

def test_clean_product_passes():
    v = verify_product(product(moq=50), PAGE)
    assert v.passed, v.failure_reasons
    assert v.confidence_score == 1.0


def test_hallucinated_price_is_quarantined():
    v = verify_product(product(b2b_price=899), PAGE)
    assert not v.passed
    assert any("b2b_price 899 not found" in r for r in v.failure_reasons)


def test_price_logic_violation_fails():
    v = verify_product(product(mrp=999, b2b_price=1450), PAGE)
    assert not v.passed
    assert any("price logic" in r for r in v.failure_reasons)


def test_hallucinated_snippet_fails():
    v = verify_product(product(raw_text_snippet="Premium copper flask with engraving"), PAGE)
    assert not v.passed


def test_unverifiable_scanned_page_is_quarantined():
    v = verify_product(product(), "", text_source="none")
    assert not v.passed
    assert any("cannot be verified" in r for r in v.failure_reasons)


def test_missing_values_are_not_guessed_and_still_pass():
    v = verify_product(product(mrp=None, b2b_price=None), PAGE)
    assert v.passed


def test_missing_photo_passes_with_warning():
    v = verify_product(product(images=[]), PAGE)
    assert v.passed and v.confidence_score == 0.95
    assert any("no product photo" in r for r in v.failure_reasons)


def test_price_slabs_set_moq_and_b2b_price():
    page = "Steel Bottle SB-750 MRP 1450\n50-100 pcs: ₹450\n100-500 pcs: ₹400"
    p = product(b2b_price=None, moq=None, price_slabs=[
        {"min_qty": 100, "max_qty": 500, "price": 400}, {"min_qty": 50, "max_qty": 100, "price": 450}])
    v = verify_product(p, page)
    assert p.moq == 50 and p.b2b_price == 450
    assert v.passed, v.failure_reasons


def test_conflicting_slab_values_fail():
    page = "Steel Bottle SB-750 MRP 1450\n50-100 pcs: ₹450 MOQ 100"
    p = product(b2b_price=450, moq=100, price_slabs=[{"min_qty": 50, "max_qty": 100, "price": 450}])
    v = verify_product(p, page)
    assert not v.passed


def test_ocr_page_tolerates_recognition_noise():
    ocr_page = "GIFTEECO\nMY CAFEII\nYour favourite, Ready in Minutes\nMRP: 27,999/-"
    p = product(product_name="My Cafe II", sku_or_model=None, mrp=7999, b2b_price=None,
                raw_text_snippet="MY CAFE II Your favourite, Ready in Minutes")
    v = verify_product(p, ocr_page, text_source="ocr")
    assert v.passed, v.failure_reasons
    assert any("OCR-misread" in r for r in v.failure_reasons)


def test_ocr_snippet_tolerates_confusions_and_line_order():
    ocr_page = "Ready in minutes\n300W Hybrid Lunchbox\nMRP: % 4,999/-\nHOT BOX Il"
    p = product(product_name="HOT BOX II", sku_or_model=None, mrp=4999, b2b_price=None,
                raw_text_snippet="HOT BOX II 300W Hybrid Lunchbox")
    assert verify_product(p, ocr_page, text_source="ocr").passed


def test_multi_product_page_requires_price_near_product():
    blocks = [
        TextBlock("Steel Bottle SB-750 MRP 1450", x0=50, y0=50, x1=300, y1=100),
        TextBlock("Copper Mug CM-200 MRP 999", x0=600, y0=800, x1=900, y1=850),
    ]
    page = "\n".join(b.text for b in blocks)
    near = product(b2b_price=None, raw_text_snippet="Steel Bottle SB-750",
                   images=[{"box_2d": [100, 50, 400, 300], "type": "primary"}])
    assert verify_product(near, page, blocks, products_on_page=2).passed
    far = product(b2b_price=999, mrp=None, raw_text_snippet="Steel Bottle SB-750",
                  images=[{"box_2d": [100, 50, 400, 300], "type": "primary"}])
    v = verify_product(far, page, blocks, products_on_page=2)
    assert not v.passed
    assert any("not near this product" in r for r in v.failure_reasons)


def test_elided_snippet_checks_every_segment():
    ok = verify_product(product(raw_text_snippet="Steel Bottle SB-750 ... MOQ 50 pcs"), PAGE)
    assert ok.passed, ok.failure_reasons
    bad = verify_product(product(raw_text_snippet="Steel Bottle SB-750 ... Free engraving included"), PAGE)
    assert not bad.passed


def test_tiny_variant_thumbnails_are_warnings_not_failures():
    p = product(images=[{"box_2d": GOOD_BOX, "type": "primary"},
                        {"box_2d": [500, 500, 560, 560], "type": "variant"}])  # 0.36% thumbnail
    v = verify_product(p, PAGE)
    assert v.passed
    assert any(r.startswith("warning: variant image skipped") for r in v.failure_reasons)


def test_unusable_primary_photo_fails():
    v = verify_product(product(images=[{"box_2d": [400, 100, 100, 400], "type": "primary"}]), PAGE)
    assert not v.passed


def test_name_must_be_printed_in_its_title_block():
    blocks = [
        TextBlock("HYDROBOIL GO", x0=60, y0=110, x1=280, y1=123),
        TextBlock("Rechargeable Kettle & Cup", x0=60, y0=130, x1=360, y1=144),
        TextBlock("MRP: 11,999/-", x0=60, y0=400, x1=200, y1=415),
        TextBlock("Pairs with any Flask or bottle", x0=60, y0=700, x1=400, y1=715),  # far from the title
    ]
    text = "\n".join(b.text for b in blocks)
    ok = product(product_name="HYDROBOIL GO Rechargeable Kettle & Cup", sku_or_model=None, mrp=11999,
                 b2b_price=None, raw_text_snippet="HYDROBOIL GO")
    assert verify_product(ok, text, blocks, products_on_page=1).passed
    wrong = product(product_name="HYDROBOIL GO Rechargeable Kettle & Flask", sku_or_model=None, mrp=11999,
                    b2b_price=None, raw_text_snippet="HYDROBOIL GO")
    v = verify_product(wrong, text, blocks, products_on_page=1)
    assert not v.passed and v.text_only_failure            # all numbers verified; only the name is wrong
    assert any("flask" in r for r in v.failure_reasons)


def test_price_belongs_to_nearest_product_on_a_spread():
    # Two-page spread: each product's price is at the very bottom of its own half, far from its photo.
    blocks = [
        TextBlock("BENTO BOX", x0=60, y0=470, x1=250, y1=500),
        TextBlock("MEAL PRO", x0=560, y0=470, x1=740, y1=500),
        TextBlock("MRP", x0=60, y0=896, x1=100, y1=905),
        TextBlock("1499", x0=60, y0=911, x1=140, y1=940),
        TextBlock("MRP", x0=560, y0=896, x1=600, y1=905),
        TextBlock("1999", x0=560, y0=911, x1=640, y1=940),
    ]
    text = "\n".join(b.text for b in blocks)

    def mk(name, mrp, x0):
        return ExtractedProduct.model_validate({
            "product_name": name, "mrp": mrp, "raw_text_snippet": name,
            "images": [{"box_2d": [100, x0, 420, x0 + 300], "type": "primary"}]})

    bento, meal = mk("BENTO BOX", 1499, 60), mk("MEAL PRO", 1999, 560)
    page = [bento, meal]
    assert verify_product(bento, text, blocks, products_on_page=2, page_products=page).passed
    assert verify_product(meal, text, blocks, products_on_page=2, page_products=page).passed

    swapped_bento, swapped_meal = mk("BENTO BOX", 1999, 60), mk("MEAL PRO", 1499, 560)
    page = [swapped_bento, swapped_meal]
    assert not verify_product(swapped_bento, text, blocks, products_on_page=2, page_products=page).passed
    assert not verify_product(swapped_meal, text, blocks, products_on_page=2, page_products=page).passed


# ── bounding boxes ───────────────────────────────────────────────────────────

def test_bbox_rules():
    assert bbox_problems(GOOD_BOX) == []
    assert bbox_problems([400, 100, 100, 400])               # inverted
    assert bbox_problems([0, 0, 50, 50])                     # 0.25% — micro icon
    assert bbox_problems([0, 0, 1000, 950])                  # 95% — full-page background
    assert bbox_problems([1, 2, 3])


def test_swatches_are_not_bbox_validated():
    p = product(images=[{"box_2d": GOOD_BOX, "type": "primary"}, {"box_2d": [0, 0, 20, 20], "type": "swatch"}])
    assert verify_product(p, PAGE).passed


def test_unknown_image_type_is_coerced():
    p = product(images=[{"box_2d": GOOD_BOX, "type": "Lifestyle"}])
    assert p.images[0].type == "group_shot"
