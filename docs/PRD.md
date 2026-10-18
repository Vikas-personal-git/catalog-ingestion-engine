> Original design spec for the MVP. The implementation has since moved past it in places (a multi-model cost ladder instead of a single VLM, OCR-backed verification for scanned pages, persistent catalog diffing). The [README](../README.md) describes the system as built.

# Technical Specification & Product Requirements Document (PRD)

## Project Name: Catalog Ingestion Engine (autonomous brand catalog ingestion)

---

## 1. Problem Statement

B2B corporate gifting, merchandising, and procurement platforms work with hundreds of supplier brands. Each brand releases periodic, unstructured PDF catalogs ranging from 20 to 300+ pages.

* These catalogs have **zero standardized layouts**, inconsistent formatting, missing tables, multiple photos per SKU, and unannounced price/product updates.
* Current workflows rely on dedicated listing personnel who manually read pages, copy/paste specs into Excel/admin portals, take screenshot cutouts, use Canva or Photoshop to strip backgrounds, and upload assets.
* This process is slow (20–40 products/day), expensive, and error-prone.
* **Core Constraint:** In B2B commerce, data inaccuracy (wrong MOQ, inverted pricing, misattributed SKU images) causes catastrophic margin loss or unfulfillable client commitments. **Accuracy cannot be compromised.**

---

## 2. Vision & Product Objective

Build an autonomous ingestion engine that processes raw brand PDF catalogs, extracts structured product specifications with dual-validation consistency checks, crops and cleans product photography down to transparent backgrounds, and automatically separates high-confidence records from edge-case exceptions.

The objective is to eliminate dedicated manual data entry, enabling an operational manager to ingest complete catalogs in minutes with **zero human intervention for validated items** and strict quarantine for anomalies.

---

## 3. Core Principles & Accuracy Guardrails

1. **Never Hallucinate / Never Guess:** If a specification (e.g., MOQ, HSN code, dimensions) is absent from the page, output `null`—never extrapolate.
2. **Dual-Engine Consensus Verification:** Visual bounding box data must cross-reference native underlying PDF digital text streams. If numbers (e.g., MRP, Price, MOQ) do not match the exact character sequences found in the PDF stream, confidence drops and the record is flagged.
3. **Fail-Safe Quarantine:** A record must satisfy strict deterministic rules before it is auto-promoted. Any ambiguous item is routed to an exception log rather than corrupting the database.

---

## 4. Architecture & Technical Pipeline

```
[Raw Brand PDF]
       │
       ▼
[Stage 1: Page Renderer & Text Stream Extractor]
  ├── PyMuPDF: Render page to high-res PNG (300 DPI)
  └── PyMuPDF: Extract text blocks + bounding coordinates
       │
       ▼
[Stage 2: Vision-LLM Structured Extractor]
  └── VLM (Gemini 1.5 Pro / GPT-4o / Claude 3.5 Sonnet)
      Prompted with page image + JSON Schema
      Extracts: Titles, SKUs, Prices, Specs, Image Coordinates [ymin, xmin, ymax, xmax]
       │
       ▼
[Stage 3: Verification & Cross-Consistency Engine]
  ├── Text Consensus: Match VLM numbers against PyMuPDF native text
  ├── Price Sanity: Unit Cost <= MRP, Numeric format validation
  └── Image Validity: Bounding box dimensions & aspect ratio checks
       │
       ├── PASS (Score >= 0.95) ──► [Stage 4: Automated Image Processing]
       │                               ├── Crop via PIL/OpenCV (with 3% padding)
       │                               ├── Background removal via RMBG-1.4 / BiRefNet
       │                               └── Save transparent PNG to storage
       │                                       │
       │                                       ▼
       │                            [Stage 5: Deduplication & DB Write]
       │                            Check existing SKUs -> Auto-Publish
       │
       └── FAIL (Score < 0.95)  ──► [Quarantine & Exception Flagging]

```

---

## 5. Detailed End-to-End Workflow

### Step 1: Ingestion & Decomposition

* Ingest PDF via CLI or API.
* Use `fitz` (`PyMuPDF`) to split into individual pages:
* Extract page image at 300 DPI (`page.get_pixmap(dpi=300)`).
* Extract raw textual layout blocks (`page.get_text("blocks")`).


* Heuristic skip: Pages containing only cover art, table of contents, or index (detected by high-level keyword presence or low text density with no price indicators) are bypassed.

### Step 2: Multi-Modal Extraction

* Send the page image to the Vision Model using structured function calling / Pydantic schema.
* **Target JSON Schema:**
```json
{
  "products": [
    {
      "product_name": "string",
      "sku_or_model": "string | null",
      "category": "string | null",
      "description": "string | null",
      "mrp": "number | null",
      "b2b_price": "number | null",
      "moq": "number | null",
      "color_variants": ["string"],
      "specifications": {
        "material": "string | null",
        "capacity_or_size": "string | null",
        "dimensions": "string | null"
      },
      "images": [
        {
          "box_2d": [ymin, xmin, ymax, xmax], 
          "type": "primary | variant | packaging"
        }
      ],
      "raw_text_snippet": "Exact text substring visible on the page for this product"
    }
  ]
}

```


*(Coordinates normalized on a 0–1000 scale).*

### Step 3: Deterministic Accuracy Verification Engine

Every product record must pass programmatic validation tests:

1. **Verbatim Text Check:** Does `raw_text_snippet` exist inside the page's native digital text? If the PDF is digitally readable and the text doesn't exist, flag potential hallucination.
2. **Numeric Cross-Check:** For any extracted `mrp`, `b2b_price`, or `moq`, regex search the surrounding coordinates in the native text stream for that exact number.
3. **Price Logic Rules:**
* If both `b2b_price` and `mrp` exist: assert `b2b_price <= mrp`.
* Assert prices are positive non-zero floats.


4. **Bounding Box Validation:**
* Assert `ymax > ymin` and `xmax > xmin`.
* Area threshold: Bounding box must cover between 2% and 80% of total page area (filters out micro-icons or full-page backgrounds).



### Step 4: Asset Processing

* Convert normalized box coordinates `[ymin, xmin, ymax, xmax]` to absolute pixel coordinates against the 300 DPI image.
* Crop image using PIL with a 3% safety margin.
* Run background removal using `briaai/RMBG-1.4` (open-source) or rembg to output a clean, transparent RGBA PNG.
* Save image file to `/output/images/{sku_or_id}_{index}.png`.

### Step 5: Catalog Diffing & Output

* Compare extracted SKU/Name against existing catalog:
* **Brand-New:** Create new SKU entry.
* **Existing Match:** Check if price/specs changed. If yes, record price delta. If unchanged, skip.


* Export outputs to:
* `catalog_clean.json` & `catalog_clean.xlsx` (Validated, high-confidence listings).
* `exceptions.json` (Flagged edge cases with exact failure reasons).



---

## 6. Edge Cases & Handling Strategies

| Edge Case | Description | Handling Strategy |
| --- | --- | --- |
| **Multi-Product Single Image** | One lifestyle photo shows 4 different color bottles or a gift set. | VLM outputs bounding boxes for individual items where distinct; if bundled, flags image type as `"group_shot"` and assigns to all color variants. |
| **Grid Tables vs. Floating Photos** | Table on left with 10 SKUs, photos on right with labels like "A, B, C". | Instruct VLM prompt to explicitly resolve letter/number callout markers (e.g., "Match Item A in table to Photo marked A"). |
| **Price Tiering** | PDF shows volume slabs: "50-100 pcs: ₹450, 100-500 pcs: ₹400". | Schema accepts `price_slabs: [{"min_qty": 50, "price": 450}, ...]`. Lowest tier sets standard `b2b_price`, minimum slab sets `moq`. |
| **Scanned/Flattened PDFs** | Older catalogs where text is rasterized with no native digital layer. | Mark `is_scanned: true`. Fallback to OCR (Tesseract / Cloud OCR) for Stage 3 cross-check instead of native stream. |
| **Color Variants with Swatches** | Page has 1 main product image and 5 tiny color circles. | Bounding box extractor classifies main item as `"primary"` and color dots as `"swatch"`. Pipeline only crops `"primary"` and `"variant"`. |

---

## 7. MVP Scope & Acceptance Criteria for Coding Agent

### In-Scope for Initial Version (CLI / Script-Based MVP)

1. **Input:** Single PDF file path (or folder of pages).
2. **Execution:**
* Process 10–20 test pages from real vendor catalogs.
* Page rendering & text parsing via `PyMuPDF`.
* Vision extraction using Gemini 1.5 Pro/Flash or GPT-4o API.
* Auto-crop and background removal via `rembg` or `RMBG-1.4`.


3. **Output:**
* `/output/data.xlsx`: Fully structured Excel sheet ready for catalog import.
* `/output/images/`: Transparent PNGs mapped to each row in the Excel sheet.
* `/output/exceptions.log`: Detailed log of any product that failed consensus checks.



### Acceptance Criteria

* **Accuracy:** 0% hallucinated price numbers. If a price cannot be matched to the raw text, it must be flagged in `exceptions.log`.
* **Image Cleanliness:** All extracted images must be cleanly cropped around the product without cutting off product edges, with white/patterned backgrounds stripped.
* **Speed:** Process a 20-page catalog batch in under 3 minutes asynchronously.

