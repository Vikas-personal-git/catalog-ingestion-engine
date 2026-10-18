# Catalog Ingestion Engine

Turns unstructured B2B supplier PDF catalogs into a clean, import-ready product catalog:
structured data in Excel / JSON / CSV, plus transparent product cut-outs.

Vision models do the reading. Code decides whether each record is trusted: **every extracted
price must be found character-for-character in the page's own text** (the native PDF text layer,
or OCR for scanned pages). Anything that can't be verified is quarantined with a reason. A wrong
price or MOQ in B2B procurement costs real money, so the system leaves a field empty rather than
guess.

**Stack:** Python · FastAPI · asyncio · PyMuPDF · Tesseract OCR · Gemini & Claude (structured
output) · Pydantic · rembg / ONNX · openpyxl · pytest

---

## Results on real supplier catalogs

Measured on two real B2B catalogs. Aggregate numbers only: the catalogs belong to their brands
and aren't redistributed. Full breakdown in [docs/results.md](docs/results.md).

### Manual listing vs. this engine

**Catalog A**: 107 pages, 206 products, 469 product photos

| | Manual | Engine |
|---|---|---|
| Staff time | 55.4 h (≈ 7 working days) | **2.9 h** (−95%) |
| Processing | — | 16.6 min, unattended |
| Total cost | ₹3,992 | **₹272** (₹211 staff + ₹61 AI) |
| Cost per product | ₹19.4 | **₹1.3** |

**Catalog B**: 157 pages, 148 products, 218 product photos

| | Manual | Engine |
|---|---|---|
| Staff time | 28.2 h (≈ 3.5 working days) | **1.5 h** (−95%) |
| Processing | — | 22.5 min, unattended |
| Total cost | ₹2,037 | **₹130** (₹111 staff + ₹20 AI) |
| Cost per product | ₹13.8 | **₹0.9** |

With the engine, people only review the quarantined products and spot-check the validated
ones. Typing every product, and cropping and cleaning every photo, goes away. Engine time,
AI cost and the product and photo counts are measured. Manual effort is an estimate from these
assumptions, the defaults in the job page's built-in calculator:

| Task | Minutes |
|---|---|
| Data entry per product (manual) | 6 |
| Crop + background removal per photo (manual) | 3 |
| Review per quarantined product (engine) | 2 |
| Spot-check per validated product (engine) | 0.3 |

Staff cost is ₹15,000 / month for 208 h (≈ ₹72 / h). These assumptions put a person at 30–42
products a day, in line with the typical 20–40 for manual catalog listing.

### Cheap cascade vs. premium-only model

The same 66 pages, compared with `python -m app.compare`:

| | Premium only (Claude Opus) | Cascade (Gemini first) |
|---|---|---|
| AI cost per page | ≈ $0.054 | **$0.0013** (~40× cheaper) |
| Products validated | 62 | 63 |

On the 61 products both runs validated: **MRP, B2B price and MOQ agreed 100%, with 0 value
disagreements.**

The only differences were product-name wording (title vs. title + subtitle) and three products
that one run validated and the other quarantined. No validated number differed.

### Full catalog runs

| Catalog | Pages | Products | Validated | Quarantined | AI cost | Cost / page | Time |
|---|---|---|---|---|---|---|---|
| A: consumer accessories (1 PDF) | 107 | 206 | 139 (67%) | 67 | $0.63 | $0.006 | 16.6 min |
| B: corporate gifting (10 PDF parts)\* | 157 | 148 | 120 (81%) | 28 | $0.21 | $0.0013 | 22.5 min |

\* On catalog B, the upper tiers ran out of quota and credits mid-run. They were disabled
automatically and the run finished on the cheapest tier, with affected pages flagged for re-run.

**Where the cascade spent its effort (catalog A):** of 106 pages, 46 were accepted at the
cheapest tier, 59 at the mid tier and 1 at the premium tier.

A quarantined record isn't an error. It means a price or title couldn't be confirmed against
the page text, so it's held for review instead of being published.

---

## The problem

Supplier catalogs have no standard layout. One brand prints one product per page, the next uses
dense grids, tables with lettered photo callouts, volume price slabs, or flattened 100-DPI
scans with no text layer at all. Done by hand, this means retyping specs into spreadsheets and
cutting out photos in an image editor, at roughly 20–40 products per person per day.

Using an LLM alone doesn't fix it: a model that reads `₹1,450` as `₹450` one time in a thousand
is unusable here. The design question was how to get model-level flexibility with deterministic
guarantees on the numbers.

## How it works

```
PDF / images
  │
  ├─ Stage 1  Page analysis                                              local, free
  │           native text layer, or Tesseract OCR for image-only pages;
  │           cover / TOC / index pages skipped before any model spend
  │
  ├─ Stage 2  Vision-model cost ladder, per page
  │             cheap model ──verify──► accepted
  │                │ failed check · malformed output · unattributed price · provider down
  │             mid model ──verify──► accepted
  │                │ still failing (and not the same answer as the tier below)
  │             premium model ──verify──► accepted or quarantined
  │
  ├─ Stage 3  Deterministic verifier (no AI)
  │           exact price tokens near the product · name + snippet on the page ·
  │           price logic · photo box geometry · duplicate photos · prices nobody claimed
  │           score ≥ 0.95 and no hard failure → validated
  │
  ├─ Stage 4  Images
  │           crop at the source's native resolution · auto-widen cut-off crops ·
  │           background removal · upscale small photos (≤ 3×)
  │
  └─ Stage 5  Duplicate resolution · diff against a persistent master catalog · exports
```

## Design decisions

- **The verifier is the source of truth, not the model.** Every record passes the same
  deterministic gate whichever model produced it. Model choice only changes cost, never what
  gets accepted.
- **Cost ladder instead of one big model.** Each page starts on the cheapest tier and moves up
  only when verification fails. Escalation stops early when two tiers return the same answer
  with the same failures: the problem is then the page (e.g. unreadable print), and paying for
  a stronger model won't change it.
- **Exact numeric matching.** `450` matches `450.00` or `₹450/-` but never `1450`. On
  multi-product pages the number must also sit near *this* product's text, so a price can't be
  borrowed from the next item. One known OCR quirk is handled explicitly: Tesseract often reads
  `₹` as a leading digit. That case is allowed and flagged on the record.
- **Recall check, not just precision.** A price label on the page that no extracted product
  accounts for triggers escalation, and if it persists it's reported as a possibly missed
  product.
- **Fail-safe by default.** Missing fields are `null`. Volume slabs set `moq` / `b2b_price` in
  code, not by the model. Pages blocked by provider quota are reported as "needs re-run", not as
  verification failures, and the CLI prints the exact re-run command.
- **Bounded spend.** A per-job USD cap (`MAX_JOB_COST_USD`), live per-tier token and cost
  accounting, and a comparison tool that shows whether a cheaper configuration matches a premium
  reference run.
- **Responsive under load.** Blocking work (rendering, OCR, background removal) runs in worker
  threads behind a page-level semaphore, so the async event loop and web UI stay responsive
  while pages are processed concurrently.

## Quick start

**Requirements:** Python 3.11+ (developed on 3.14), and an API key for Gemini and/or Claude.

```bash
git clone https://github.com/Vikas-personal-git/catalog-ingestion-engine.git
cd catalog-ingestion-engine

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # Windows: copy .env.example .env
# then set GEMINI_API_KEY and/or ANTHROPIC_API_KEY in .env
```

Tiers whose provider has no key are skipped, so a single key is enough. With only a Gemini key,
set `VLM_TIERS` to Gemini models only (there's an example in `.env.example`).

**System dependencies** (optional but recommended):

| Component | Needed for | Windows | macOS | Debian / Ubuntu |
|---|---|---|---|---|
| Tesseract OCR 5 | scanned / image-only PDFs | `winget install UB-Mannheim.TesseractOCR` | `brew install tesseract` | `apt install tesseract-ocr` |
| VC++ Redistributable x64 | rembg / onnxruntime | `winget install Microsoft.VCRedist.2015+.x64` | — | — |

`GET /api/health` reports which model tiers have keys and whether OCR and background removal are
ready.

## Usage

### Web UI

```bash
python -m app.main                # uses HOST / PORT from .env (default 127.0.0.1:8000)
```

Open http://127.0.0.1:8000, upload one or more PDFs (or a whole folder) and follow progress and
live AI cost. You can then browse validated products with full-size cut-outs and a
"view on page" overlay of the extracted boxes, see quarantined products with reasons, and check
which model handled each page. Jobs persist to `output/<job_id>/job.json` and survive restarts.

### CLI

```bash
python -m app catalogs/                               # every PDF / image in a folder (identical copies skipped)
python -m app catalog.pdf --pages 1-20 --brand Acme   # page range, brand-scoped catalog matching
python -m app catalog.pdf --tiers claude:claude-opus-5 # single model, no cascade
python -m app catalog.pdf --no-catalog-update          # dry run: diff without updating the master catalog
python -m app --help
```

### Utilities

| Command | Purpose |
|---|---|
| `python -m app.compare output/<reference> output/<candidate>` | Accuracy of a cheaper config against a reference run: field agreement, value disagreements on products both validated (must be zero), coverage gaps |
| `python -m app.reimage output/<job> <original input>` | Redo only the image stage (no AI calls, no cost) |
| `python -m app.export_zip output/<job>` | Folder-structured hand-off ZIP: `Category/Brand/Product/` images plus `products.xlsx` / `.csv` |

## Outputs

Each job writes to `output/<job_id>/`:

| File | Contents |
|---|---|
| `catalog_clean.xlsx` | Validated products (thumbnail, hyperlinked image, new / updated / unchanged status) and an Exceptions sheet |
| `catalog_clean.json` | Validated products with per-check scores and the model tier used |
| `images/{sku_or_id}_{n}.png` | Transparent product cut-outs (originals kept alongside) |
| `pages/p{n}.jpg` | Page previews for review in the UI |
| `exceptions.json` / `exceptions.log` | Quarantined products, page errors, possibly missed products, model escalations |
| `job.json` | Job status, per-tier usage and cost |

`output/master_catalog.json` is the persistent catalog used to classify products as new, updated
(with price deltas) or unchanged across runs, scoped by brand.

## Supported inputs

Nothing is template- or brand-specific. The pipeline detects each page's type:

| Page type | Handling |
|---|---|
| Digital (native text) | Exact text layer used for verification |
| Image-only / scanned | Tesseract OCR (layout and sparse-text passes merged) |
| Hybrid (flattened image + thin text layer, e.g. a footer) | Native lines **plus** OCR |
| Any density (1 product to dense grids) | Photo-size floor scales with products per page |
| Any resolution (≈100 DPI flattened to 600 DPI originals, vectors) | Crops at the source's native resolution, normalised to `MIN_IMAGE_PX`–`MAX_IMAGE_PX` |

Accepted inputs are PDFs and page images (PNG / JPG / WEBP / TIFF), as single files or whole
folders.

**Known limits:** OCR languages are limited to the installed Tesseract packs (`OCR_LANGS`,
default English). Numbers are parsed with Indian / Western grouping (`1,00,000` / `100,000`);
European decimal commas (`1.234,50`) aren't supported. Image quality is capped by the source:
low-resolution photos are upscaled and flagged as such.

## Configuration

Everything is set through environment variables. [`.env.example`](.env.example) is the
documented reference. The main settings:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` | — | Provider keys; tiers without a key are skipped |
| `VLM_TIERS` | Gemini Flash-Lite → Gemini Flash → Claude Opus | The cost ladder, cheapest first |
| `MAX_JOB_COST_USD` | `2` | Hard spend cap per job (`0` = none) |
| `PASS_THRESHOLD` | `0.95` | Minimum verification score to validate a record |
| `NAME_CONSENSUS` | `true` | Accept OCR-unreadable titles when two models agree exactly (numbers still verified) |
| `REMBG_MODEL` | `u2net` | Background-removal model |
| `OCR_LANGS` | `eng` | Tesseract languages, e.g. `eng+hin` |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Web server bind address |

Model prices used for cost reporting are in `MODEL_PRICING` in [`app/config.py`](app/config.py).
Use paid-tier API keys for real catalogs: free tiers have small daily quotas.

> **Security note:** the web UI has no authentication and is meant for local use. Keep the
> default `HOST=127.0.0.1` unless you're on a trusted network.

## Tests

```bash
python -m pytest
```

The suite runs the whole pipeline end to end, including the escalation ladder, on a generated
PDF with fake model clients. It makes no API calls and needs no keys. It also covers the
verifier rules, layout handling, image processing, exports and the web routes.

## Project structure

```
app/
  __main__.py, cli.py   CLI entry point (python -m app)
  main.py               FastAPI server and UI routes
  pipeline.py           Async orchestrator and model cost ladder
  pdf_renderer.py       Stage 1: pages, text layer, OCR, skip heuristics, rendering
  vlm_extractor.py      Stage 2: Gemini / Claude structured extraction
  verifier.py           Stage 3: deterministic verification
  image_processor.py    Stage 4: native-resolution crop and background removal
  catalog_store.py      Stage 5: duplicate resolution and master-catalog diff
  output_writer.py      Stage 5: Excel / JSON / log exports
  export_zip.py         Folder-structured hand-off package
  compare.py            Accuracy comparison between two runs
  reimage.py            Re-run the image stage only
  schemas.py            Pydantic models shared across stages
  config.py             Environment-driven settings and model pricing
  templates/            Jinja2 UI (index, job)
tests/                  pytest suite (no network)
docs/results.md         Measured results on real catalogs
docs/PRD.md             Original design spec
```

## License

[MIT](LICENSE)
