# Results on real supplier catalogs

Measured on two real B2B supplier catalogs (October 2026). The catalogs belong to their brands
and aren't redistributed here, so this page reports aggregate numbers only: no product names or
prices. Costs come from provider-reported token counts and the per-model prices in
[`app/config.py`](../app/config.py).

## Manual listing vs. the engine

| | Catalog A (206 products, 469 photos) | Catalog B (148 products, 218 photos) |
|---|---|---|
| Manual staff time | 55.4 h (≈ 7 working days) | 28.2 h (≈ 3.5 working days) |
| Staff time with the engine | **2.9 h** | **1.5 h** |
| Manual work saved | **95%** | **95%** |
| Manual cost | ₹3,992 | ₹2,037 |
| Engine cost (staff + AI) | **₹272** (₹211 + ₹61) | **₹130** (₹111 + ₹20) |
| Cost per product, manual → engine | ₹19.4 → **₹1.3** | ₹13.8 → **₹0.9** |

Product counts, photo counts, processing time and AI cost are measured. Manual effort uses the
defaults of the built-in calculator on the job page:

- 6 min of data entry per product
- 3 min to crop and remove the background per photo
- 2 min to review each quarantined product
- 0.3 min to spot-check each validated product
- ₹15,000 / month for 208 h of staff time

Manually, every product is typed in and every photo is cropped and cleaned. With the engine,
people only review quarantined products and spot-check validated ones. These assumptions imply
30–42 products per person per day manually, consistent with the typical 20–40.

## Catalog A: 107-page consumer accessories catalog (single PDF)

Full run with the default ladder (Gemini Flash-Lite → Gemini Flash → Claude Opus).

| Metric | Value |
|---|---|
| Pages | 107 (1 cover page skipped automatically, 106 processed) |
| Products extracted | 206 |
| Validated (every number verified against the page) | 139 (67%) |
| Quarantined with reasons | 67 |
| Page errors | 0 |
| Total AI cost | **$0.63**, about $0.006 per page |
| Wall-clock time | 16.6 min |

Where each page ended up on the ladder:

| Tier | Pages attempted | Pages accepted | Cost |
|---|---|---|---|
| gemini-3.5-flash-lite | 106 | 46 | $0.29 |
| gemini-3.8-flash | 63 | 59 | $0.31 |
| claude-opus-5 | 1 | 1 | $0.04 |

Most pages were settled by the two cheap tiers. The premium model was needed for one page.
The 33% quarantine rate is the verifier doing its job: records whose prices or titles couldn't
be confirmed against the page text were held back for review instead of being published.

## Catalog B: 157-page corporate-gifting catalog (10 PDF parts)

### Does the cheap cascade match a premium model?

The same 66 pages were run twice: once with the premium model only (Claude Opus) as the
reference, and once with the cascade. They were compared with `python -m app.compare`.

| Metric | Value |
|---|---|
| Pages compared | 66 |
| Products validated by both runs | 61 |
| MRP agreement | **100%** |
| B2B price agreement | **100%** |
| MOQ agreement | **100%** |
| **Value disagreements** (both validated, different numbers) | **0** |
| Product-name wording differences | 12, all title-only vs title + subtitle (e.g. `NAME` vs `NAME – Desk Lamp`) |
| Validated only by the reference | 1 (cascade rejected a photo box as too small) |
| Validated only by the cascade | 2 |

| | Premium only | Cascade |
|---|---|---|
| AI cost per page | ≈ $0.054 | **$0.0013** |

The cascade cost about **40× less per page** with zero disagreement on any validated price or
MOQ. That's the point of the design: the deterministic verifier, not the model, decides what's
accepted, so a cheaper model can't lower accuracy, only coverage. The premium figure is computed
from the reference run's token usage (618,563 input / 42,854 output tokens over 77 pages) at
Claude Opus 5 list prices.

### Full catalog run

| Metric | Value |
|---|---|
| Pages | 157 (1 skipped, 156 processed; duplicate PDF parts detected and skipped) |
| Products | 148, of which 120 validated (81%) and 28 quarantined |
| Catalog diff against an earlier run | 56 new, 28 updated (price deltas recorded), 36 unchanged |
| Possible missed products flagged | 2 pages |
| Total AI cost | $0.21, about $0.0013 per page |
| Wall-clock time | 22.5 min |

On this run the upper tiers were unavailable (Gemini quota and Anthropic credits ran out). The
pipeline disabled each one once retries were exhausted instead of stalling every page, finished on the cheapest
tier, and reported the affected pages as "needs re-run" rather than as verification failures.
