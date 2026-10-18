"""
Stage 2 – Vision-LLM Structured Extractor.

Sends a rendered page image (plus the page's native text layer, when present) to a
vision LLM and gets back structured product records via JSON-schema structured output.

Providers (which models run, and in what order, is set by VLM_TIERS):
  - gemini: Google Gen AI SDK. Gemini natively emits 0-1000 [ymin, xmin, ymax, xmax].
  - claude: Anthropic Messages API. Boxes are requested in pixel coordinates of the image
    we send (Claude's coordinates map 1:1 to pixels) and normalized here to the 0-1000 scale.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from app.schemas import ExtractedProduct, PageExtractionResult

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """The VLM call failed or returned unusable output for a page."""


class FatalExtractionError(ExtractionError):
    """An account-level failure (bad key, no credits, unknown model): every further page would fail too."""


# ── VLM-facing output schema ─────────────────────────────────────────────────
# Kept separate from ExtractedProduct so every field is explicitly required-but-nullable
# in the JSON schema (the model must state `null` rather than silently omit a field).

class _VLMImage(BaseModel):
    box_2d: List[float] = Field(description="[ymin, xmin, ymax, xmax] — see coordinate rules")
    type: Literal["primary", "variant", "packaging", "swatch", "group_shot"]


class _VLMSpecs(BaseModel):
    material: Optional[str]
    capacity_or_size: Optional[str]
    dimensions: Optional[str]


class _VLMSlab(BaseModel):
    min_qty: int
    max_qty: Optional[int]
    price: float


class _VLMProduct(BaseModel):
    product_name: str = Field(description="model name exactly as printed in the title, with every suffix/badge")
    subtitle: Optional[str] = Field(description="descriptor line printed directly under the title, "
                                                "e.g. 'Digital Clock with Pen Stand'; null if none")
    sku_or_model: Optional[str]
    brand: Optional[str] = Field(description="brand name as printed on the page (logo text / header); null if none")
    category: Optional[str] = Field(description="catalog section / category printed on the page "
                                                "(header, side tab, section title), e.g. 'Lunch Box'; null if none")
    description: Optional[str]
    mrp: Optional[float]
    b2b_price: Optional[float]
    moq: Optional[int]
    color_variants: List[str]
    specifications: _VLMSpecs
    images: List[_VLMImage]
    price_slabs: List[_VLMSlab]
    raw_text_snippet: Optional[str]


class _VLMPage(BaseModel):
    products: List[_VLMProduct]


SYSTEM_PROMPT = """\
You extract structured product data from pages of B2B supplier catalogs. The output feeds an \
automated procurement catalog where a wrong price, MOQ or SKU causes real financial loss, so \
precision matters far more than coverage. Every number you output is later checked character-for-\
character against the PDF's text layer; anything that cannot be found there is quarantined.

Rules:
1. Never infer, compute, convert or guess. If a field is not printed on the page for that product, \
output null (or [] for lists). Copy numbers exactly as printed (drop currency symbols and thousands \
separators only).
2. Extract every distinct product / SKU the page sells. Pages with no products (cover, contents, \
lifestyle filler) return an empty products list. Do not extract cross-sell or "family" strips \
("Find your…", "Also available", comparison rows) that show other products as small thumbnails \
without their own price — those products are listed on their own pages.
   product_name: the model name exactly as printed in the product's title, with every suffix and \
badge (e.g. "NOVA DIGITAL", "HYDROBOIL MINI DIGITAL", "SOLARIS X LITE (MARS)"). subtitle: the \
descriptor line printed directly under that title (e.g. "3-in-1 Lamp, Digital Clock & Organizer"), \
or null. Take both from the title block only, never from family strips or feature text.
   brand: the brand as printed on the page (logo lettering or header). category: the catalog \
section the product sits in, as printed (page header, side tab or section title). Both null when \
not printed — do not infer them.
3. mrp is the printed MRP / list / retail price. b2b_price is the printed trade / dealer / \
corporate / net price. If only one unlabeled price is shown, put it in b2b_price and leave mrp null.
4. Volume pricing ("50-100 pcs: ₹450, 100-500 pcs: ₹400") goes into price_slabs, one entry per \
tier. Then set moq to the smallest tier's min_qty and b2b_price to the price of that smallest-\
quantity tier.
5. raw_text_snippet: copy one contiguous line of text exactly as printed for this product — its \
name / title line (with the SKU if printed on the same line). Do not paraphrase, abbreviate, use \
ellipses, or join text from different places on the page. If a native PDF text layer is provided below, copy the snippet from it verbatim; \
if only OCR text is provided, copy the snippet as printed in the image, not the OCR errors.
6. Tables with photos labelled A, B, C… (or 1, 2, 3…): match each table row to the photo with the \
same callout marker and attach that photo's box to the row's product.
7. images: one entry per photo that shows the whole product, tightly enclosing the product itself \
including all edges, handles, lids and cables that belong to it, but not neighbouring text, hands, \
props (cups, plants, food) or other products. Skip close-ups of a single feature or part, icons, \
logos and decorative backgrounds.
   - primary: the main product photo
   - variant: an alternate colour / angle / size photo of the same product
   - packaging: box or packaging shot
   - swatch: small colour dots / fabric swatches
   - group_shot: one photo showing several products or a bundled set; attach the same box to \
every product it depicts
8. {coordinate_rule}
"""

CLAUDE_COORDINATE_RULE = (
    "box_2d is [ymin, xmin, ymax, xmax] in PIXEL coordinates of the attached image, which is "
    "{width} px wide and {height} px tall; (0, 0) is the top-left corner."
)
GEMINI_COORDINATE_RULE = (
    "box_2d is [ymin, xmin, ymax, xmax] normalized to a 0-1000 scale; (0, 0) is the top-left corner."
)


def _user_text(page_number: int, native_text: str, text_source: str) -> str:
    parts = [f"This is page {page_number} of a B2B supplier product catalog. Extract all products."]
    if native_text.strip():
        label = "OCR text (may contain recognition errors)" if text_source == "ocr" else "native PDF text layer"
        parts.append(f"\n<page_text source=\"{label}\">\n{native_text.strip()}\n</page_text>")
    return "\n".join(parts)


def _fit_image(image_bytes: bytes, max_long_edge: int, max_pixels: int) -> tuple[bytes, int, int]:
    """Downscale the page so the provider never resizes it server-side; return JPEG bytes and size."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_long_edge / max(w, h), math.sqrt(max_pixels / (w * h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        return buf.getvalue(), img.width, img.height


def _merge_subtitle(record: dict) -> None:
    """Listing name = title + descriptor line ("SOLARIS X LITE (MARS) Digital Clock with Pen Stand")."""
    name = (record.get("product_name") or "").strip()
    sub = (record.pop("subtitle", None) or "").strip()
    if name and sub and sub.lower() not in name.lower():
        record["product_name"] = f"{name} {sub}"


def parse_page_payload(data: dict, page_number: int, to_norm=None) -> PageExtractionResult:
    """
    Validate each product record independently so one malformed record does not
    discard the rest of the page. `to_norm` converts a provider box to 0-1000 space.
    """
    if not isinstance(data, dict) or not isinstance(data.get("products"), list):
        raise ExtractionError(f"Page {page_number}: response has no 'products' list")

    result = PageExtractionResult()
    for record in data["products"]:
        try:
            if isinstance(record, dict):
                _merge_subtitle(record)
            if to_norm and isinstance(record, dict):
                for img in record.get("images") or []:
                    if isinstance(img, dict) and isinstance(img.get("box_2d"), list) and len(img["box_2d"]) == 4:
                        img["box_2d"] = to_norm(img["box_2d"])
            result.products.append(ExtractedProduct.model_validate(record))
        except (ValidationError, TypeError, ValueError) as exc:
            result.invalid_records.append({"record": record, "error": str(exc)})
    return result


class BaseExtractor:
    provider = "base"

    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    async def extract_page(
        self,
        image_bytes: bytes,
        page_number: int,
        native_text: str = "",
        text_source: str = "native",
    ) -> PageExtractionResult:
        raise NotImplementedError


class ClaudeExtractor(BaseExtractor):
    provider = "claude"
    # Opus 4.7+ / Sonnet 5 high-res vision limits; staying inside them keeps coordinates 1:1.
    HIGH_RES = (2576, 3_750_000)
    STANDARD_RES = (1568, 1_150_000)   # Haiku 4.5 and older models

    def __init__(self, api_key: str, model: str = "claude-opus-5", effort: str = "high"):
        super().__init__(model)
        import anthropic
        from anthropic import transform_schema

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None, max_retries=4)
        self._schema = transform_schema(_VLMPage)
        self._effort = effort
        self._legacy = model.startswith("claude-haiku-4") or "-4-5" in model  # no effort / adaptive thinking
        self.MAX_LONG_EDGE, self.MAX_PIXELS = self.STANDARD_RES if self._legacy else self.HIGH_RES

    def _request_kwargs(self) -> dict:
        fmt = {"format": {"type": "json_schema", "schema": self._schema}}
        if self._legacy:
            return {"output_config": fmt}
        kwargs = {"thinking": {"type": "adaptive"}, "output_config": {"effort": self._effort, **fmt}}
        if self.model.startswith(("claude-opus-5", "claude-fable")):
            # Server-side refusal fallback: a declined page is retried on a fallback model.
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return kwargs

    async def extract_page(self, image_bytes, page_number, native_text="", text_source="native"):
        jpeg, w, h = _fit_image(image_bytes, self.MAX_LONG_EDGE, self.MAX_PIXELS)
        system = SYSTEM_PROMPT.format(coordinate_rule=CLAUDE_COORDINATE_RULE.format(width=w, height=h))
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpeg).decode()}},
            {"type": "text", "text": _user_text(page_number, native_text, text_source)},
        ]
        anthropic = self._anthropic
        try:
            async with self._client.beta.messages.stream(
                model=self.model,
                max_tokens=32000,
                system=system,
                messages=[{"role": "user", "content": content}],
                **self._request_kwargs(),
            ) as stream:
                message = await stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise FatalExtractionError("Claude API key is invalid or missing (set ANTHROPIC_API_KEY)") from exc
        except anthropic.PermissionDeniedError as exc:
            raise FatalExtractionError(f"Claude API key lacks permission: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise FatalExtractionError(f"Claude model not found: {self.model}") from exc
        except anthropic.BadRequestError as exc:
            if "credit balance" in str(exc.message).lower():
                raise FatalExtractionError("Anthropic account is out of API credits — top up under "
                                           "Plans & Billing, then re-run the failed pages") from exc
            raise ExtractionError(f"Claude rejected the request: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise ExtractionError("Claude rate limit exceeded after retries") from exc
        except anthropic.APIStatusError as exc:
            raise ExtractionError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ExtractionError(f"Could not reach the Claude API: {exc}") from exc

        self.input_tokens += message.usage.input_tokens
        self.output_tokens += message.usage.output_tokens
        if message.stop_reason == "refusal":
            raise ExtractionError("Claude declined to process this page (refusal)")
        if message.stop_reason == "max_tokens":
            raise ExtractionError("Claude output was truncated (max_tokens) — page has too much content")

        text = next((b.text for b in message.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"Claude returned invalid JSON: {exc}") from exc

        def to_norm(box):
            ymin, xmin, ymax, xmax = (float(v) for v in box)
            return [ymin / h * 1000, xmin / w * 1000, ymax / h * 1000, xmax / w * 1000]

        result = parse_page_payload(data, page_number, to_norm)
        logger.info("Page %d: Claude extracted %d product(s).", page_number, len(result.products))
        return result


class GeminiExtractor(BaseExtractor):
    provider = "gemini"
    MAX_LONG_EDGE = 3072
    MAX_PIXELS = 9_000_000
    RETRIES = 5

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite", thinking: str = "low"):
        super().__init__(model)
        from google import genai
        from google.genai import errors, types

        self._types = types
        self._errors = errors
        self._client = genai.Client(api_key=api_key or None)
        self._schema = _VLMPage.model_json_schema()
        self._thinking = thinking

    async def extract_page(self, image_bytes, page_number, native_text="", text_source="native"):
        types = self._types
        jpeg, _, _ = _fit_image(image_bytes, self.MAX_LONG_EDGE, self.MAX_PIXELS)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(coordinate_rule=GEMINI_COORDINATE_RULE),
            response_mime_type="application/json",
            response_json_schema=self._schema,
            # Gemini 3 is tuned for its default temperature; small catalog print needs high media resolution.
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            thinking_config=types.ThinkingConfig(thinking_level=self._thinking.upper()),
        )
        contents = [
            types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            _user_text(page_number, native_text, text_source),
        ]
        last_exc: Exception | None = None
        for attempt in range(self.RETRIES):
            try:
                resp = await self._client.aio.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
                if resp.usage_metadata:
                    self.input_tokens += resp.usage_metadata.prompt_token_count or 0
                    self.output_tokens += (resp.usage_metadata.candidates_token_count or 0) + \
                        (resp.usage_metadata.thoughts_token_count or 0)
                data = json.loads(resp.text or "")
                result = parse_page_payload(data, page_number)
                logger.info("Page %d: Gemini extracted %d product(s).", page_number, len(result.products))
                return result
            except json.JSONDecodeError as exc:
                last_exc = exc
            except self._errors.ClientError as exc:
                if exc.code in (401, 403):
                    raise FatalExtractionError(f"Gemini API key rejected ({exc.code}): {exc.message}") from exc
                if exc.code == 404:
                    raise FatalExtractionError(f"Gemini model unavailable: {self.model} — {exc.message}") from exc
                if exc.code != 429:
                    raise ExtractionError(f"Gemini rejected the request ({exc.code}): {exc.message}") from exc
                last_exc = exc  # rate limited → back off and retry
            except Exception as exc:  # 5xx / transport errors
                last_exc = exc
            await asyncio.sleep(min(30, 2 ** (attempt + 1)))
        if isinstance(last_exc, self._errors.ClientError) and last_exc.code == 429:
            # Still throttled after ~1 minute of back-off: the key's quota is spent — stop using this
            # tier for the rest of the job instead of stalling every remaining page.
            raise FatalExtractionError(f"Gemini quota exhausted for {self.model} (429 after "
                                       f"{self.RETRIES} attempts) — use a paid-tier key or retry later") from last_exc
        raise ExtractionError(f"Gemini extraction failed after {self.RETRIES} attempts: {last_exc}") from last_exc


def create_extractor(provider: str, api_key: str, model: str,
                     effort: str = "high", thinking: str = "low") -> BaseExtractor:
    provider = (provider or "").lower()
    if provider == "claude":
        return ClaudeExtractor(api_key=api_key, model=model, effort=effort)
    if provider == "gemini":
        return GeminiExtractor(api_key=api_key, model=model, thinking=thinking)
    raise ValueError(f"Unknown VLM provider: {provider!r} (expected 'claude' or 'gemini')")
