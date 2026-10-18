"""
Pydantic schemas for Catalog Ingestion Engine.
These models are used as structured outputs from the VLM and across pipeline stages.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ImageType = Literal["primary", "variant", "packaging", "swatch", "group_shot"]

# Image types that are cropped + background-removed (PRD §6: swatches/packaging are not).
CROPPABLE_TYPES = {"primary", "variant", "group_shot"}

_IMAGE_TYPE_ALIASES = {
    "main": "primary", "hero": "primary", "product": "primary",
    "alternate": "variant", "alt": "variant", "color_variant": "variant",
    "package": "packaging", "box": "packaging",
    "color_swatch": "swatch", "colour_swatch": "swatch", "dot": "swatch",
    "group": "group_shot", "groupshot": "group_shot", "lifestyle": "group_shot", "set": "group_shot",
}


class PriceSlab(BaseModel):
    min_qty: int
    max_qty: Optional[int] = None
    price: float


class ImageRegion(BaseModel):
    """
    Bounding box normalized on 0-1000 scale: [ymin, xmin, ymax, xmax].
    type: primary | variant | packaging | swatch | group_shot
    """
    box_2d: List[float] = Field(..., description="[ymin, xmin, ymax, xmax] normalized 0-1000")
    type: ImageType = "primary"

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        # An unexpected label must not invalidate the whole product record.
        key = str(v or "primary").strip().lower().replace(" ", "_").replace("-", "_")
        key = _IMAGE_TYPE_ALIASES.get(key, key)
        return key if key in {"primary", "variant", "packaging", "swatch", "group_shot"} else "primary"


class Specifications(BaseModel):
    material: Optional[str] = None
    capacity_or_size: Optional[str] = None
    dimensions: Optional[str] = None


class ExtractedProduct(BaseModel):
    """Raw product data as extracted by the Vision LLM."""
    product_name: str
    sku_or_model: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    mrp: Optional[float] = None
    b2b_price: Optional[float] = None
    moq: Optional[int] = None
    color_variants: List[str] = Field(default_factory=list)
    specifications: Specifications = Field(default_factory=Specifications)
    images: List[ImageRegion] = Field(default_factory=list)
    price_slabs: List[PriceSlab] = Field(default_factory=list)
    raw_text_snippet: Optional[str] = None

    @field_validator("specifications", mode="before")
    @classmethod
    def _none_specs(cls, v):
        return v if v is not None else {}

    @field_validator("color_variants", "images", "price_slabs", mode="before")
    @classmethod
    def _none_list(cls, v):
        return v if v is not None else []


class PageExtractionResult(BaseModel):
    """Output from the VLM for one catalog page."""
    products: List[ExtractedProduct] = Field(default_factory=list)
    # Records the VLM returned that could not be parsed into ExtractedProduct.
    invalid_records: List[dict] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Outcome of the deterministic verification layer (Stage 3)."""
    confidence_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failure_reasons: List[str] = Field(default_factory=list)
    check_scores: Dict[str, float] = Field(default_factory=dict)
    is_scanned_pdf: bool = False
    text_source: Literal["native", "ocr", "none"] = "native"
    text_only_failure: bool = False   # every number verified; only title/snippet text unconfirmed
    consensus: Optional[str] = None   # set when accepted by agreement of two independent models


class ProcessedProduct(BaseModel):
    """A fully processed product record including asset paths and validation."""
    page_number: int
    source: str = ""                                   # file the page came from
    tier: str = ""                                     # model that produced the accepted extraction
    extracted: ExtractedProduct
    validation: ValidationResult
    image_paths: List[str] = Field(default_factory=list)      # relative to the job output dir
    original_image_paths: List[str] = Field(default_factory=list)   # same crops before background removal
    image_warnings: List[str] = Field(default_factory=list)
    background_removed: Optional[bool] = None
    # Stage 5 catalog diff
    change_status: Literal["new", "updated", "unchanged", "n/a"] = "n/a"
    changes: Dict[str, Dict[str, object]] = Field(default_factory=dict)  # field -> {old, new, delta}

    @property
    def is_new(self) -> bool:
        return self.change_status == "new"

    @property
    def price_changed(self) -> bool:
        return any(k in self.changes for k in ("mrp", "b2b_price"))


class PageIssue(BaseModel):
    page_number: int
    source: str = ""
    reason: str


class TierAttempt(BaseModel):
    tier: str
    outcome: str            # "accepted", or why the page moved up a tier


class PageResult(BaseModel):
    page_number: int
    source: str = ""
    preview: Optional[str] = None          # relative path of the page preview image
    tier_used: Optional[str] = None
    attempts: List[TierAttempt] = Field(default_factory=list)
    products: int = 0
    validated: int = 0
    needs_rerun: bool = False   # escalation was blocked by provider capacity, not by verification


class TierStats(BaseModel):
    tier: str
    pages_attempted: int = 0
    pages_accepted: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    disabled_reason: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "done", "error"]
    filename: str
    created_at: str = ""
    brand: str = ""            # brand given at upload — fallback when a page does not print one
    tiers: str = ""
    stage: str = ""
    total_pages: int = 0           # pages selected for VLM processing
    document_pages: int = 0        # pages in the source document(s)
    processed_pages: int = 0
    total_products: int = 0
    validated_products: int = 0
    exception_products: int = 0
    new_products: int = 0
    updated_products: int = 0
    unchanged_products: int = 0
    skipped_pages: List[PageIssue] = Field(default_factory=list)
    page_errors: List[PageIssue] = Field(default_factory=list)
    page_warnings: List[PageIssue] = Field(default_factory=list)   # e.g. possibly missed products
    pages: List[PageResult] = Field(default_factory=list)
    tier_stats: List[TierStats] = Field(default_factory=list)
    cost_usd: float = 0.0
    cost_inr: float = 0.0
    error_message: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    output_excel: Optional[str] = None
    output_json: Optional[str] = None
    output_exceptions: Optional[str] = None
    output_log: Optional[str] = None
