"""
Configuration loader.
Reads settings from environment variables (or a .env file in the project root).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


# USD per 1M tokens (input, output). Sources: ai.google.dev/gemini-api/docs/pricing and Anthropic
# model pricing, Oct 2026. Gemini 3.8 Flash is introductory pricing until 2026-12-31 (doubles after).
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
}

# Escalation ladder: every page starts on the first tier; a page moves up only when the
# deterministic verifier (or the provider) fails it. "provider:model" entries, comma-separated.
DEFAULT_TIERS = "gemini:gemini-3.5-flash-lite,gemini:gemini-3.8-flash,claude:claude-opus-5"


@dataclass
class Settings:
    # ── VLM providers / cascade ──────────────────────────────────────────────
    tiers: str = field(default_factory=lambda: _env("VLM_TIERS", DEFAULT_TIERS))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    claude_effort: str = field(default_factory=lambda: _env("CLAUDE_EFFORT", "high"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_thinking: str = field(default_factory=lambda: _env("GEMINI_THINKING", "low"))
    # Display only: converts the USD cost computed from provider token counts into ₹ for reports.
    usd_to_inr: float = field(default_factory=lambda: float(_env("USD_INR", "96")))
    # Hard ceiling on AI spend per job (USD); 0 = no cap. Pages left when it is reached are
    # reported as "needs re-run" — the job never spends past it.
    max_job_cost_usd: float = field(default_factory=lambda: float(_env("MAX_JOB_COST_USD", "2")))

    # ── Pipeline ─────────────────────────────────────────────────────────────
    max_concurrent_pages: int = field(default_factory=lambda: int(_env("MAX_CONCURRENT_PAGES", "6")))
    render_dpi: int = field(default_factory=lambda: int(_env("RENDER_DPI", "150")))
    pass_threshold: float = field(default_factory=lambda: float(_env("PASS_THRESHOLD", "0.95")))
    # Accept a product whose title text OCR cannot read (stylised logos / badges) when two different
    # models return identical names and values — every number must still be verified against the page.
    name_consensus: bool = field(default_factory=lambda: _env("NAME_CONSENSUS", "true").lower() == "true")
    rembg_model: str = field(default_factory=lambda: _env("REMBG_MODEL", "u2net"))
    # Product images smaller than this (long edge, px) are upscaled with Lanczos + light sharpening.
    min_image_px: int = field(default_factory=lambda: int(_env("MIN_IMAGE_PX", "1000")))
    # …and larger ones (high-resolution brand PDFs) are downscaled to this long edge.
    max_image_px: int = field(default_factory=lambda: int(_env("MAX_IMAGE_PX", "2000")))
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("OUTPUT_DIR", "output"))
    upload_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("UPLOAD_DIR", "uploads"))

    # ── Server ───────────────────────────────────────────────────────────────
    # Presentation mode for customer demos: hides AI cost, model names and internal downloads.
    demo_mode: bool = field(default_factory=lambda: _env("DEMO_MODE", "false").lower() == "true")
    # Local-only by default: the UI has no authentication. Use 0.0.0.0 only on a trusted network.
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "info"))

    def api_key_for(self, provider: str) -> str:
        return self.anthropic_api_key if provider == "claude" else self.gemini_api_key

    def tier_list(self, spec: str | None = None) -> List[Tuple[str, str]]:
        """Parse 'provider:model,provider:model' into [(provider, model), ...]."""
        tiers = []
        for item in (spec or self.tiers).split(","):
            item = item.strip()
            if not item:
                continue
            provider, _, model = item.partition(":")
            if provider not in ("claude", "gemini") or not model:
                raise ValueError(f"Invalid tier '{item}' — expected 'claude:<model>' or 'gemini:<model>'")
            tiers.append((provider, model))
        return tiers

    @property
    def master_catalog_path(self) -> Path:
        """Persistent catalog used for Stage 5 diffing across runs."""
        return self.output_dir / "master_catalog.json"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = MODEL_PRICING.get(model, (0.0, 0.0))
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out


settings = Settings()
