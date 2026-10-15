"""Run configuration: models, prices, seeds, paths, budget."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "out"
FONTS_DIR = ROOT / "assets" / "fonts"

SEED = 20261015
BOOTSTRAP_RESAMPLES = 1000
DEFAULT_N_ITEMS = 120  # 60 questions x (right answer, hallucinated answer)
DRY_RUN_N_ITEMS = 6
BUDGET_USD = 3.00
CONCURRENCY = 5
MAX_TOKENS = 4096
# Same effort for every config so the comparison only varies model and guardrail.
EFFORT = "low"

SMALL_MODEL = os.getenv("SMALL_MODEL", "claude-haiku-5-5")
LARGE_MODEL = os.getenv("LARGE_MODEL", "claude-sonnet-5-5")

# USD per 1M tokens, Anthropic first-party list prices as published on 2026-10-06
# (Claude Haiku 5.5 prices apply to prompts up to 100K tokens; ours are ~1K).
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

# Assumed output tokens per call for the pre-run budget estimate (includes any
# adaptive-thinking tokens, which are billed as output). Deliberately pessimistic.
EST_OUTPUT_TOKENS = {"plain": 600, "guardrail": 900}


@dataclass(frozen=True)
class RunConfig:
    key: str          # stable id used in cache paths and CSV columns
    label: str        # human label for report/dashboard
    model: str
    mode: str         # "plain" | "guardrail"


def configs() -> list[RunConfig]:
    return [
        RunConfig("haiku_plain", "Haiku · plain", SMALL_MODEL, "plain"),
        RunConfig("sonnet_plain", "Sonnet · plain", LARGE_MODEL, "plain"),
        RunConfig("haiku_guardrail", "Haiku + guardrail", SMALL_MODEL, "guardrail"),
    ]


def price(model: str) -> dict[str, float]:
    if model not in PRICES_PER_MTOK:
        raise KeyError(f"No price configured for {model!r}; add it to PRICES_PER_MTOK")
    return PRICES_PER_MTOK[model]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = price(model)
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
