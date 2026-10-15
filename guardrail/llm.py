"""Anthropic API calls with an on-disk cache, bounded concurrency, and retry/backoff."""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path

import anthropic

from . import config as C
from .data import Item
from .prompts import PROMPT_VERSION, system_prompt, user_message

RETRYABLE_STATUS = {408, 409, 429}
MAX_ATTEMPTS = 7


def cache_path(cfg: C.RunConfig, item: Item) -> Path:
    return C.CACHE_DIR / cfg.key / cfg.model / f"{PROMPT_VERSION}__{item.item_id}.json"


def load_cached(cfg: C.RunConfig, item: Item) -> dict | None:
    p = cache_path(cfg, item)
    return json.loads(p.read_text()) if p.exists() else None


def estimate_cost(cfgs: list[C.RunConfig], items: list[Item]) -> tuple[float, int]:
    """Projected $ for all uncached calls (pessimistic ~3.5 chars/token, fixed output allowance)."""
    total, n_calls = 0.0, 0
    for cfg in cfgs:
        for item in items:
            if cache_path(cfg, item).exists():
                continue
            chars = len(system_prompt(cfg.mode)) + len(user_message(item.knowledge, item.question, item.answer))
            total += C.cost_usd(cfg.model, int(chars / 3.5) + 20, C.EST_OUTPUT_TOKENS[cfg.mode])
            n_calls += 1
    return total, n_calls


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)):
        return True
    return isinstance(e, anthropic.APIStatusError) and (e.status_code in RETRYABLE_STATUS or e.status_code >= 500)


def _retry_after(e: Exception) -> float | None:
    resp = getattr(e, "response", None)
    try:
        return float(resp.headers.get("retry-after")) if resp is not None else None
    except (TypeError, ValueError):
        return None


async def _call(client: anthropic.AsyncAnthropic, cfg: C.RunConfig, item: Item) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.perf_counter()
        try:
            resp = await client.messages.create(
                model=cfg.model,
                max_tokens=C.MAX_TOKENS,
                system=system_prompt(cfg.mode),
                messages=[{"role": "user", "content": user_message(item.knowledge, item.question, item.answer)}],
                output_config={"effort": C.EFFORT},
            )
        except Exception as e:  # noqa: BLE001
            if not _is_retryable(e) or attempt == MAX_ATTEMPTS:
                raise
            delay = _retry_after(e) or min(60.0, 2 ** attempt) * (0.5 + random.random())
            print(f"  [retry] {cfg.key} {item.item_id}: {type(e).__name__}; sleeping {delay:.1f}s")
            await asyncio.sleep(delay)
            continue
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {
            "config": cfg.key,
            "item_id": item.item_id,
            "model": resp.model,
            "prompt_version": PROMPT_VERSION,
            "text": text,
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "latency_s": latency,
            "attempts": attempt,
        }
    raise RuntimeError("unreachable")


async def run_calls(cfgs: list[C.RunConfig], items: list[Item]) -> dict[tuple[str, str], dict]:
    """Return {(config_key, item_id): record}; cached records are reused, new ones are written to disk."""
    results: dict[tuple[str, str], dict] = {}
    todo: list[tuple[C.RunConfig, Item]] = []
    for cfg in cfgs:
        for item in items:
            rec = load_cached(cfg, item)
            if rec is not None:
                results[(cfg.key, item.item_id)] = rec
            else:
                todo.append((cfg, item))
    print(f"[api] {len(results)} cached, {len(todo)} to call (concurrency {C.CONCURRENCY})")
    if not todo:
        return results

    sem = asyncio.Semaphore(C.CONCURRENCY)
    done = 0
    async with anthropic.AsyncAnthropic(max_retries=0) as client:

        async def worker(cfg: C.RunConfig, item: Item) -> None:
            nonlocal done
            async with sem:
                rec = await _call(client, cfg, item)
            p = cache_path(cfg, item)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rec, indent=1))
            results[(cfg.key, item.item_id)] = rec
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"[api] {done}/{len(todo)} calls done")

        await asyncio.gather(*(worker(c, i) for c, i in todo))
    return results
