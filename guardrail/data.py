"""Load HaluEval QA and build the balanced, seeded eval set."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import numpy as np

from .config import CACHE_DIR

HALUEVAL_RAW_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


@dataclass(frozen=True)
class Item:
    item_id: str        # e.g. "q4821-R" (right answer) / "q4821-H" (hallucinated answer)
    question_idx: int   # row index in HaluEval QA; the two items of a question share it
    knowledge: str
    question: str
    answer: str
    gold: str           # "supported" | "hallucinated"


def load_halueval_qa() -> list[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset("pminervini/HaluEval", "qa")["data"]
        return [dict(r) for r in ds]
    except Exception as e:  # noqa: BLE001 - any failure falls back to the official raw file
        print(f"[data] Hugging Face load failed ({e!r}); fetching raw JSON from GitHub")
    path = CACHE_DIR / "qa_data.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HALUEVAL_RAW_URL, path)
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_eval_set(n_items: int, seed: int) -> list[Item]:
    """Sample n_items/2 questions with a fixed seed; each yields one supported and one hallucinated item.

    The sample is a prefix of one seeded permutation, so a smaller run (e.g. --dry-run)
    uses a subset of the full run's items and its cache entries are reused.
    """
    if n_items % 2:
        raise ValueError("n_items must be even (two items per question)")
    rows = load_halueval_qa()
    order = np.random.default_rng(seed).permutation(len(rows))[: n_items // 2]
    items: list[Item] = []
    for idx in order:
        r = rows[int(idx)]
        items.append(Item(f"q{idx}-R", int(idx), r["knowledge"], r["question"], r["right_answer"], "supported"))
        items.append(Item(f"q{idx}-H", int(idx), r["knowledge"], r["question"], r["hallucinated_answer"], "hallucinated"))
    return items
