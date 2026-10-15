"""Turn cached API records into scored items, summary metrics, and output files."""

from __future__ import annotations

import csv
import datetime as dt
import json
import time

import numpy as np

from . import config as C
from .checks import RULES, KeyFact, apply_guardrail, fact_covered, parse_plain_verdict
from .data import Item
from .metrics import cluster_bootstrap_ci, mcnemar, scores
from .prompts import PROMPT_VERSION

GUARD = "haiku_guardrail"
SMALL = "haiku_plain"
LARGE = "sonnet_plain"


def score_items(items: list[Item], records: dict[tuple[str, str], dict]) -> list[dict]:
    rows = []
    for item in items:
        row = {
            "item_id": item.item_id, "question_idx": item.question_idx, "gold": item.gold,
            "knowledge": item.knowledge, "question": item.question, "answer": item.answer,
            "configs": {}, "guardrail": None,
        }
        for cfg in C.configs():
            rec = records[(cfg.key, item.item_id)]
            latency = rec["latency_s"]
            if cfg.mode == "guardrail":
                t0 = time.perf_counter()
                decision = apply_guardrail(rec["text"], item.knowledge, item.answer)
                latency += time.perf_counter() - t0
                verdict, parsed_ok = decision.final_verdict, decision.model_verdict is not None
                row["guardrail"] = decision.to_dict()
            else:
                verdict, parsed_ok = parse_plain_verdict(rec["text"])
            row["configs"][cfg.key] = {
                "verdict": verdict,
                "correct": verdict == item.gold,
                "parsed_ok": parsed_ok,
                "input_tokens": rec["input_tokens"],
                "output_tokens": rec["output_tokens"],
                "cost_usd": C.cost_usd(cfg.model, rec["input_tokens"], rec["output_tokens"]),
                "latency_s": latency,
                "stop_reason": rec["stop_reason"],
                "model": rec["model"],
                "raw_text": rec["text"],
            }
        rows.append(row)
    return rows


def _explain(row: dict, kind: str) -> str:
    g = row["guardrail"]
    plain = row["configs"][SMALL]["verdict"]
    parts = []
    if kind == "caught":
        parts.append(f"Plain Haiku judged this hallucinated answer as **{plain}**.")
    else:
        parts.append("The gold label says this answer is **correct**, but the guardrail config rejected it.")
    if g["overridden"]:
        parts.append(f"In the guardrail config Haiku also said **{g['model_verdict'] or 'nothing parseable'}**, "
                     f"and code overrode it via `{g['rule']}`: {g['detail']}.")
        if g["rule"] == "uncovered_key_fact":
            for f in g["facts"]:
                if not f["covered"]:
                    where = ("does appear in the knowledge passage, so the model simply failed to cite it"
                             if fact_covered_in_knowledge(f, row["knowledge"]) else
                             "does not appear anywhere in the knowledge passage")
                    parts.append(f"The key fact \"{f['text']}\" {where}.")
    else:
        parts.append("Haiku itself returned **hallucinated** once it had to cite verbatim evidence; "
                     "no code override was needed.")
    return " ".join(parts)


def fact_covered_in_knowledge(fact: dict, knowledge: str) -> bool:
    return fact_covered(KeyFact(fact["text"], fact["kind"], fact["start"], fact["end"]), knowledge)


def _pick_examples(rows: list[dict]) -> dict:
    caught = [r for r in rows if r["gold"] == "hallucinated"
              and r["configs"][SMALL]["verdict"] == "supported"
              and r["configs"][GUARD]["verdict"] == "hallucinated"]
    refusals = [r for r in rows if r["gold"] == "supported" and r["configs"][GUARD]["verdict"] == "hallucinated"]

    def pick(pool: list[dict], k: int) -> list[dict]:
        # Prefer items where code (not the model) made the call, and spread across rules.
        pool = sorted(pool, key=lambda r: (not r["guardrail"]["overridden"], r["item_id"]))
        chosen, seen = [], set()
        for r in pool:
            rule = r["guardrail"]["rule"] if r["guardrail"]["overridden"] else None
            if rule not in seen:
                chosen.append(r)
                seen.add(rule)
            if len(chosen) == k:
                return chosen
        return chosen + [r for r in pool if r not in chosen][: k - len(chosen)]

    return {
        "caught": [{"item_id": r["item_id"], "explanation": _explain(r, "caught")} for r in pick(caught, 3)],
        "false_refusals": [{"item_id": r["item_id"], "explanation": _explain(r, "refusal")} for r in pick(refusals, 2)],
    }


def summarize(rows: list[dict], seed: int) -> dict:
    gold = np.array([r["gold"] == "hallucinated" for r in rows])
    clusters = np.array([r["question_idx"] for r in rows])
    preds = {c.key: np.array([r["configs"][c.key]["verdict"] == "hallucinated" for r in rows]) for c in C.configs()}
    boot = cluster_bootstrap_ci(gold, preds, clusters, C.BOOTSTRAP_RESAMPLES, seed)

    cfg_out = {}
    for c in C.configs():
        cs = [r["configs"][c.key] for r in rows]
        cost = sum(x["cost_usd"] for x in cs)
        cfg_out[c.key] = {
            "label": c.label, "model": c.model, "mode": c.mode,
            **scores(gold, preds[c.key]),
            "ci95": boot["configs"][c.key],
            "input_tokens": sum(x["input_tokens"] for x in cs),
            "output_tokens": sum(x["output_tokens"] for x in cs),
            "cost_usd": cost,
            "cost_per_100_items_usd": cost / len(rows) * 100,
            "avg_latency_s": float(np.mean([x["latency_s"] for x in cs])),
            "p50_latency_s": float(np.median([x["latency_s"] for x in cs])),
            "unparsed_replies": sum(not x["parsed_ok"] for x in cs),
            "refusals": sum(x["stop_reason"] == "refusal" for x in cs),
        }

    by_rule = {rule: {"overrides": 0, "correct_catches": 0, "false_refusals": 0} for rule in RULES}
    fired_any = {rule: 0 for rule in RULES}
    n_over = 0
    for r in rows:
        g = r["guardrail"]
        for rule in g["rules_fired"]:
            fired_any[rule] += 1
        if g["overridden"]:
            n_over += 1
            b = by_rule[g["rule"]]
            b["overrides"] += 1
            b["correct_catches" if r["gold"] == "hallucinated" else "false_refusals"] += 1
    model_only_flags = sum(1 for r in rows if r["guardrail"]["model_verdict"] == "hallucinated")
    changed_vs_plain = sum(1 for r in rows if r["guardrail"]["model_verdict"] != r["configs"][SMALL]["verdict"])
    # Was an uncovered fact genuinely absent from the passage, or just left out of the citation?
    diagnosis = {"fact_absent_from_knowledge": 0, "fact_in_knowledge_but_not_cited": 0}
    for r in rows:
        g = r["guardrail"]
        if g["overridden"] and g["rule"] == "uncovered_key_fact":
            missing = [f for f in g["facts"] if not f["covered"]]
            absent = any(not fact_covered_in_knowledge(f, r["knowledge"]) for f in missing)
            diagnosis["fact_absent_from_knowledge" if absent else "fact_in_knowledge_but_not_cited"] += 1

    return {
        "run": {
            "n_items": len(rows), "n_questions": int(len(set(clusters.tolist()))), "seed": seed,
            "date": dt.date.today().isoformat(), "dataset": "HaluEval QA (pminervini/HaluEval, qa)",
            "small_model": C.SMALL_MODEL, "large_model": C.LARGE_MODEL, "effort": C.EFFORT,
            "max_tokens": C.MAX_TOKENS, "prompt_version": PROMPT_VERSION,
            "prices_per_mtok": {m: C.PRICES_PER_MTOK[m] for m in {C.SMALL_MODEL, C.LARGE_MODEL}},
            "total_cost_usd": sum(v["cost_usd"] for v in cfg_out.values()),
        },
        "configs": cfg_out,
        "bootstrap": {k: v for k, v in boot.items() if k != "configs"},
        "mcnemar": {
            f"{GUARD}_vs_{SMALL}": mcnemar(gold, preds[GUARD], preds[SMALL]),
            f"{GUARD}_vs_{LARGE}": mcnemar(gold, preds[GUARD], preds[LARGE]),
            f"{LARGE}_vs_{SMALL}": mcnemar(gold, preds[LARGE], preds[SMALL]),
        },
        "guardrail": {
            "overrides_total": n_over,
            "by_rule": by_rule,
            "rules_fired_any": fired_any,
            "model_said_hallucinated": model_only_flags,
            "model_verdict_differs_from_plain_haiku": changed_vs_plain,
            "uncovered_fact_diagnosis": diagnosis,
            "note": "An override is an item where Haiku's own verdict was 'supported' (or unparseable) "
                    "and code changed it to 'hallucinated'. Each override is attributed to the first rule that fired.",
        },
        "examples": _pick_examples(rows),
    }


def write_outputs(rows: list[dict], summary: dict) -> None:
    C.RESULTS_DIR.mkdir(exist_ok=True)
    with open(C.RESULTS_DIR / "items.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (C.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    fields = ["item_id", "question_idx", "gold", "question", "answer"]
    for c in C.configs():
        fields += [f"{c.key}_verdict", f"{c.key}_correct", f"{c.key}_input_tokens",
                   f"{c.key}_output_tokens", f"{c.key}_latency_s"]
    fields += ["guardrail_model_verdict", "guardrail_overridden", "guardrail_override_reason",
               "guardrail_rules_fired", "guardrail_detail", "knowledge"]
    with open(C.RESULTS_DIR / "items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r[k] for k in ("item_id", "question_idx", "gold", "question", "answer", "knowledge")}
            for c in C.configs():
                x = r["configs"][c.key]
                out.update({f"{c.key}_verdict": x["verdict"], f"{c.key}_correct": int(x["correct"]),
                            f"{c.key}_input_tokens": x["input_tokens"], f"{c.key}_output_tokens": x["output_tokens"],
                            f"{c.key}_latency_s": round(x["latency_s"], 3)})
            g = r["guardrail"]
            out.update({"guardrail_model_verdict": g["model_verdict"] or "unparseable",
                        "guardrail_overridden": int(g["overridden"]),
                        "guardrail_override_reason": g["rule"] if g["overridden"] else "",
                        "guardrail_rules_fired": ";".join(g["rules_fired"]),
                        "guardrail_detail": g["detail"]})
            w.writerow(out)


def load_outputs() -> tuple[list[dict], dict]:
    rows = [json.loads(l) for l in open(C.RESULTS_DIR / "items.jsonl") if l.strip()]
    return rows, json.loads((C.RESULTS_DIR / "summary.json").read_text())
