"""Markdown report: out/report.md."""

from __future__ import annotations

from pathlib import Path

from . import config as C
from .pipeline import GUARD, LARGE, SMALL


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def ci(v: list[float]) -> str:
    return f"[{100 * v[0]:.1f}, {100 * v[1]:.1f}]"


def p_str(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def mcnemar_sentence(m: dict, a: str, b: str) -> str:
    sig = "a statistically significant difference" if m["exact_p"] < 0.05 else "not statistically significant at α = 0.05"
    return (f"{a} vs. {b}: {p_str(m['exact_p'])}, {sig} "
            f"({m['a_right_b_wrong']} items only {a} got right, {m['a_wrong_b_right']} only {b} got right)")


def results_table(summary: dict) -> str:
    lines = [
        "| Config | Accuracy [95% CI] | F1 halluc. [95% CI] | Precision | Recall | False refusal rate | $ / 100 items | Avg latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, c in summary["configs"].items():
        name = f"**{c['label']}**" if key == GUARD else c["label"]
        lines.append(
            f"| {name} (`{c['model']}`) | {pct(c['accuracy'])} {ci(c['ci95']['accuracy'])} | "
            f"{pct(c['f1'])} {ci(c['ci95']['f1'])} | {pct(c['precision'])} | {pct(c['recall'])} | "
            f"{pct(c['false_refusal_rate'])} | ${c['cost_per_100_items_usd']:.4f} | {c['avg_latency_s']:.2f}s |"
        )
    return "\n".join(lines)


def _item_block(row: dict, explanation: str) -> str:
    g = row["guardrail"]
    spans = "\n".join(f"  - {'✅' if s['verbatim'] else '❌ not verbatim:'} \"{s['text']}\"" for s in g["spans"]) or "  - (none)"
    facts = ", ".join(f"{'✅' if f['covered'] else '❌'} {f['text']}" for f in g["facts"]) or "(none extracted)"
    verdicts = " · ".join(f"{summary_label(k)}: **{v['verdict']}**" for k, v in row["configs"].items())
    return (
        f"**`{row['item_id']}`** (gold: **{row['gold']}**)\n\n"
        f"- **Question:** {row['question']}\n"
        f"- **Answer:** {row['answer']}\n"
        f"- **Knowledge:** {row['knowledge']}\n"
        f"- **Verdicts:** {verdicts}\n"
        f"- **Cited evidence:**\n{spans}\n"
        f"- **Key facts:** {facts}\n\n"
        f"{explanation}\n"
    )


_LABELS = {c.key: c.label for c in C.configs()}


def summary_label(key: str) -> str:
    return _LABELS.get(key, key)


def write_report(rows: list[dict], summary: dict) -> Path:
    run, cfg, g = summary["run"], summary["configs"], summary["guardrail"]
    by_id = {r["item_id"]: r for r in rows}
    m31 = summary["mcnemar"][f"{GUARD}_vs_{SMALL}"]
    m32 = summary["mcnemar"][f"{GUARD}_vs_{LARGE}"]
    d31 = summary["bootstrap"]["differences"][f"{GUARD}-{SMALL}:accuracy"]
    d32 = summary["bootstrap"]["differences"][f"{GUARD}-{LARGE}:accuracy"]
    hg, hp, sp = cfg[GUARD], cfg[SMALL], cfg[LARGE]

    out = [
        "# Deterministic Guardrails vs. Plain LLMs on HaluEval",
        "",
        f"*{run['n_items']} items ({run['n_questions']} HaluEval QA questions × right/hallucinated answer) · "
        f"seed {run['seed']} · {run['date']} · small = `{run['small_model']}`, large = `{run['large_model']}` · "
        f"effort `{run['effort']}`*",
        "",
        "## Summary",
        "",
        f"Plain Haiku reached {pct(hp['accuracy'])} accuracy and plain Sonnet {pct(sp['accuracy'])}. "
        f"Haiku with the deterministic guardrail reached {pct(hg['accuracy'])} "
        f"(paired bootstrap 95% CI for the gain over plain Haiku: {ci(d31)} points; vs. Sonnet: {ci(d32)} points). "
        f"The guardrail overrode Haiku's verdict {g['overrides_total']} times; "
        f"{plural(sum(b['correct_catches'] for b in g['by_rule'].values()), 'was a correct catch', 'were correct catches')} and "
        f"{plural(sum(b['false_refusals'] for b in g['by_rule'].values()), 'was a false refusal', 'were false refusals')}. "
        f"Its false refusal rate was {pct(hg['false_refusal_rate'])} vs. {pct(hp['false_refusal_rate'])} for plain Haiku. "
        f"Per 100 items it cost ${hg['cost_per_100_items_usd']:.4f}, vs. ${sp['cost_per_100_items_usd']:.4f} for plain Sonnet "
        f"({sp['cost_per_100_items_usd'] / max(hg['cost_per_100_items_usd'], 1e-12):.1f}× more).",
        "",
        "## Results",
        "",
        results_table(summary),
        "",
        "Positive class = *hallucinated*. False refusal rate = share of correct answers flagged as hallucinated. "
        f"CIs: {summary['bootstrap']['n_resamples']} cluster-bootstrap resamples over questions (seed {summary['bootstrap']['seed']}), "
        "percentile method. Resampling questions rather than items keeps each question's right/hallucinated pair together.",
        "",
        "### Confusion matrices",
        "",
        "| Config | TP (caught) | FN (missed) | FP (false refusal) | TN |",
        "|---|---|---|---|---|",
        *[f"| {c['label']} | {c['confusion']['tp']} | {c['confusion']['fn']} | {c['confusion']['fp']} | {c['confusion']['tn']} |"
          for c in cfg.values()],
        "",
        "## McNemar's test (paired, same items)",
        "",
        f"- **{mcnemar_sentence(m31, 'Guardrail', 'plain Haiku')}.** "
        f"Exact binomial on {m31['n_discordant']} discordant items; chi-square with continuity correction "
        f"χ² = {m31['chi2_cc']:.2f}, {p_str(m31['chi2_cc_p'])}.",
        f"- {mcnemar_sentence(m32, 'Guardrail', 'plain Sonnet')}.",
        f"- {mcnemar_sentence(summary['mcnemar'][f'{LARGE}_vs_{SMALL}'], 'Plain Sonnet', 'plain Haiku')}.",
        "",
        "Caveat: McNemar treats the items as independent pairs, but each question contributes two items that share "
        "a knowledge passage. The cluster-bootstrap CI for the accuracy difference above accounts for that pairing.",
        "",
        "## Cost and latency",
        "",
        "| Config | Input tokens | Output tokens | Total $ | $ / 100 items | Avg latency | Median latency |",
        "|---|---|---|---|---|---|---|",
        *[f"| {c['label']} | {c['input_tokens']:,} | {c['output_tokens']:,} | ${c['cost_usd']:.4f} | "
          f"${c['cost_per_100_items_usd']:.4f} | {c['avg_latency_s']:.2f}s | {c['p50_latency_s']:.2f}s |" for c in cfg.values()],
        "",
        f"Total spend for this run: **${run['total_cost_usd']:.4f}**. Prices (USD / 1M tokens): "
        + ", ".join(f"`{m}` ${p['input']:.2f} in / ${p['output']:.2f} out" for m, p in run["prices_per_mtok"].items())
        + ". Output tokens include adaptive-thinking tokens. Guardrail latency includes the deterministic checks, "
        "which take well under a millisecond.",
        "",
        "## Guardrail override breakdown",
        "",
        "| Rule | Overrides | Correct catches | False refusals | Fired at all (incl. items Haiku already flagged) |",
        "|---|---|---|---|---|",
        *[f"| `{rule}` | {b['overrides']} | {b['correct_catches']} | {b['false_refusals']} | {g['rules_fired_any'][rule]} |"
          for rule, b in g["by_rule"].items()],
        f"| **Total** | **{g['overrides_total']}** | {sum(b['correct_catches'] for b in g['by_rule'].values())} | "
        f"{sum(b['false_refusals'] for b in g['by_rule'].values())} | |",
        "",
        f"Haiku's own verdict returned *hallucinated* on {g['model_said_hallucinated']} items in the guardrail config "
        f"(vs. {hp['confusion']['tp'] + hp['confusion']['fp']} in the plain config). Its own verdict differed from its "
        f"plain-config verdict on {g['model_verdict_differs_from_plain_haiku']} items, so the evidence prompt alone "
        "changes some answers before any rule fires.",
        "",
        "**Why the key-fact rule fired.** Of the overrides by `uncovered_key_fact`, "
        f"{g['uncovered_fact_diagnosis']['fact_absent_from_knowledge']} involved a fact that appears nowhere in the knowledge "
        "passage (the rule found an unsupported fact). "
        f"{g['uncovered_fact_diagnosis']['fact_in_knowledge_but_not_cited']} involved a fact that *is* in the passage but "
        "that Haiku left out of its citation, so the override came from strictness rather than from detecting the "
        "false claim. Both kinds count as correct catches when the gold label is hallucinated, but only the first "
        "is the mechanism working as intended.",
        "",
        "## Examples: the guardrail caught a hallucination plain Haiku missed",
        "",
    ]
    for ex in summary["examples"]["caught"]:
        out.append(_item_block(by_id[ex["item_id"]], ex["explanation"]))
    if not summary["examples"]["caught"]:
        out.append("_No such items in this run._\n")
    out += ["## Examples: the guardrail caused a false refusal", ""]
    for ex in summary["examples"]["false_refusals"]:
        out.append(_item_block(by_id[ex["item_id"]], ex["explanation"]))
    if not summary["examples"]["false_refusals"]:
        out.append("_No false refusals in this run._\n")
    out += [
        "## Limitations",
        "",
        f"- **Small sample.** {run['n_items']} items from {run['n_questions']} questions; the CIs above are wide.",
        "- **One subset.** HaluEval QA only (not dialogue, summarization, or general). Its hallucinated answers are "
        "often plausible rewrites that introduce a new entity, which suits an entity-coverage rule.",
        "- **Heuristic key-fact extraction.** Regex rules for numbers, dates, and capitalized spans. These miss spelled-out "
        "numbers and lowercase facts, and can catch sentence-initial words.",
        "- **Not a reproduction of CTGT's method.** This is an independent experiment testing a similar idea.",
        "",
    ]
    C.OUT_DIR.mkdir(exist_ok=True)
    path = C.OUT_DIR / "report.md"
    path.write_text("\n".join(out))
    return path
