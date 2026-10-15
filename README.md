# Deterministic Guardrails vs. Plain LLMs on HaluEval

**The idea:** let a small model reason, but hold it to evidence rules enforced in code that it can't get around. Haiku must return a verdict plus verbatim evidence spans from the knowledge passage. Plain Python then rejects any "supported" verdict whose spans aren't really in the passage, or whose cited evidence leaves out a key fact (name, number, date) from the answer. I compare this against plain Haiku and plain Sonnet on a balanced HaluEval QA sample, with paired statistics and per-item cost and latency.

![dashboard](out/dashboard.png)

Open `out/dashboard.html` in a browser. It is one self-contained file and makes no network requests.

## Run it

```bash
uv venv -p 3.12 && uv pip install -r requirements.txt   # once
.venv/bin/python run.py run                              # the whole experiment: data → API → metrics → report → dashboard
```

Put `ANTHROPIC_API_KEY=...` in `.env`. Every API response is cached under `cache/`, so a rerun makes no API calls and reproduces the same numbers. Other commands: `run --dry-run` (6 items), `run --n 200` (sample size, in items), `report` and `dashboard` (rebuild from `results/` with no API calls), and `fonts` (re-download the dashboard fonts). Tests: `.venv/bin/python -m pytest tests`.

## Results (120 items = 60 questions × {right, hallucinated} answer, seed 20261015)

| Config | Accuracy [95% CI] | F1 halluc. [95% CI] | Precision | Recall | False refusal rate | $ / 100 items | Avg latency |
|---|---|---|---|---|---|---|---|
| Haiku · plain (`claude-haiku-5-5`) | 89.2% [83.3, 94.2] | 88.7% [82.6, 94.0] | 92.7% | 85.0% | 6.7% | $0.0086 | 1.00s |
| Sonnet · plain (`claude-sonnet-5-5`) | 90.8% [85.8, 95.8] | 90.4% [85.0, 95.7] | 94.5% | 86.7% | 5.0% | $0.1442 | 1.27s |
| **Haiku + guardrail** (`claude-haiku-5-5`) | 90.8% [85.8, 95.8] | 90.8% [85.2, 95.8] | 91.5% | 90.0% | 8.3% | $0.0163 | 1.32s |

- **McNemar, guardrail vs. plain Haiku:** exact p = 0.69. The guardrail got 4 items right that plain Haiku missed, and plain Haiku got 2 right that the guardrail missed. **Not statistically significant.** Paired cluster-bootstrap 95% CI for the accuracy gain: −2.5 to +5.8 points.
- **Guardrail vs. plain Sonnet:** same point accuracy (p = 1.0) at about 1/9 the cost per item ($0.016 vs. $0.144 per 100 items).
- **Overrides:** code overruled Haiku 4 times, all through the key-fact rule: 3 correct catches and 1 false refusal. The fabricated-span and malformed-output rules never fired. Haiku's spans were always verbatim and its JSON always parsed.
- **Total API spend:** $0.20 for the full run plus the dry run (projected beforehand: $0.86; the budget guard aborts above $3).

Full write-up with examples: [`out/report.md`](out/report.md). Per-item data: [`results/items.csv`](results/items.csv). Every metric: [`results/summary.json`](results/summary.json).

### How to read this honestly

The headline, "small model + guardrail matches the large model at 1/9 the cost," is true of the point estimates. **The evidence behind it is thin**, and I checked why:

- **The gain is 2 net items out of 120**, and the CI includes zero.
- **None of the 4 overrides found a fact that was missing from the passage.** In every case the "uncovered" fact *was* in the passage; Haiku just didn't include it in its citation. So the rule succeeded through strictness, not by detecting the invented claim. (`uncovered_fact_diagnosis` in `summary.json` counts this automatically.)
- **2 of the 3 "catches" look like HaluEval label noise.** `q2251-H`: "Tyler Winklevoss founded ConnectU" is labelled hallucinated, but the passage says Tyler Winklevoss co-founded ConnectU and is the rower the question asks about. `q9284-H`: "Matt Thiessen is not a member" (of Guy) is also true according to the passage. The third, `q1891-H` ("The lead actress of The Doll Squad is Francine York"), does make an unsupported "lead actress" claim, but the rule fired on the uncited "Doll Squad", not on that claim.
- **The false refusal (`q8929-R`, answer "Monaco")** is the same mechanism working against us. Haiku cited "the band was formed in 1995", which leaves out the word "Monaco", so the code rejected a correct answer.

**My conclusion:** on this sample, a cheap deterministic layer costs almost nothing and doesn't hurt accuracy. But the current key-fact rule mostly acts as a citation-completeness check, and HaluEval QA at n = 120 can't tell whether that helps. The misses (e.g. "born in the early 1930s", "from Canada") are paraphrase-level hallucinations that regex coverage cannot see.

## Design choices (made before running; noted per the brief)

- **Data:** `pminervini/HaluEval`, config `qa` (10,000 rows), with a fallback to the raw JSON on the official GitHub repo. 60 questions are drawn as the prefix of one seeded permutation, so the dry run's items are a subset of the full run's and reuse its cache. `--n` counts items, not questions.
- **Models:** `claude-haiku-5-5` and `claude-sonnet-5-5` (overridable via `SMALL_MODEL` / `LARGE_MODEL` in `.env`). All configs use the models' default adaptive thinking at `effort: low`, so the model reasons but the comparison varies only model and guardrail. No sampling parameters are sent (these models reject non-default values). Server-side refusal fallbacks are **off**: a fallback would silently swap in a different model mid-eval. Refusals are recorded (there were 0).
- **No structured-output mode.** JSON is requested in the prompt instead of through the API's structured-output feature. This keeps the approach model-agnostic and keeps the malformed-output rule meaningful.
- **Guardrail rules** (`guardrail/checks.py`), in order; the first one that fires is the recorded reason:
  1. `malformed_output`: unparseable JSON, a missing or invalid `verdict`/`evidence`, or "supported" with no evidence spans → hallucinated.
  2. `fabricated_span`: a span not found in the passage after NFKC normalization, lower-casing, ignoring whitespace, and unifying quote and dash characters → hallucinated. Spans under 3 characters are ignored.
  3. `uncovered_key_fact`: a key fact in the answer is not in the cited spans → hallucinated. Key facts are dates (`March 2, 1991` and similar), numbers (`1,200`, `19th`, `1990s`), and runs of capitalized words (allowing `of`/`the`/`for`/`van`…), with leading stopwords like "The" or "Yes" removed. Coverage uses exact number matching, date-part matching, and entity matching by phrase or per-token fuzzy match (difflib ratio ≥ 0.85, tokens of 4+ characters).
  The guardrail can only make a verdict stricter, never flip "hallucinated" to "supported".
- **Plain configs** that return unparseable output would be scored "supported", i.e. no flag raised. This happened 0 times.
- **Statistics:** the positive class is *hallucinated*. Bootstrap CIs resample **questions**, not items (1,000 resamples, seed 20261015, percentile method), because each question's two items share a passage. McNemar uses the exact binomial test on discordant pairs; the continuity-corrected χ² is also reported.
- **Cost:** list prices in `guardrail/config.py` (USD/MTok: Haiku 5.5 $0.10 in / $0.50 out; Sonnet 5.5 $2 in / $10 out, as published 2026-10-06). Output tokens include thinking tokens. Latency is the wall-clock time of the API call, plus the guardrail check for config 3.
- **Did I change anything after seeing results?** The rules, prompts, and thresholds were written and unit-tested before the full run, and none of them changed afterwards. I did see the 6 dry-run items, which are part of the 120. After the full run I changed only reporting: I added the `uncovered_fact_diagnosis` and `model_verdict_differs_from_plain_haiku` counts, fixed wording in the report, and adjusted dashboard layout.

## Limitations

- **Small sample.** 120 items from 60 questions; the CIs are about ±5 points wide, and every between-config difference is within noise.
- **One subset.** HaluEval QA only; dialogue, summarization, and general-chat hallucinations were not tested. The labels themselves are noisy (see above).
- **Heuristic key-fact extraction.** Regex rules miss spelled-out numbers ("thirty"), lowercase facts ("early 1930s" passes because "1930s" is cited), and paraphrased claims, and can catch sentence-initial common words.
- **Not a reproduction of CTGT's method.** This is an independent experiment testing a similar idea, not CTGT's system or their published HaluEval result.

## What I'd try next

- **An NLI-based entailment check instead of regex.** Score whether the cited spans entail the answer with a small open NLI model, still deterministic at inference. That targets the paraphrase-level hallucinations the regex misses.
- **More HaluEval subsets** (dialogue, summarization, general) and a larger n, so a few-point difference becomes detectable. With the 5% discordance rate seen here, McNemar needs about 430 items (about 220 questions, before any clustering adjustment) for 80% power to detect a 3-point paired difference at α = 0.05.
- **Open-source models** as the reasoner (Llama, Qwen, Mistral), to test whether the layer is really model-agnostic and whether it helps weaker models more.

## Layout

```
run.py                     CLI (typer): run | report | dashboard | fonts
guardrail/checks.py        the deterministic rules (pure Python, unit-tested)
guardrail/llm.py           async Anthropic calls: disk cache, concurrency 5, retry/backoff, budget estimate
guardrail/pipeline.py      scoring, summary metrics, example selection, CSV/JSON output
guardrail/metrics.py       accuracy/P/R/F1/false-refusal, cluster bootstrap, McNemar
guardrail/report.py        out/report.md
guardrail/dashboard*.{py,html}  out/dashboard.html (inline CSS/JS/SVG, fonts embedded)
tests/test_checks.py       span check, key-fact extraction/coverage, JSON-failure handling, metrics
results/  out/             committed deliverables
```

The dashboard uses Switzer (Fontshare, ITF Free Font License) and Geist Mono (OFL), embedded as base64 so the page makes no network requests. The Switzer files themselves are not committed; `python run.py fonts` downloads them.
