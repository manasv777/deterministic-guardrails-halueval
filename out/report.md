# Deterministic Guardrails vs. Plain LLMs on HaluEval

*120 items (60 HaluEval QA questions × right/hallucinated answer) · seed 20261015 · 2026-10-15 · small = `claude-haiku-5-5`, large = `claude-sonnet-5-5` · effort `low`*

## Summary

Plain Haiku reached 89.2% accuracy and plain Sonnet 90.8%. Haiku with the deterministic guardrail reached 90.8% (paired bootstrap 95% CI for the gain over plain Haiku: [-2.5, 5.8] points; vs. Sonnet: [-3.3, 3.3] points). The guardrail overrode Haiku's verdict 4 times; 3 were correct catches and 1 was a false refusal. Its false refusal rate was 8.3% vs. 6.7% for plain Haiku. Per 100 items it cost $0.0163, vs. $0.1442 for plain Sonnet (8.8× more).

## Results

| Config | Accuracy [95% CI] | F1 halluc. [95% CI] | Precision | Recall | False refusal rate | $ / 100 items | Avg latency |
|---|---|---|---|---|---|---|---|
| Haiku · plain (`claude-haiku-5-5`) | 89.2% [83.3, 94.2] | 88.7% [82.6, 94.0] | 92.7% | 85.0% | 6.7% | $0.0086 | 1.00s |
| Sonnet · plain (`claude-sonnet-5-5`) | 90.8% [85.8, 95.8] | 90.4% [85.0, 95.7] | 94.5% | 86.7% | 5.0% | $0.1442 | 1.27s |
| **Haiku + guardrail** (`claude-haiku-5-5`) | 90.8% [85.8, 95.8] | 90.8% [85.2, 95.8] | 91.5% | 90.0% | 8.3% | $0.0163 | 1.32s |

Positive class = *hallucinated*. False refusal rate = share of correct answers flagged as hallucinated. CIs: 1000 cluster-bootstrap resamples over questions (seed 20261015), percentile method. Resampling questions rather than items keeps each question's right/hallucinated pair together.

### Confusion matrices

| Config | TP (caught) | FN (missed) | FP (false refusal) | TN |
|---|---|---|---|---|
| Haiku · plain | 51 | 9 | 4 | 56 |
| Sonnet · plain | 52 | 8 | 3 | 57 |
| Haiku + guardrail | 54 | 6 | 5 | 55 |

## McNemar's test (paired, same items)

- **Guardrail vs. plain Haiku: p = 0.688, not statistically significant at α = 0.05 (4 items only Guardrail got right, 2 only plain Haiku got right).** Exact binomial on 6 discordant items; chi-square with continuity correction χ² = 0.17, p = 0.683.
- Guardrail vs. plain Sonnet: p = 1.000, not statistically significant at α = 0.05 (2 items only Guardrail got right, 2 only plain Sonnet got right).
- Plain Sonnet vs. plain Haiku: p = 0.500, not statistically significant at α = 0.05 (2 items only Plain Sonnet got right, 0 only plain Haiku got right).

Caveat: McNemar treats the items as independent pairs, but each question contributes two items that share a knowledge passage. The cluster-bootstrap CI for the accuracy difference above accounts for that pairing.

## Cost and latency

| Config | Input tokens | Output tokens | Total $ | $ / 100 items | Avg latency | Median latency |
|---|---|---|---|---|---|---|
| Haiku · plain | 53,972 | 9,883 | $0.0103 | $0.0086 | 1.00s | 0.77s |
| Sonnet · plain | 53,972 | 6,515 | $0.1731 | $0.1442 | 1.27s | 1.04s |
| Haiku + guardrail | 81,572 | 22,824 | $0.0196 | $0.0163 | 1.32s | 1.01s |

Total spend for this run: **$0.2030**. Prices (USD / 1M tokens): `claude-sonnet-5-5` $2.00 in / $10.00 out, `claude-haiku-5-5` $0.10 in / $0.50 out. Output tokens include adaptive-thinking tokens. Guardrail latency includes the deterministic checks, which take well under a millisecond.

## Guardrail override breakdown

| Rule | Overrides | Correct catches | False refusals | Fired at all (incl. items Haiku already flagged) |
|---|---|---|---|---|
| `fabricated_span` | 0 | 0 | 0 | 0 |
| `uncovered_key_fact` | 4 | 3 | 1 | 44 |
| `malformed_output` | 0 | 0 | 0 | 0 |
| **Total** | **4** | 3 | 1 | |

Haiku's own verdict returned *hallucinated* on 55 items in the guardrail config (vs. 55 in the plain config). Its own verdict differed from its plain-config verdict on 2 items, so the evidence prompt alone changes some answers before any rule fires.

**Why the key-fact rule fired.** Of the overrides by `uncovered_key_fact`, 0 involved a fact that appears nowhere in the knowledge passage (the rule found an unsupported fact). 4 involved a fact that *is* in the passage but that Haiku left out of its citation, so the override came from strictness rather than from detecting the false claim. Both kinds count as correct catches when the gold label is hallucinated, but only the first is the mechanism working as intended.

## Examples: the guardrail caught a hallucination plain Haiku missed

**`q1891-H`** (gold: **hallucinated**)

- **Question:** The Doll Squad features what actress that had a birthday on August 26?
- **Answer:** The lead actress of The Doll Squad is Francine York.
- **Knowledge:** The Doll Squad is a 1973 low-budget action film Z movie by Feature-Faire that was later re-released under the title Seduce and Destroy. Directed, edited, co-written and co-produced by Ted V. Mikels, it features Francine York, Michael Ansara, John Carter, Anthony Eisley, Leigh Christian and Tura Satana.Francine York (born Francine Yerich; August 26, 1936 – January 6, 2017) was an American film and television actress.
- **Verdicts:** Haiku · plain: **supported** · Sonnet · plain: **hallucinated** · Haiku + guardrail: **hallucinated**
- **Cited evidence:**
  - ✅ "it features Francine York, Michael Ansara, John Carter, Anthony Eisley, Leigh Christian and Tura Satana."
  - ✅ "Francine York (born Francine Yerich; August 26, 1936 – January 6, 2017) was an American film and television actress."
- **Key facts:** ❌ Doll Squad, ✅ Francine York

Plain Haiku judged this hallucinated answer as **supported**. In the guardrail config Haiku also said **supported**, and code overrode it via `uncovered_key_fact`: key fact(s) not in cited evidence: "Doll Squad". The key fact "Doll Squad" does appear in the knowledge passage, so the model simply failed to cite it.

**`q2251-H`** (gold: **hallucinated**)

- **Question:** ConnectU was founded by Cameron Winklevoss, Divya Narendra, and this professional rower?
- **Answer:** Tyler Winklevoss founded ConnectU.
- **Knowledge:** ConnectU (originally HarvardConnection) was a social networking website launched on May 21, 2004, that was founded by Harvard students Cameron Winklevoss, Tyler Winklevoss, and Divya Narendra in December 2002.Tyler Howard Winklevoss (born August 21, 1981) is an American rower and entrepreneur.
- **Verdicts:** Haiku · plain: **supported** · Sonnet · plain: **supported** · Haiku + guardrail: **hallucinated**
- **Cited evidence:**
  - ✅ "founded by Harvard students Cameron Winklevoss, Tyler Winklevoss, and Divya Narendra in December 2002"
  - ✅ "Tyler Howard Winklevoss (born August 21, 1981) is an American rower and entrepreneur."
- **Key facts:** ✅ Tyler Winklevoss, ❌ ConnectU

Plain Haiku judged this hallucinated answer as **supported**. In the guardrail config Haiku also said **supported**, and code overrode it via `uncovered_key_fact`: key fact(s) not in cited evidence: "ConnectU". The key fact "ConnectU" does appear in the knowledge passage, so the model simply failed to cite it.

**`q9284-H`** (gold: **hallucinated**)

- **Question:** Who is a current member of the group Guy, Aaron Hall or Matt Thiessen?
- **Answer:** Matt Thiessen is not a member.
- **Knowledge:** Aaron Hall (born August 10, 1964, The Bronx, New York City and growing up in Brooklyn and Roosevelt, Long Island), is an American soul singer and songwriter. He is a current member of the group Guy, which he founded in the late 1980s, along with New jack swing producer Teddy Riley and songwriter Timmy Gatling, who was later replaced by Hall's brother Damion Hall.Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist, pianist, and primary songwriter for the Christian rock band Relient K.
- **Verdicts:** Haiku · plain: **supported** · Sonnet · plain: **supported** · Haiku + guardrail: **hallucinated**
- **Cited evidence:**
  - ✅ "He is a current member of the group Guy, which he founded in the late 1980s"
  - ✅ "Aaron Hall (born August 10, 1964, The Bronx, New York City and growing up in Brooklyn and Roosevelt, Long Island), is an American soul singer and songwriter."
- **Key facts:** ❌ Matt Thiessen

Plain Haiku judged this hallucinated answer as **supported**. In the guardrail config Haiku also said **supported**, and code overrode it via `uncovered_key_fact`: key fact(s) not in cited evidence: "Matt Thiessen". The key fact "Matt Thiessen" does appear in the knowledge passage, so the model simply failed to cite it.

## Examples: the guardrail caused a false refusal

**`q8929-R`** (gold: **supported**)

- **Question:** Which band was formed first, Monaco or Rise Against?
- **Answer:** Monaco
- **Knowledge:** Monaco was a side project of New Order bassist Peter Hook. Together with David Potts, the only remaining member of Revenge, the band was formed in 1995.Rise Against is an American melodic hardcore band from Chicago, Illinois, formed in 1999.
- **Verdicts:** Haiku · plain: **supported** · Sonnet · plain: **supported** · Haiku + guardrail: **hallucinated**
- **Cited evidence:**
  - ✅ "Together with David Potts, the only remaining member of Revenge, the band was formed in 1995."
  - ✅ "Rise Against is an American melodic hardcore band from Chicago, Illinois, formed in 1999."
- **Key facts:** ❌ Monaco

The gold label says this answer is **correct**, but the guardrail config rejected it. In the guardrail config Haiku also said **supported**, and code overrode it via `uncovered_key_fact`: key fact(s) not in cited evidence: "Monaco". The key fact "Monaco" does appear in the knowledge passage, so the model simply failed to cite it.

**`q3340-R`** (gold: **supported**)

- **Question:** What was the pen name of the husband of Grand Duchess Yelizaveta Mavrikiyevna ?
- **Answer:** "K.R."
- **Knowledge:** Prince Georgy Konstantinovich of Russia (6 May 1903 – 7 November 1938), was the youngest son of Grand Duke Konstantin Konstantinovich of Russia and his wife Grand Duchess Yelizaveta Mavrikiyevna. He wrote under the pen name "K.R.", initials of his given name and family name, Konstantin Romanov.
- **Verdicts:** Haiku · plain: **hallucinated** · Sonnet · plain: **hallucinated** · Haiku + guardrail: **hallucinated**
- **Cited evidence:**
  - ✅ "Prince Georgy Konstantinovich of Russia (6 May 1903 – 7 November 1938), was the youngest son of Grand Duke Konstantin Konstantinovich of Russia and his wife Grand Duchess Yelizaveta Mavrikiyevna."
  - ✅ "He wrote under the pen name "K.R.", initials of his given name and family name, Konstantin Romanov."
- **Key facts:** ✅ K.R.

The gold label says this answer is **correct**, but the guardrail config rejected it. Haiku itself returned **hallucinated** once it had to cite verbatim evidence; no code override was needed.

## Limitations

- **Small sample.** 120 items from 60 questions; the CIs above are wide.
- **One subset.** HaluEval QA only (not dialogue, summarization, or general). Its hallucinated answers are often plausible rewrites that introduce a new entity, which suits an entity-coverage rule.
- **Heuristic key-fact extraction.** Regex rules for numbers, dates, and capitalized spans. These miss spelled-out numbers and lowercase facts, and can catch sentence-initial words.
- **Not a reproduction of CTGT's method.** This is an independent experiment testing a similar idea.
