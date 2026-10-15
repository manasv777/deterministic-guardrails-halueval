import json

import numpy as np
import pytest

from guardrail.checks import (
    RULE_FABRICATED,
    RULE_MALFORMED,
    RULE_UNCOVERED,
    KeyFact,
    apply_guardrail,
    check_span,
    extract_key_facts,
    fact_covered,
    parse_guardrail_reply,
    parse_plain_verdict,
)
from guardrail.metrics import mcnemar, scores

KNOWLEDGE = (
    "Arthur's Magazine (1844–1846) was an American literary periodical published in "
    "Philadelphia in the 19th century.First for Women is a woman's magazine published by "
    "Bauer Media Group in the USA."
)


def reply(verdict, evidence):
    return json.dumps({"verdict": verdict, "evidence": evidence})


# --- verbatim span check -----------------------------------------------------

def test_exact_span_is_verbatim_and_located():
    s = check_span("published in Philadelphia", KNOWLEDGE)
    assert s.verbatim
    assert KNOWLEDGE[s.start:s.end] == "published in Philadelphia"


def test_span_whitespace_case_quote_and_dash_insensitive():
    # Extra spaces, different case, straight dash instead of en dash, space after the run-on period.
    assert check_span("ARTHUR'S   Magazine (1844-1846)", KNOWLEDGE).verbatim
    assert check_span("19th century. First for Women", KNOWLEDGE).verbatim
    assert check_span("Arthur’s Magazine", KNOWLEDGE).verbatim


def test_paraphrased_or_invented_span_is_not_verbatim():
    assert not check_span("published in Boston", KNOWLEDGE).verbatim
    assert not check_span("Arthur's Magazine was founded in 1844", KNOWLEDGE).verbatim
    assert not check_span("Arthur's Magazine ... Philadelphia", KNOWLEDGE).verbatim


def test_fabricated_span_fails_closed():
    d = apply_guardrail(reply("supported", ["Arthur's Magazine was first published in 1844"]),
                        KNOWLEDGE, "Arthur's Magazine")
    assert d.final_verdict == "hallucinated"
    assert d.overridden and d.rule == RULE_FABRICATED


# --- key-fact extraction -----------------------------------------------------

def facts(answer):
    return [(f.text, f.kind) for f in extract_key_facts(answer)]


def test_extracts_entities_numbers_and_dates():
    got = facts("Allie Goertz was born on March 2, 1991 and released 3 albums with Matt Groening.")
    assert ("Allie Goertz", "entity") in got
    assert ("March 2, 1991", "date") in got
    assert ("3", "number") in got
    assert ("Matt Groening", "entity") in got
    # date components are not double-counted as separate numbers
    assert ("1991", "number") not in got


def test_strips_sentence_initial_stopwords_and_punctuation():
    assert facts("The Oberoi family's hotel company is based in Mumbai.") == [
        ("Oberoi", "entity"), ("Mumbai", "entity")]
    assert facts("Yes") == []
    assert facts("yes, both are magazines") == []


def test_multiword_entity_with_connector_and_year():
    got = facts("First for Women was started in 1989.")
    assert got == [("First for Women", "entity"), ("1989", "number")]


def test_number_normalization():
    assert fact_covered(KeyFact("1,200", "number", 0, 5), "about 1200 people")
    assert not fact_covered(KeyFact("1845", "number", 0, 4), "Arthur's Magazine (1844-1846)")
    assert fact_covered(KeyFact("19th", "number", 0, 4), "in the 19th century")


# --- key-fact coverage -------------------------------------------------------

def test_entity_coverage_exact_possessive_and_fuzzy():
    ev = "Arthur's Magazine (1844–1846) was an American literary periodical"
    assert fact_covered(KeyFact("Arthur's Magazine", "entity", 0, 0), ev)
    assert fact_covered(KeyFact("Arthurs Magazine", "entity", 0, 0), ev)       # fuzzy token match
    assert fact_covered(KeyFact("Americans", "entity", 0, 0), ev)              # light fuzzy
    assert not fact_covered(KeyFact("Mumbai", "entity", 0, 0), ev)
    assert not fact_covered(KeyFact("Women", "entity", 0, 0), ev)


def test_date_coverage():
    assert fact_covered(KeyFact("March 2, 1991", "date", 0, 0), "(born March 2, 1991) is an American")
    assert fact_covered(KeyFact("2 March 1991", "date", 0, 0), "(born March 2, 1991)")
    assert not fact_covered(KeyFact("March 3, 1991", "date", 0, 0), "(born March 2, 1991)")


def test_uncovered_key_fact_overrides_supported_verdict():
    know = "The Oberoi Group is a hotel company with its head office in Delhi."
    d = apply_guardrail(reply("supported", ["The Oberoi Group is a hotel company"]), know,
                        "The Oberoi family's hotel company is based in Mumbai.")
    assert d.final_verdict == "hallucinated"
    assert d.rule == RULE_UNCOVERED
    assert [f.text for f in d.facts if not f.covered] == ["Mumbai"]


def test_fully_covered_supported_answer_passes():
    know = "The Oberoi Group is a hotel company with its head office in Delhi."
    d = apply_guardrail(reply("supported", ["its head office in Delhi"]), know, "Delhi")
    assert d.final_verdict == "supported"
    assert not d.overridden and d.rule is None


def test_guardrail_never_flips_hallucinated_to_supported():
    d = apply_guardrail(reply("hallucinated", []), KNOWLEDGE, "Arthur's Magazine")
    assert d.final_verdict == "hallucinated" and not d.overridden


# --- JSON failure handling ---------------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "The answer is supported.",
    '{"verdict": "supported"',                       # truncated JSON
    '{"verdict": "supported"}',                      # missing evidence
    '{"evidence": ["Delhi"]}',                       # missing verdict
    '{"verdict": "probably", "evidence": []}',       # invalid verdict
    '{"verdict": "supported", "evidence": "Delhi"}', # wrong type
])
def test_malformed_output_fails_closed(text):
    d = apply_guardrail(text, KNOWLEDGE, "Arthur's Magazine")
    assert d.final_verdict == "hallucinated"
    assert d.overridden and d.rule == RULE_MALFORMED
    assert d.model_verdict is None


def test_supported_without_evidence_is_malformed():
    d = apply_guardrail(reply("supported", []), KNOWLEDGE, "yes")
    assert d.final_verdict == "hallucinated" and d.rule == RULE_MALFORMED


def test_json_in_code_fence_and_surrounding_text_is_parsed():
    text = 'Here you go:\n```json\n{"verdict": "Supported", "evidence": ["published in Philadelphia"]}\n```'
    r, err = parse_guardrail_reply(text)
    assert err is None and r.verdict == "supported"


def test_plain_verdict_parsing():
    assert parse_plain_verdict('{"verdict": "hallucinated"}') == ("hallucinated", True)
    assert parse_plain_verdict("Verdict: supported") == ("supported", True)
    assert parse_plain_verdict("I can't tell") == ("supported", False)


# --- metrics sanity ----------------------------------------------------------

def test_scores_and_mcnemar():
    gold = np.array([True, True, False, False])
    pred = np.array([True, False, True, False])
    s = scores(gold, pred)
    assert s["accuracy"] == 0.5 and s["false_refusal_rate"] == 0.5
    assert s["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    m = mcnemar(gold, gold.copy(), pred)
    assert m["a_right_b_wrong"] == 2 and m["a_wrong_b_right"] == 0
    assert m["exact_p"] == pytest.approx(0.5)
