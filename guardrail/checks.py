"""The deterministic guardrail: rules enforced in code that the model cannot talk its way past.

Everything in this module is pure Python with no model calls, so it gives the same
answer for the same input every time and runs in microseconds.

Rules (applied in this order; the first one that fires is the recorded reason):

1. ``malformed_output``  - the model's reply does not parse as JSON with a valid
   ``verdict`` and an ``evidence`` list, or it says "supported" without citing a
   single evidence span. Fail closed: hallucinated.
2. ``fabricated_span``   - an evidence span does not occur verbatim in the
   knowledge passage (whitespace-, case-, quote- and dash-insensitive).
   Fail closed: hallucinated.
3. ``uncovered_key_fact`` - a key fact in the answer (number, year, date, or
   capitalized named entity) does not appear in the cited spans. Hallucinated.

The guardrail can only make a verdict stricter: if the model already says
"hallucinated", the rules are still evaluated and recorded but cannot flip it back.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, ValidationError

Verdict = Literal["supported", "hallucinated"]

RULE_MALFORMED = "malformed_output"
RULE_FABRICATED = "fabricated_span"
RULE_UNCOVERED = "uncovered_key_fact"
RULES = (RULE_FABRICATED, RULE_UNCOVERED, RULE_MALFORMED)

MIN_SPAN_CHARS = 3        # spans shorter than this (ignoring whitespace) are not evidence
FUZZY_TOKEN_RATIO = 0.85  # SequenceMatcher ratio for entity-token fuzzy matching
FUZZY_MIN_LEN = 4         # tokens shorter than this must match exactly

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

_CHAR_MAP = {
    **{c: "'" for c in "‘’‚‛′`´"},
    **{c: '"' for c in "“”„‟″"},
    **{c: "-" for c in "‐‑‒–—―−"},
}


def _norm_char(c: str) -> str:
    c = unicodedata.normalize("NFKC", c)
    return "".join(_CHAR_MAP.get(ch, ch) for ch in c).lower()


def _squash_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` and drop all whitespace, keeping a map back to original offsets."""
    chars: list[str] = []
    index: list[int] = []
    for i, c in enumerate(text):
        if c.isspace():
            continue
        for ch in _norm_char(c):
            if ch.isspace():
                continue
            chars.append(ch)
            index.append(i)
    return "".join(chars), index


def squash(text: str) -> str:
    return _squash_with_map(text)[0]


def _strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def word_tokens(text: str) -> list[str]:
    """Lowercase, accent-free word tokens with possessive 's removed."""
    s = _strip_accents("".join(_norm_char(c) for c in text))
    s = re.sub(r"'s\b", "", s)
    return re.findall(r"[a-z0-9]+", s)


# ---------------------------------------------------------------------------
# Rule 2: verbatim evidence spans
# ---------------------------------------------------------------------------


@dataclass
class SpanCheck:
    text: str
    verbatim: bool
    start: int | None = None  # character offsets into the knowledge passage
    end: int | None = None


def check_span(span: str, knowledge: str) -> SpanCheck:
    """Is ``span`` copied verbatim from ``knowledge`` (ignoring whitespace/case/quote/dash style)?"""
    s = squash(span)
    k, index = _squash_with_map(knowledge)
    pos = k.find(s) if s else -1
    if pos < 0:
        return SpanCheck(span, False)
    return SpanCheck(span, True, index[pos], index[pos + len(s) - 1] + 1)


# ---------------------------------------------------------------------------
# Rule 3: key-fact extraction and coverage
# ---------------------------------------------------------------------------

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_RE = re.compile(
    rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"  # March 2, 1991
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\.?,?\s+\d{{4}}\b"  # 2 March 1991
    rf"|\b(?:{_MONTHS})\.?,?\s+\d{{4}}\b"  # March 1991
    rf"|\b(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"  # March 2
)
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:st|nd|rd|th|s)?(?!\w)")

_CONNECTORS = {"of", "the", "de", "da", "del", "della", "di", "du", "van", "von", "der", "den",
               "la", "le", "for", "y"}
_CAP_TOKEN = r"[A-Z][\w'’&.\-]*"
ENTITY_RE = re.compile(
    rf"{_CAP_TOKEN}(?:\s+(?:(?:{'|'.join(sorted(_CONNECTORS))})\s+)*{_CAP_TOKEN})*"
)
# Capitalized words that are almost never the factual payload of an answer.
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "he", "she", "they", "we",
    "i", "his", "her", "their", "our", "you", "your", "yes", "no", "both", "neither", "either",
    "there", "here", "however", "although", "though", "while", "after", "before", "during",
    "according", "based", "as", "by", "with", "from", "to", "of", "for", "and", "but", "or",
    "not", "so", "also", "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "all", "some", "many", "most", "one", "in", "on", "at", "is", "was", "are", "were", "be",
    "did", "does", "do", "has", "had", "have", "only", "because", "since", "if", "then", "than",
    "unlike", "like", "each", "every", "none", "nothing", "yet", "still",
}


@dataclass
class KeyFact:
    text: str
    kind: Literal["number", "date", "entity"]
    start: int  # character offsets into the answer
    end: int


def _number_value(tok: str) -> str:
    tok = re.sub(r"(st|nd|rd|th|s)$", "", tok.lower())
    return tok.replace(",", "")


def extract_key_facts(answer: str) -> list[KeyFact]:
    """Pull numbers, years, dates and capitalized named entities out of an answer with regex rules."""
    facts: list[KeyFact] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for m in DATE_RE.finditer(answer):
        facts.append(KeyFact(m.group(), "date", m.start(), m.end()))
        taken.append(m.span())
    for m in NUMBER_RE.finditer(answer):
        if free(*m.span()):
            facts.append(KeyFact(m.group(), "number", m.start(), m.end()))
            taken.append(m.span())
    for m in ENTITY_RE.finditer(answer):
        if not free(*m.span()):
            continue
        # Trim leading stopwords / trailing connectors and punctuation, tracking offsets.
        words = [(w.start() + m.start(), w.end() + m.start(), w.group())
                 for w in re.finditer(r"\S+", m.group())]
        cleaned = []
        for a, b, w in words:
            core = w.rstrip(".,;:!?\"')’")
            if re.fullmatch(r"(?:[A-Z]\.)+", w):  # keep abbreviations like U.S.
                core = w
            cleaned.append((a, a + len(core), core))
        while cleaned and (cleaned[0][2].lower() in _STOPWORDS or cleaned[0][2].lower() in _CONNECTORS):
            cleaned.pop(0)
        while cleaned and cleaned[-1][2].lower() in _CONNECTORS:
            cleaned.pop()
        if not cleaned:
            continue
        a, b = cleaned[0][0], cleaned[-1][1]
        text = answer[a:b]
        if len(text) < 2 or not word_tokens(text):
            continue
        facts.append(KeyFact(text, "entity", a, b))
        taken.append((a, b))
    facts.sort(key=lambda f: f.start)
    return facts


def _numbers_in(text: str) -> set[str]:
    return {_number_value(m.group()) for m in NUMBER_RE.finditer(text)}


def _token_match(tok: str, evidence_tokens: set[str]) -> bool:
    if tok in evidence_tokens:
        return True
    if len(tok) < FUZZY_MIN_LEN or tok.isdigit():
        return False
    return any(
        len(e) >= FUZZY_MIN_LEN and SequenceMatcher(None, tok, e).ratio() >= FUZZY_TOKEN_RATIO
        for e in evidence_tokens
    )


def fact_covered(fact: KeyFact, evidence: str) -> bool:
    """Does the cited evidence contain this key fact (light normalization + simple fuzzy match)?"""
    if fact.kind == "number":
        return _number_value(fact.text) in _numbers_in(evidence)

    if fact.kind == "date":
        if squash(fact.text) in squash(evidence):
            return True
        month = re.search(_MONTHS, fact.text)
        nums = _numbers_in(fact.text)
        ev_words = word_tokens(evidence)
        month_ok = month is None or any(w.startswith(month.group().lower()[:3]) for w in ev_words)
        return month_ok and nums <= _numbers_in(evidence)

    toks = [t for t in word_tokens(fact.text) if t not in _CONNECTORS] or word_tokens(fact.text)
    ev_tokens = word_tokens(evidence)
    if f" {' '.join(word_tokens(fact.text))} " in f" {' '.join(ev_tokens)} ":
        return True
    ev_set = set(ev_tokens)
    return all(_token_match(t, ev_set) for t in toks)


# ---------------------------------------------------------------------------
# Rule 1: parse the model's JSON; never silently accept
# ---------------------------------------------------------------------------


class GuardrailReply(BaseModel):
    verdict: Verdict
    evidence: list[str]


def _first_json_object(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_guardrail_reply(text: str) -> tuple[GuardrailReply | None, str | None]:
    obj = _first_json_object(text or "")
    if obj is None:
        return None, "reply is not valid JSON"
    if isinstance(obj.get("verdict"), str):
        obj["verdict"] = obj["verdict"].strip().lower()
    try:
        return GuardrailReply.model_validate(obj), None
    except ValidationError as e:
        missing = [".".join(map(str, err["loc"])) for err in e.errors()]
        return None, f"missing or invalid field(s): {', '.join(missing)}"


def parse_plain_verdict(text: str) -> tuple[Verdict, bool]:
    """Parse a plain-config reply. Returns (verdict, parsed_ok).

    Plain configs have no guardrail, so an unparseable reply is recorded as
    ``parsed_ok=False`` and scored as "supported" (the model raised no flag).
    """
    obj = _first_json_object(text or "")
    if obj and isinstance(obj.get("verdict"), str) and obj["verdict"].strip().lower() in ("supported", "hallucinated"):
        return obj["verdict"].strip().lower(), True  # type: ignore[return-value]
    words = set(re.findall(r"\b(supported|hallucinated)\b", (text or "").lower()))
    if len(words) == 1:
        return words.pop(), True  # type: ignore[return-value]
    return "supported", False


# ---------------------------------------------------------------------------
# The guardrail
# ---------------------------------------------------------------------------


@dataclass
class FactCheck:
    text: str
    kind: str
    start: int
    end: int
    covered: bool


@dataclass
class GuardrailDecision:
    final_verdict: Verdict
    model_verdict: Verdict | None
    overridden: bool               # code changed (or supplied) the verdict
    rule: str | None               # first rule that fired, if any
    rules_fired: list[str] = field(default_factory=list)
    detail: str = ""
    spans: list[SpanCheck] = field(default_factory=list)
    facts: list[FactCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def apply_guardrail(reply_text: str, knowledge: str, answer: str) -> GuardrailDecision:
    reply, err = parse_guardrail_reply(reply_text)
    facts = extract_key_facts(answer)
    if reply is None:
        return GuardrailDecision(
            "hallucinated", None, True, RULE_MALFORMED, [RULE_MALFORMED], err or "malformed",
            facts=[FactCheck(f.text, f.kind, f.start, f.end, False) for f in facts],
        )

    spans = [check_span(s, knowledge) for s in reply.evidence if len(squash(s)) >= MIN_SPAN_CHARS]
    evidence = " ".join(s.text for s in spans if s.verbatim)
    fact_checks = [FactCheck(f.text, f.kind, f.start, f.end, fact_covered(f, evidence)) for f in facts]

    fired: list[str] = []
    details: list[str] = []
    if reply.verdict == "supported" and not spans:
        fired.append(RULE_MALFORMED)
        details.append("verdict is 'supported' but no evidence span was cited")
    fabricated = [s.text for s in spans if not s.verbatim]
    if fabricated:
        fired.append(RULE_FABRICATED)
        details.append("span(s) not found verbatim in knowledge: " + "; ".join(f'"{t}"' for t in fabricated))
    uncovered = [f.text for f in fact_checks if not f.covered]
    if uncovered and (spans or reply.verdict == "supported"):
        fired.append(RULE_UNCOVERED)
        details.append("key fact(s) not in cited evidence: " + ", ".join(f'"{t}"' for t in uncovered))

    final: Verdict = "hallucinated" if (fired or reply.verdict == "hallucinated") else "supported"
    return GuardrailDecision(
        final_verdict=final,
        model_verdict=reply.verdict,
        overridden=final != reply.verdict,
        rule=fired[0] if fired else None,
        rules_fired=fired,
        detail=" | ".join(details),
        spans=spans,
        facts=fact_checks,
    )
