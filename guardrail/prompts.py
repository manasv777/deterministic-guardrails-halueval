"""Prompts. Both plain configs share one prompt; the guardrail config adds the evidence contract."""

PROMPT_VERSION = "v1"

_TASK = """You are a careful fact-checking judge. You will be given a KNOWLEDGE passage, a QUESTION, and a candidate ANSWER.

Decide whether the ANSWER is:
- "supported": it correctly answers the question and every fact it states is backed by the KNOWLEDGE passage, or
- "hallucinated": it states something that is not backed by the KNOWLEDGE passage, contradicts it, or does not correctly answer the question.

Use only the KNOWLEDGE passage, not outside knowledge."""

PLAIN_SYSTEM = _TASK + """

Reply with only a JSON object, no prose and no code fences:
{"verdict": "supported" | "hallucinated"}"""

GUARDRAIL_SYSTEM = _TASK + """

You must also cite evidence. Copy the exact sentences or phrases from the KNOWLEDGE passage that support the ANSWER, character for character. Do not paraphrase, merge, abbreviate or use ellipses; give each contiguous excerpt as a separate list item. Your evidence must contain every name, number and date that the ANSWER states. If the answer is hallucinated, cite the excerpts you relied on (or an empty list if none apply).

Your reply will be checked by code: any evidence span not found verbatim in the passage, or any name/number/date in the ANSWER missing from your evidence, will cause your verdict to be rejected.

Reply with only a JSON object, no prose and no code fences:
{"verdict": "supported" | "hallucinated", "evidence": ["<verbatim excerpt>", ...]}"""


def user_message(knowledge: str, question: str, answer: str) -> str:
    return f"KNOWLEDGE:\n{knowledge}\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}"


def system_prompt(mode: str) -> str:
    return GUARDRAIL_SYSTEM if mode == "guardrail" else PLAIN_SYSTEM
