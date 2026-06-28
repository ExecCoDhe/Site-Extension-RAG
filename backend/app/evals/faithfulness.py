"""LLM-as-a-judge evaluator for answer faithfulness against evidence."""

from html import escape

from langsmith import traceable
from pydantic import BaseModel

from app.rag.generation import RawGenerationClient, _parse_json_object


class FaithfulnessResult(BaseModel):
    faithful: bool
    score: float  # 0.0 to 1.0
    reasoning: str
    unsupported_claims: list[str] = []


_FAITHFULNESS_PROMPT = """\
You are an impartial judge evaluating the faithfulness of an AI-generated answer.

Your task: determine whether the answer is fully supported by the evidence.

<instructions>
1. Read the question, answer, and all evidence snippets.
2. Identify every factual claim in the answer.
3. For each claim, check if it is directly supported by at least one evidence snippet.
4. A claim is "unsupported" if no evidence snippet contains the information.
5. Score the answer from 0.0 (completely unfaithful) to 1.0 (fully faithful).
</instructions>

<question>{question}</question>

<answer>{answer}</answer>

<evidence_snippets>
{evidence}
</evidence_snippets>

Return ONLY a JSON object with these keys:
- "reasoning": Your step-by-step analysis (string)
- "faithful": true if ALL claims are supported by evidence (boolean)
- "score": float between 0.0 and 1.0
- "unsupported_claims": Array of claim strings that are NOT supported by evidence"""


@traceable(name="faithfulness_eval")
def evaluate_faithfulness(
    *,
    question: str,
    answer: str,
    evidence_snippets: list[str],
    generation_client: RawGenerationClient,
) -> FaithfulnessResult:
    """Use a secondary LLM call to evaluate answer faithfulness against evidence."""
    evidence_block = "\n".join(
        f"<snippet index=\"{i + 1}\">{escape(snippet)}</snippet>"
        for i, snippet in enumerate(evidence_snippets)
    )
    prompt = _FAITHFULNESS_PROMPT.format(
        question=escape(question),
        answer=escape(answer),
        evidence=evidence_block,
    )

    try:
        raw_text = generation_client.generate_raw(prompt)
        payload = _parse_json_object(raw_text)
        score = float(payload.get("score", 0.0))
    except Exception:
        return FaithfulnessResult(
            faithful=False,
            score=0.0,
            reasoning="Failed to evaluate: generation error.",
        )

    score = max(0.0, min(1.0, score))  # clamp to valid range

    return FaithfulnessResult(
        faithful=bool(payload.get("faithful", False)),
        score=score,
        reasoning=str(payload.get("reasoning", "")),
        unsupported_claims=[
            str(claim) for claim in payload.get("unsupported_claims", [])
        ],
    )
