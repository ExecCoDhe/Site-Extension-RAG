"""LLM-as-a-judge evaluator for answer correctness against a golden reference."""

from html import escape

from langsmith import traceable
from pydantic import BaseModel

from app.rag.generation import RawGenerationClient, _parse_json_object


class CorrectnessResult(BaseModel):
    correct: bool
    score: float  # 0.0 to 1.0
    reasoning: str


_CORRECTNESS_PROMPT = """\
You are an impartial judge evaluating the correctness of an AI-generated answer.

Your task: determine whether the generated answer is semantically equivalent to the reference answer.

<instructions>
1. Read the question, generated answer, and reference answer.
2. Compare semantic meaning, not exact wording.
3. If the reference answer indicates the information is NOT in the corpus (e.g. "Not covered in the indexed corpus"), the response is correct only if it also declines to answer / says it lacks the information.
4. Score the answer from 0.0 (completely incorrect) to 1.0 (fully correct).
</instructions>

<question>{question}</question>

<generated_answer>{generated_answer}</generated_answer>

<reference_answer>{expected_answer}</reference_answer>

Return ONLY a JSON object with these keys:
- "reasoning": Your step-by-step analysis (string)
- "correct": true if the generated answer is semantically correct (boolean)
- "score": float between 0.0 and 1.0"""


@traceable(name="correctness_eval")
def judge_correctness(
    *,
    question: str,
    generated_answer: str,
    expected_answer: str,
    generation_client: RawGenerationClient,
) -> CorrectnessResult:
    """Use a secondary LLM call to evaluate answer correctness against a reference."""
    prompt = _CORRECTNESS_PROMPT.format(
        question=escape(question),
        generated_answer=escape(generated_answer),
        expected_answer=escape(expected_answer),
    )

    try:
        raw_text = generation_client.generate_raw(prompt)
        payload = _parse_json_object(raw_text)
        score = float(payload.get("score", 0.0))
    except Exception:
        return CorrectnessResult(
            correct=False,
            score=0.0,
            reasoning="Failed to evaluate: generation error.",
        )

    score = max(0.0, min(1.0, score))

    return CorrectnessResult(
        correct=bool(payload.get("correct", False)),
        score=score,
        reasoning=str(payload.get("reasoning", "")),
    )
