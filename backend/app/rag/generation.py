import json
from html import escape
from typing import Protocol

from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable
from pydantic import BaseModel, ValidationError

from app.index.embeddings import MissingGoogleConfiguration
from app.retrieval.models import EvidenceSnippet

NOT_FOUND_ANSWER = "The indexed site content does not contain enough information to answer that."
VALID_GROUNDEDNESS = {"grounded", "partially_grounded", "not_grounded"}


class GeneratedAnswer(BaseModel):
    answer: str
    reasoning: str = ""
    grounded: bool = False
    supporting_chunk_ids: list[str] = []
    groundedness: str = "not_grounded"
    claims: list[dict[str, object]] = []
    supporting_evidence_ids: list[str] = []


class EvidenceGenerationClient(Protocol):
    def generate_answer_from_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceSnippet],
    ) -> GeneratedAnswer:
        pass


class RawGenerationClient(Protocol):
    def generate_raw(self, prompt: str) -> str:
        pass


DEFAULT_TIMEOUT_SECONDS = 60


class _ClaimSchema(BaseModel):
    text: str = ""
    supporting_evidence_ids: list[str] = []
    supported: bool = False


class GeneratedAnswerSchema(BaseModel):
    reasoning: str = ""
    answer: str = ""
    groundedness: str = "not_grounded"
    claims: list[_ClaimSchema] = []
    supporting_evidence_ids: list[str] = []


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content or "")


def _structured_payload_is_usable(payload: dict[str, object]) -> bool:
    if payload.get("answer") or payload.get("claims") or payload.get("supporting_evidence_ids"):
        return True
    if str(payload.get("reasoning", "")).strip():
        return True
    groundedness = str(payload.get("groundedness", "not_grounded"))
    return groundedness in VALID_GROUNDEDNESS and groundedness != "not_grounded"


def _dump_structured_result(result: object) -> dict[str, object]:
    if isinstance(result, BaseModel):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {}


class LangChainGenerationClient:
    def __init__(self, *, api_key: str | None, model: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if not api_key:
            raise MissingGoogleConfiguration("GEMINI_API_KEY is not configured.")

        self._model = model
        self._chat = ChatGoogleGenerativeAI(
            model=model, api_key=api_key, timeout=timeout_seconds, vertexai=False
        )
        self._structured = self._chat.with_structured_output(
            GeneratedAnswerSchema, method="json_schema"
        )

    def generate_raw(self, prompt: str) -> str:
        return _message_text(self._chat.invoke(prompt))

    @traceable(name="generate_answer_from_evidence")
    def generate_answer_from_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceSnippet],
    ) -> GeneratedAnswer:
        prompt = _build_evidence_prompt(question=question, evidence=evidence)
        try:
            payload = self._structured_payload(prompt)
        except Exception as error:
            raise MissingGoogleConfiguration("Google chat configuration is unusable.") from error
        return _answer_from_evidence_payload(payload, evidence)

    def _structured_payload(self, prompt: str) -> dict[str, object]:
        try:
            result = self._structured.invoke(prompt)
            payload = _dump_structured_result(result)
            if _structured_payload_is_usable(payload):
                return payload
        except (ValidationError, TypeError, ValueError):
            pass
        return _parse_json_object(self.generate_raw(prompt))


def _build_evidence_prompt(*, question: str, evidence: list[EvidenceSnippet]) -> str:
    evidence_blocks = "\n".join(
        f'<evidence id="{escape(item.evidence_id)}" chunk="{escape(item.chunk_id)}">\n'
        f"  <title>{escape(item.title)}</title>\n"
        f"  <url>{escape(item.url)}</url>\n"
        f"  <heading_path>{escape(' > '.join(item.heading_path))}</heading_path>\n"
        f"  <snippet>{escape(item.snippet)}</snippet>\n"
        f"  <nearby_context>{escape(item.nearby_context or item.snippet)}</nearby_context>\n"
        f"</evidence>"
        for item in evidence
    )
    return (
        "You are a grounded answering system. "
        "Answer the question using ONLY the provided evidence snippets.\n\n"
        "<instructions>\n"
        "1. Analyze each evidence snippet for relevance to the question.\n"
        "2. Reason step-by-step about which evidence supports the answer.\n"
        "3. Synthesize a clear, accurate answer citing specific evidence IDs.\n"
        "4. Every material claim must include supporting_evidence_ids.\n"
        "5. If evidence is insufficient, say the indexed site content does "
        "not contain enough information.\n"
        "</instructions>\n\n"
        f"<question>{escape(question)}</question>\n\n"
        f"<evidence_collection>\n{evidence_blocks}\n</evidence_collection>\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '- "reasoning": Your step-by-step analysis of the evidence (string)\n'
        '- "answer": The final answer based only on evidence (string)\n'
        '- "groundedness": One of "grounded", "partially_grounded", "not_grounded"\n'
        '- "claims": Array of {"text", "supporting_evidence_ids", "supported"}\n'
        '- "supporting_evidence_ids": Array of evidence IDs that support the answer'
    )


def _parse_json_object(text: str) -> dict[str, object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        return {}
    return payload


def _answer_from_evidence_payload(
    payload: dict[str, object],
    evidence: list[EvidenceSnippet],
) -> GeneratedAnswer:
    valid_evidence_ids = {item.evidence_id for item in evidence}
    answer = str(payload.get("answer", "")).strip()
    supporting_evidence_ids = [
        str(item)
        for item in payload.get("supporting_evidence_ids", [])
        if str(item) in valid_evidence_ids
    ]
    claims = _normalize_claims(payload.get("claims", []), valid_evidence_ids)
    if not claims and answer and supporting_evidence_ids:
        claims = [
            {
                "text": answer,
                "supporting_evidence_ids": supporting_evidence_ids,
                "supported": True,
            }
        ]

    claim_supported_ids = [
        evidence_id
        for claim in claims
        if claim.get("supported") is True
        for evidence_id in claim["supporting_evidence_ids"]
    ]
    supporting_evidence_ids = sorted(set(supporting_evidence_ids + claim_supported_ids))
    groundedness = _groundedness_from_claims(
        requested=str(payload.get("groundedness", "not_grounded")),
        claims=claims,
        supporting_evidence_ids=supporting_evidence_ids,
    )
    reasoning = str(payload.get("reasoning", "")).strip()
    return GeneratedAnswer(
        answer=answer if groundedness != "not_grounded" else NOT_FOUND_ANSWER,
        reasoning=reasoning,
        grounded=groundedness == "grounded",
        groundedness=groundedness,
        claims=claims,
        supporting_evidence_ids=supporting_evidence_ids,
    )


def _normalize_claims(
    raw_claims: object,
    valid_evidence_ids: set[str],
) -> list[dict[str, object]]:
    if not isinstance(raw_claims, list):
        return []
    claims: list[dict[str, object]] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue
        text = str(raw_claim.get("text", "")).strip()
        if not text:
            continue
        supporting_evidence_ids = [
            str(item)
            for item in raw_claim.get("supporting_evidence_ids", [])
            if str(item) in valid_evidence_ids
        ]
        claims.append(
            {
                "text": text,
                "supporting_evidence_ids": supporting_evidence_ids,
                "supported": bool(raw_claim.get("supported")) and bool(supporting_evidence_ids),
            }
        )
    return claims


def _groundedness_from_claims(
    *,
    requested: str,
    claims: list[dict[str, object]],
    supporting_evidence_ids: list[str],
) -> str:
    if not supporting_evidence_ids:
        return "not_grounded"
    if not claims:
        return requested if requested in VALID_GROUNDEDNESS else "partially_grounded"
    supported_count = sum(1 for claim in claims if claim.get("supported") is True)
    if supported_count == len(claims):
        return "grounded"
    if supported_count > 0:
        return "partially_grounded"
    return "not_grounded"
