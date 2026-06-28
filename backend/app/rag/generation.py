import json
from html import escape
from typing import Protocol

import httpx
from google import genai
from langsmith import traceable
from pydantic import BaseModel

from app.index.embeddings import MissingGoogleConfiguration
from app.index.vector_index import SearchHit
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


class GenerationClient(Protocol):
    def generate_answer(self, *, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        pass


class EvidenceGenerationClient(Protocol):
    def generate_answer_from_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceSnippet],
    ) -> GeneratedAnswer:
        pass


DEFAULT_TIMEOUT_SECONDS = 60


class GoogleGenerationClient:
    def __init__(self, *, api_key: str | None, model: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if not api_key:
            raise MissingGoogleConfiguration("GEMINI_API_KEY is not configured.")

        self._model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_seconds * 1000},
        )

    def generate_raw(self, prompt: str) -> str:
        """Send a raw prompt and return the text response."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text or ""

    @traceable(name="generate_answer")
    def generate_answer(self, *, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_build_prompt(question=question, hits=hits),
            )
            payload = _parse_json_object(response.text or "{}")
        except Exception as error:
            raise MissingGoogleConfiguration("Google chat configuration is unusable.") from error

        return GeneratedAnswer(
            answer=str(payload.get("answer", "")),
            reasoning=str(payload.get("reasoning", "")),
            grounded=bool(payload.get("grounded", False)),
            supporting_chunk_ids=[str(item) for item in payload.get("supporting_chunk_ids", [])],
        )

    @traceable(name="generate_answer_from_evidence")
    def generate_answer_from_evidence(
        self,
        *,
        question: str,
        evidence: list[EvidenceSnippet],
    ) -> GeneratedAnswer:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_build_evidence_prompt(question=question, evidence=evidence),
            )
            payload = _parse_json_object(response.text or "{}")
        except Exception as error:
            raise MissingGoogleConfiguration("Google chat configuration is unusable.") from error

        return _answer_from_evidence_payload(payload, evidence)


def _build_prompt(*, question: str, hits: list[SearchHit]) -> str:
    chunk_blocks = "\n".join(
        f'<chunk id="{escape(hit.chunk.chunk_id)}">\n'
        f"  <title>{escape(hit.chunk.title)}</title>\n"
        f"  <url>{escape(hit.chunk.url)}</url>\n"
        f"  <text>{escape(hit.chunk.text)}</text>\n"
        f"</chunk>"
        for hit in hits
    )
    return (
        "You are a grounded answering system. "
        "Answer the question using ONLY the provided chunks.\n\n"
        "<instructions>\n"
        "1. Analyze each chunk for relevance to the question.\n"
        "2. Reason step-by-step about which chunks support the answer.\n"
        "3. If the chunks do not support an answer, say the indexed site "
        "content does not contain enough information.\n"
        "</instructions>\n\n"
        f"<question>{escape(question)}</question>\n\n"
        f"<chunks>\n{chunk_blocks}\n</chunks>\n\n"
        "Return ONLY a JSON object with these keys:\n"
        '- "reasoning": Your step-by-step analysis (string)\n'
        '- "answer": The final answer based only on the chunks (string)\n'
        '- "grounded": true if the answer is fully supported (boolean)\n'
        '- "supporting_chunk_ids": Array of chunk IDs that support the answer'
    )


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
