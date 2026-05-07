import json
from typing import Protocol

import httpx
from google import genai
from pydantic import BaseModel

from app.index.embeddings import MissingGoogleConfiguration
from app.index.vector_index import SearchHit
from app.retrieval.models import EvidenceSnippet

NOT_FOUND_ANSWER = "The indexed site content does not contain enough information to answer that."
VALID_GROUNDEDNESS = {"grounded", "partially_grounded", "not_grounded"}


class GeneratedAnswer(BaseModel):
    answer: str
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

    def generate_answer(self, *, question: str, hits: list[SearchHit]) -> GeneratedAnswer:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_build_prompt(question=question, hits=hits),
            )
            payload = json.loads(response.text or "{}")
        except Exception as error:
            raise MissingGoogleConfiguration("Google chat configuration is unusable.") from error

        return GeneratedAnswer(
            answer=str(payload.get("answer", "")),
            grounded=bool(payload.get("grounded", False)),
            supporting_chunk_ids=[str(item) for item in payload.get("supporting_chunk_ids", [])],
        )

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
    context = "\n\n".join(
        f"Chunk ID: {hit.chunk.chunk_id}\nTitle: {hit.chunk.title}\nURL: {hit.chunk.url}\nText: {hit.chunk.text}"
        for hit in hits
    )

    return (
        "Answer the question using only the provided chunks. "
        "If the chunks do not support an answer, say the indexed site content does not contain enough information. "
        "Return only JSON with keys: answer, grounded, supporting_chunk_ids.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )


def _build_evidence_prompt(*, question: str, evidence: list[EvidenceSnippet]) -> str:
    context = "\n\n".join(
        "\n".join(
            [
                f"Evidence ID: {item.evidence_id}",
                f"Chunk ID: {item.chunk_id}",
                f"Title: {item.title}",
                f"URL: {item.url}",
                f"Heading path: {' > '.join(item.heading_path)}",
                f"Snippet: {item.snippet}",
                f"Nearby parent context: {item.nearby_context or item.snippet}",
            ]
        )
        for item in evidence
    )
    return (
        "Answer the question using only the provided evidence snippets. "
        "Return only JSON with keys: answer, groundedness, claims, supporting_evidence_ids. "
        "groundedness must be one of grounded, partially_grounded, not_grounded. "
        "Every material claim must include supporting_evidence_ids. "
        "If evidence is insufficient, say the indexed site content does not contain enough information.\n\n"
        f"Question: {question}\n\nEvidence:\n{context}"
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
    return GeneratedAnswer(
        answer=answer if groundedness != "not_grounded" else NOT_FOUND_ANSWER,
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
