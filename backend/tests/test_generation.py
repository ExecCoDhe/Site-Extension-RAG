from app.rag.generation import _answer_from_evidence_payload, _parse_json_object
from app.retrieval.models import EvidenceSnippet


def evidence() -> list[EvidenceSnippet]:
    return [
        EvidenceSnippet(
            evidence_id="evidence_1",
            chunk_id="chunk_1",
            section_id="section_1",
            parent_context_id="section_1",
            url="https://example.com/a",
            title="A",
            heading_path=["A"],
            snippet="alpha content",
            nearby_context="alpha content with surrounding context",
            dense_score=1.0,
            sparse_score=1.0,
            rerank_score=1.0,
        )
    ]


def test_evidence_generation_validates_claim_support() -> None:
    answer = _answer_from_evidence_payload(
        {
            "answer": "Alpha is supported, omega is not.",
            "groundedness": "grounded",
            "claims": [
                {
                    "text": "Alpha is supported.",
                    "supporting_evidence_ids": ["evidence_1"],
                    "supported": True,
                },
                {
                    "text": "Omega is unsupported.",
                    "supporting_evidence_ids": [],
                    "supported": False,
                },
            ],
            "supporting_evidence_ids": ["evidence_1", "missing"],
        },
        evidence(),
    )

    assert answer.groundedness == "partially_grounded"
    assert answer.supporting_evidence_ids == ["evidence_1"]
    assert answer.claims[1]["supported"] is False


def test_malformed_generation_json_becomes_empty_payload() -> None:
    assert _parse_json_object("not json") == {}
