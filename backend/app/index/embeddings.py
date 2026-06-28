from typing import Protocol

import httpx
from google import genai
from langsmith import traceable

GOOGLE_EMBEDDING_BATCH_SIZE = 50
DEFAULT_TIMEOUT_SECONDS = 60


class MissingGoogleConfiguration(Exception):
    """Raised when backend Google API configuration cannot be used."""


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    def embed_query(self, text: str) -> list[float]:
        pass


class GoogleEmbeddingClient:
    def __init__(self, *, api_key: str | None, model: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if not api_key:
            raise MissingGoogleConfiguration("GEMINI_API_KEY is not configured.")

        self._model = model
        self._client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_seconds * 1000},
        )

    @traceable(name="embed_documents")
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    @traceable(name="embed_query")
    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), GOOGLE_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + GOOGLE_EMBEDDING_BATCH_SIZE]
            embeddings.extend(self._embed_batch(batch, task_type=task_type))
        return embeddings

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        try:
            response = self._client.models.embed_content(
                model=self._model,
                contents=texts,
                config={"task_type": task_type},
            )
        except Exception as error:
            raise MissingGoogleConfiguration(_configuration_error_message(error)) from error

        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            raise MissingGoogleConfiguration("Google embedding response did not include vectors.")

        return [embedding.values for embedding in embeddings]


def _configuration_error_message(error: Exception) -> str:
    message = str(error)
    if "API_KEY_INVALID" in message or "API Key not found" in message:
        return "GEMINI_API_KEY is invalid or not enabled for the Gemini API."
    return "Google embedding configuration is unusable."
