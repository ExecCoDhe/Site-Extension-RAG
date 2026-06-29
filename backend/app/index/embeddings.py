from typing import Protocol

from langchain_google_genai import GoogleGenerativeAIEmbeddings
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


class LangChainEmbeddingClient:
    def __init__(self, *, api_key: str | None, model: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        if not api_key:
            raise MissingGoogleConfiguration("GEMINI_API_KEY is not configured.")

        self._model = model
        self._timeout_seconds = timeout_seconds  # signature parity; GGE has no per-request timeout
        # Do NOT pass output_dimensionality -> preserves default dim (3072).
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=api_key, vertexai=False
        )

    @traceable(name="embed_documents")
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._embeddings.embed_documents(
                texts, task_type="RETRIEVAL_DOCUMENT", batch_size=GOOGLE_EMBEDDING_BATCH_SIZE
            )
        except Exception as error:
            raise MissingGoogleConfiguration(_configuration_error_message(error)) from error

    @traceable(name="embed_query")
    def embed_query(self, text: str) -> list[float]:
        try:
            return self._embeddings.embed_query(text, task_type="RETRIEVAL_QUERY")
        except Exception as error:
            raise MissingGoogleConfiguration(_configuration_error_message(error)) from error


def _configuration_error_message(error: Exception) -> str:
    message = str(error)
    if "API_KEY_INVALID" in message or "API Key not found" in message:
        return "GEMINI_API_KEY is invalid or not enabled for the Gemini API."
    return "Google embedding configuration is unusable."
