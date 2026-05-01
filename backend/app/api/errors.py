from typing import Any, Literal

from fastapi.responses import JSONResponse
from pydantic import BaseModel

ErrorCode = Literal[
    "BACKEND_UNAVAILABLE",
    "ACTIVE_JOB",
    "INGEST_TIMEOUT",
    "INGEST_INTERRUPTED",
    "INGEST_FAILED",
    "NO_PAGES_INDEXED",
    "MISSING_API_KEY",
    "CHAT_BEFORE_READY",
]


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    retryable: bool = True


class ErrorEnvelope(BaseModel):
    error: ErrorBody


def error_response(
    *,
    code: ErrorCode,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
    retryable: bool = True,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            details=details,
            retryable=retryable,
        )
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump())
