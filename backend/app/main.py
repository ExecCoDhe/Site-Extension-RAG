from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.evals import router as evals_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.config import get_settings
from app.db import close_pool, get_connection
from app.logging_config import configure_logging
from app.workspace import workspace_store

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook.

    Startup: recover any ingest runs interrupted by a previous crash.
    Shutdown: close the database connection pool cleanly.
    """
    settings = get_settings()
    workspace_store.recover_interrupted_ingest_runs()
    yield
    close_pool()


app = FastAPI(title="proj1 Hybrid RAG Backend", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(evals_router)


@app.get("/ready")
def readiness() -> dict[str, str]:
    try:
        with get_connection(settings) as connection:
            connection.execute("SELECT 1").fetchone()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "database": "unavailable"},
        ) from error

    return {"status": "ok", "database": "ok"}
