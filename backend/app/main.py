from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.evals import router as evals_router
from app.api.health import router as health_router
from app.api.ingest import router as ingest_router
from app.config import get_settings
from app.db import initialize_database
from app.logging_config import configure_logging
from app.workspace import workspace_store

configure_logging()

settings = get_settings()
initialize_database(settings)
workspace_store.recover_interrupted_ingest_runs()

app = FastAPI(title="proj1 Hybrid RAG Backend")

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
