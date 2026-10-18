"""DocIntel entrypoint: one FastAPI app serves the API and the static website."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.routes import limiter, router
from src.config import get_settings
from src.utils import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("docintel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at startup (added in later phases)."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("DocIntel starting (llm_provider=%s, llm_enabled=%s)",
                settings.llm_provider, settings.llm_enabled)
    yield


app = FastAPI(title="DocIntel", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    """Liveness probe; also reports whether an LLM key is configured (never the key)."""
    return {"status": "ok", "llm_enabled": settings.llm_enabled,
            "llm_provider": settings.llm_provider}


app.include_router(router)

if settings.web_dir.joinpath("index.html").exists():
    app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
