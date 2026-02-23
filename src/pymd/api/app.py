"""
FastAPI application factory.

Usage::

    uvicorn pymd.api.app:app --reload
    python -m pymd.api
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pymd
from pymd.api.models import HealthResponse
from pymd.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="pymd API",
        version=pymd.__version__,
        description="REST API for the pymd molecular-dynamics framework.",
    )

    # CORS — permissive defaults for local pywebview / browser clients.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health-check (outside /api prefix).
    @application.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(version=pymd.__version__)

    application.include_router(router, prefix="/api")
    return application


app = create_app()
