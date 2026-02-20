"""
FastAPI application factory.

Usage:
    uvicorn pymd.api.app:app --reload
"""
from fastapi import FastAPI

from pymd.api.routes import router


def create_app() -> FastAPI:
    application = FastAPI(title="pymd", version="0.1.0")
    application.include_router(router, prefix="/api")
    return application


app = create_app()
