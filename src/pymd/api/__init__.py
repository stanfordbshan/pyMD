"""FastAPI transport layer for pymd.

Models, service adapters, and routes — no domain logic.

The ``create_app()`` factory is lazily imported so that
``import pymd.api`` never forces a FastAPI dependency.
"""


def create_app():
    """Deferred import of the FastAPI application factory."""
    from pymd.api.app import create_app as _create_app

    return _create_app()
