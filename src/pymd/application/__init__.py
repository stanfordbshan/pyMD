"""Transport-agnostic orchestration layer.

Provides stable entry points consumed by CLI, GUI, and API adapters.
All functions accept plain Python primitives — no Pydantic, no HTTP,
no GUI types leak in.
"""
from .md_workflow import MDWorkflow

__all__ = ["MDWorkflow"]
