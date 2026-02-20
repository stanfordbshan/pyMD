"""
CLI entry-point for the pymd REST API server.

Usage::

    python -m pymd.api                          # defaults
    python -m pymd.api --port 9000 --reload     # dev mode
"""
from __future__ import annotations

import argparse
import importlib.util
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pymd.api",
        description="Launch the pymd REST API server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    return parser


def run_server(host: str, port: int, log_level: str, reload: bool) -> None:
    """Validate dependencies and start the uvicorn server."""
    if importlib.util.find_spec("fastapi") is None:
        raise RuntimeError(
            "Missing API dependencies. Install with: pip install -e '.[api]'"
        )
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "uvicorn is required to run the API server. "
            "Install with: pip install -e '.[api]'"
        )
    uvicorn.run(
        "pymd.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_server(
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            reload=args.reload,
        )
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
