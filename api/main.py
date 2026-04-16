"""Vercel FastAPI entrypoint — re-exports the RNDA ASGI app."""

from rnda.web.app import app

__all__ = ["app"]
