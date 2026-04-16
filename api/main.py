"""Vercel FastAPI entrypoint — slim ASGI app (no PyTorch / full pipeline)."""

from rnda.web.server_vercel import app

__all__ = ["app"]
