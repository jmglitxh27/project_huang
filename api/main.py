"""
Vercel FastAPI entrypoint.

Ensures the repo root is on ``sys.path`` before importing the package (Vercel builds
sometimes omit a working editable install).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from rnda.web.server_vercel import app as app
except Exception as exc:  # noqa: BLE001 — surface import failures in the HTTP layer
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _boot = FastAPI(title="RNDA (import failed)")

    @_boot.get("/{full_path:path}")
    async def _import_error(full_path: str) -> JSONResponse:
        return JSONResponse(
            {
                "error": "import_failed",
                "detail": repr(exc),
                "traceback": traceback.format_exc(),
                "hint": "Check Vercel installCommand (pip install -r requirements-vercel.txt && pip install -e . --no-deps) and repo layout.",
            },
            status_code=500,
        )

    app = _boot

__all__ = ["app"]
