"""
Lightweight FastAPI app for serverless (e.g. Vercel).

Does **not** import PyTorch, sentence-transformers, or the full ingest/KG pipeline.
Full pipeline: run ``rnda-web`` locally or deploy to a host without the 500MB limit.
"""

from __future__ import annotations

import io
import os
import re
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.templating import Jinja2Templates

from rnda.ingest.query_refinement import refine_search_with_llm
from rnda.report.literature_report import generate_literature_report
from rnda.web.graph_simplify import vis_network_payload_from_run

# ``rnda/web`` — works for both repo layout and editable install.
_WEB_DIR = Path(__file__).resolve().parent
# Project root (parent of ``rnda`` package); used for local ``data/`` runs.
_PROJECT_ROOT = _WEB_DIR.parent.parent
DATA_DIR = Path(os.environ.get("RNDA_DATA_DIR", str(_PROJECT_ROOT / "data")))
RUNS_ROOT = DATA_DIR / "runs"

_templates_dir = _WEB_DIR / "templates"
_static_dir = _WEB_DIR / "static"

templates = Jinja2Templates(directory=str(_templates_dir))

app = FastAPI(
    title="RNDA (serverless UI)",
    description="Dashboard shell — full pipeline runs locally or on a compute host (see POST /api/runs).",
)
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

_PIPELINE_UNAVAILABLE = (
    "The full RNDA pipeline is not available on this serverless deployment: "
    "ingest uses PyMuPDF, sentence-transformers, and PyTorch, which exceed the host size limit. "
    "Run the app locally (`rnda-web` or `uvicorn rnda.web.app:app`), or deploy to Railway, Fly.io, "
    "Render, a VM, or Modal."
)


def resolve_run_dir(run_id: str) -> Path:
    """``run_id`` is a folder name under ``data/runs/``, or an absolute path to a run directory."""
    d = RUNS_ROOT / run_id
    if (d / "manifest.json").is_file():
        return d.resolve()
    alt = Path(run_id)
    if alt.is_dir() and (alt / "manifest.json").is_file():
        return alt.resolve()
    raise HTTPException(status_code=404, detail="Run not found or missing manifest.json")


def _job_set(jid: str, **kw: Any) -> None:
    with _jobs_lock:
        cur = _jobs.get(jid, {})
        cur.update(kw)
        _jobs[jid] = cur


class RunRequest(BaseModel):
    topic: str = Field("", description="arXiv search topic / keywords (required unless resuming a run)")
    max_papers: int = Field(10, ge=1, le=300)
    categories: str = Field("", description="Comma-separated arXiv categories, e.g. cs.LG, cs.CL")
    submitted_from: str | None = Field(None, description="YYYYMMDDHHMMSS lower bound (optional)")
    submitted_to: str | None = Field(None, description="YYYYMMDDHHMMSS upper bound (optional)")
    sort: str = Field("submitted")
    grobid_url: str | None = None
    grobid_timeout: int = Field(120, ge=15, le=900, description="Seconds per PDF for GROBID HTTP read (lower = fail faster)")
    top_k_terms: int = Field(18, ge=5, le=80)
    gap_tfidf_only: bool = Field(False, description="Use TF-IDF for gap clustering (faster, no GPU)")
    gap_distance_threshold: float = Field(0.28, ge=0.05, le=1.0)
    max_hypotheses: int = Field(10, ge=1, le=50)
    llm_model: str = "gpt-4o-mini"
    use_llm: bool = Field(True, description="Use OpenAI for Stage 4 (needs OPENAI_API_KEY); else heuristic")
    resume_from_run: str | None = Field(None, description="If set, skip arXiv ingest and reuse this run id")
    natural_language_intent: str = Field(
        "",
        description="Optional prose describing what literature you want — used for semantic relevance and optional LLM refinement.",
    )
    refine_search_with_llm: bool = Field(
        False,
        description="If true and OPENAI_API_KEY is set, refine topic/keywords from NL + topic (Stage 1).",
    )
    relevance_filter: bool = Field(
        True,
        description="Pull a larger metadata pool, then keep top papers by abstract/title similarity to your intent.",
    )
    relevance_pool_size: int | None = Field(
        None,
        description="Override pool size before reranking (10–300; default: max(40, 4× max papers), capped at 300).",
    )
    relevance_min_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Drop papers below this cosine similarity (0 = keep top‑K only).",
    )


class RefineSearchBody(BaseModel):
    topic: str = ""
    natural_language: str = ""
    model: str = "gpt-4o-mini"


@app.post("/api/refine-search")
async def refine_search_api(body: RefineSearchBody) -> JSONResponse:
    """Return an arXiv-style topic string plus keywords from topic + natural language (uses OpenAI if configured)."""
    r = refine_search_with_llm(
        topic=body.topic.strip(),
        natural_language=(body.natural_language or "").strip() or None,
        model=body.model.strip() or "gpt-4o-mini",
    )
    return JSONResponse(
        {
            "arxiv_topic": r.arxiv_topic,
            "keywords": r.keywords,
            "rationale": r.rationale,
        }
    )


@app.get("/api/health")
async def health() -> JSONResponse:
    """Cheap probe for serverless cold starts (no optional deps)."""
    return JSONResponse({"status": "ok", "app": "server_vercel"})


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/viz/{run_id}", response_class=HTMLResponse)
async def graph_viz(request: Request, run_id: str) -> Any:
    resolve_run_dir(run_id)
    return templates.TemplateResponse("viz.html", {"request": request, "run_id": run_id})


@app.get("/api/runs")
async def list_runs() -> JSONResponse:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    if not RUNS_ROOT.is_dir():
        return JSONResponse(out)
    for p in sorted(RUNS_ROOT.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_dir():
            continue
        man = p / "manifest.json"
        if not man.is_file():
            continue
        out.append(
            {
                "id": p.name,
                "path": str(p),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                "has_report": (p / "novelty_report.json").is_file(),
            }
        )
    return JSONResponse(out)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job id")
    return JSONResponse(j)


@app.post("/api/runs")
async def start_run(body: RunRequest) -> JSONResponse:
    raise HTTPException(
        status_code=503,
        detail=_PIPELINE_UNAVAILABLE,
    )


@app.get("/api/runs/{run_id}/report")
async def get_report(run_id: str) -> JSONResponse:
    d = resolve_run_dir(run_id)
    p = d / "novelty_report.json"
    if not p.is_file():
        raise HTTPException(404, "No novelty_report.json for this run")
    import json

    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/runs/{run_id}/files")
async def run_files(run_id: str) -> JSONResponse:
    d = resolve_run_dir(run_id)
    files = [str(x.relative_to(d)) for x in d.rglob("*") if x.is_file()]
    return JSONResponse(sorted(files))


@app.get("/api/runs/{run_id}/viz-data")
async def viz_data(run_id: str) -> JSONResponse:
    d = resolve_run_dir(run_id)
    try:
        payload = vis_network_payload_from_run(d, max_nodes=280)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return JSONResponse(payload)


class LiteratureReportBody(BaseModel):
    model: str = Field("gpt-4o-mini")
    prefer_llm: bool = True


@app.post("/api/runs/{run_id}/literature-report")
async def post_literature_report(run_id: str, body: LiteratureReportBody) -> JSONResponse:
    d = resolve_run_dir(run_id)
    try:
        result = generate_literature_report(
            d,
            model=body.model,
            prefer_llm=body.prefer_llm,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return JSONResponse(result)


@app.get("/api/runs/{run_id}/bundle.zip")
async def download_literature_bundle(run_id: str) -> StreamingResponse:
    d = resolve_run_dir(run_id)
    buf = io.BytesIO()
    safe = re.sub(r"[^\w.\-]+", "_", run_id)[:80]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in d.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(d)
            if rel.parts and rel.parts[0] in ("__pycache__", ".git"):
                continue
            zf.write(path, arcname=str(rel).replace("\\", "/"))
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="rnda_literature_{safe}.zip"'},
    )
