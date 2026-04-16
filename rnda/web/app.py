from __future__ import annotations

import io
import re
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.templating import Jinja2Templates

from rnda.ingest.query_refinement import refine_search_with_llm
from rnda.web.graph_simplify import vis_network_payload_from_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_ROOT = DATA_DIR / "runs"

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "rnda" / "web" / "templates"))

app = FastAPI(title="RNDA", description="Research Novelty Discovery Agent — dashboard")
_static_dir = PROJECT_ROOT / "rnda" / "web" / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def resolve_run_dir(run_id: str) -> Path:
    """``run_id`` is a folder name under ``data/runs/``, or an absolute path to a run directory."""
    d = RUNS_ROOT / run_id
    if (d / "manifest.json").is_file():
        return d.resolve()
    alt = Path(run_id)
    if alt.is_dir() and (alt / "manifest.json").is_file():
        return alt.resolve()
    raise HTTPException(status_code=404, detail="Run not found or missing manifest.json")


def _format_ingest_progress(ev: dict[str, Any], grobid_timeout: int) -> str:
    st = ev.get("stage", "")
    idx = int(ev.get("index") or 0)
    total = int(ev.get("total") or 0)
    aid = ev.get("arxiv_id") or ""
    if st == "arxiv_api":
        return (
            "Waiting for arXiv API (first response can take 1–3 min if the service rate-limits you)…"
        )
    if st == "arxiv_pool":
        return (
            f"Fetching arXiv metadata pool {idx}/{total} (abstracts for relevance scoring)…"
        )
    if st == "relevance_score":
        return f"Scoring abstracts for relevance {idx}/{total}…"
    if st == "download":
        return f"Downloaded PDF {idx}/{total}: {aid}"
    if st == "parse_pdf":
        grobid_note = (
            f" GROBID timeout {grobid_timeout}s/paper — leave GROBID URL empty if not running."
            if grobid_timeout > 0
            else ""
        )
        return f"Parsing PDF {idx}/{total}: {aid} (PyMuPDF + optional GROBID).{grobid_note}"
    return str(ev)


def _job_set(jid: str, **kw: Any) -> None:
    with _jobs_lock:
        cur = _jobs.get(jid, {})
        cur.update(kw)
        _jobs[jid] = cur


def _run_full_job(jid: str, cfg: Any) -> None:
    from rnda.pipeline.orchestrator import run_pipeline

    run_dir = RUNS_ROOT / jid
    try:
        _job_set(jid, status="running", step="ingest", message="Starting arXiv ingest…", error=None)

        def on_ev(ev: dict[str, Any]) -> None:
            msg = _format_ingest_progress(ev, cfg.grobid_timeout)
            _job_set(jid, message=msg, ingest_progress=ev)

        result = run_pipeline(cfg, run_dir, on_ingest_progress=on_ev)
        _job_set(
            jid,
            status="done",
            step="done",
            message="Complete.",
            result=result,
        )
    except Exception as e:
        _job_set(jid, status="error", step="failed", error=str(e), message=str(e))


def _run_resume_job(jid: str, existing_id: str, cfg: Any) -> None:
    from rnda.pipeline.orchestrator import run_from_existing_run

    run_dir = RUNS_ROOT / existing_id
    if not (run_dir / "manifest.json").is_file():
        alt = Path(existing_id)
        if alt.is_dir() and (alt / "manifest.json").is_file():
            run_dir = alt.resolve()
    try:
        _job_set(jid, status="running", step="graph", message="Rebuilding graph and downstream stages…", error=None)
        if not (run_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"No manifest in {run_dir}")
        result = run_from_existing_run(run_dir, config=cfg)
        _job_set(jid, status="done", step="done", message="Complete.", result=result)
    except Exception as e:
        _job_set(jid, status="error", step="failed", error=str(e), message=str(e))


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


def _parse_categories(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


class RefineSearchBody(BaseModel):
    topic: str = ""
    natural_language: str = ""
    model: str = "gpt-4o-mini"


@app.get("/api/health")
async def health() -> JSONResponse:
    """Cheap liveness check for Railway / Render / load balancers (no heavy imports)."""
    return JSONResponse({"status": "ok", "app": "rnda.web.app"})


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
async def start_run(body: RunRequest, background_tasks: BackgroundTasks) -> JSONResponse:
    topic_ok = (body.topic or "").strip()
    nl_ok = (body.natural_language_intent or "").strip()
    if not body.resume_from_run and not topic_ok and not nl_ok:
        raise HTTPException(
            status_code=400,
            detail="Provide topic/keywords and/or natural_language_intent for a new run",
        )
    if body.relevance_pool_size is not None and not (10 <= body.relevance_pool_size <= 300):
        raise HTTPException(status_code=400, detail="relevance_pool_size must be between 10 and 300 or omitted")
    cats = _parse_categories(body.categories)
    from rnda.pipeline.orchestrator import PipelineConfig

    cfg = PipelineConfig(
        topic=body.topic.strip(),
        max_papers=body.max_papers,
        categories=cats,
        submitted_from=body.submitted_from,
        submitted_to=body.submitted_to,
        sort=body.sort,
        grobid_url=body.grobid_url,
        grobid_timeout=body.grobid_timeout,
        top_k_terms=body.top_k_terms,
        gap_tfidf_only=body.gap_tfidf_only,
        gap_distance_threshold=body.gap_distance_threshold,
        max_hypotheses=body.max_hypotheses,
        llm_model=body.llm_model,
        use_llm=body.use_llm,
        natural_language_intent=nl_ok or None,
        refine_search_with_llm=body.refine_search_with_llm,
        relevance_filter=body.relevance_filter,
        relevance_pool_size=body.relevance_pool_size,
        relevance_min_score=body.relevance_min_score,
    )

    jid = str(uuid.uuid4())
    _job_set(jid, status="queued", step="queued", message="Queued…")

    if body.resume_from_run:
        rid = body.resume_from_run.strip()
        background_tasks.add_task(_run_resume_job, jid, rid, cfg)
    else:
        background_tasks.add_task(_run_full_job, jid, cfg)

    return JSONResponse({"job_id": jid, "poll": f"/api/jobs/{jid}"})


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
    from rnda.report.literature_report import generate_literature_report

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
