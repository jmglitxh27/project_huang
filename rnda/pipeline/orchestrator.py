from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

import arxiv

from rnda.gap.pipeline import mine_gaps
from rnda.ingest.pipeline import ingest_with_pdfs
from rnda.ingest.query_builder import build_arxiv_query
from rnda.ingest.query_refinement import RefinedSearch, ensure_search_topic, refine_search_with_llm
from rnda.kg.graph_builder import build_graph_from_manifest
from rnda.novelty.pipeline import run_stage_4_and_5


_SORT_MAP: dict[str, arxiv.SortCriterion] = {
    "relevance": arxiv.SortCriterion.Relevance,
    "last_updated": arxiv.SortCriterion.LastUpdatedDate,
    "submitted": arxiv.SortCriterion.SubmittedDate,
}


@dataclass
class PipelineConfig:
    topic: str = ""
    max_papers: int = 10
    categories: list[str] = field(default_factory=list)
    submitted_from: str | None = None
    submitted_to: str | None = None
    sort: str = "submitted"
    grobid_url: str | None = None
    grobid_timeout: int = 120
    top_k_terms: int = 18
    gap_tfidf_only: bool = False
    gap_distance_threshold: float = 0.28
    gap_min_chars: int = 80
    max_hypotheses: int = 10
    llm_model: str = "gpt-4o-mini"
    use_llm: bool = True
    natural_language_intent: str | None = None
    refine_search_with_llm: bool = False
    relevance_filter: bool = True
    relevance_pool_size: int | None = None
    relevance_min_score: float = 0.0
    relevance_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def run_pipeline(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    on_ingest_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Stage 1 ingest → Stage 2 KG → Stage 3 gaps → Stages 4–5 novelty.
    ``run_dir`` is created or reused; outputs are written under it.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sort_by = _SORT_MAP.get(cfg.sort.lower().replace("-", "_"), arxiv.SortCriterion.SubmittedDate)
    nl = (cfg.natural_language_intent or "").strip()
    topic_seed = (cfg.topic or "").strip()
    refined: RefinedSearch | None = None
    if cfg.refine_search_with_llm and (nl or topic_seed):
        refined = refine_search_with_llm(
            topic=topic_seed,
            natural_language=nl or None,
            model=cfg.llm_model,
        )
        topic_for_query = refined.arxiv_topic.strip() or topic_seed
    else:
        topic_for_query = topic_seed

    if not (topic_for_query or "").strip() and nl:
        topic_for_query = ensure_search_topic("", nl)

    q = build_arxiv_query(
        topic_for_query,
        submitted_from_yyyymmdd=cfg.submitted_from,
        submitted_to_yyyymmdd=cfg.submitted_to,
        categories=cfg.categories or None,
    )

    stage1_extra: dict = {}
    if refined:
        stage1_extra = {
            "arxiv_topic": refined.arxiv_topic,
            "keywords_for_relevance": refined.keywords,
            "rationale": refined.rationale,
        }
    elif nl:
        stage1_extra = {"natural_language_intent": nl}

    ingest_with_pdfs(
        q,
        run_dir,
        max_results=cfg.max_papers,
        sort_by=sort_by,
        sort_order=sort_order,
        grobid_base_url=cfg.grobid_url,
        grobid_timeout=cfg.grobid_timeout,
        on_progress=on_ingest_progress,
        relevance_filter=cfg.relevance_filter,
        relevance_text=topic_for_query,
        relevance_keywords=refined.keywords if refined else None,
        natural_language_intent=nl or None,
        relevance_pool_size=cfg.relevance_pool_size,
        relevance_min_score=cfg.relevance_min_score,
        relevance_embed_model=cfg.relevance_embed_model,
        stage1_extra=stage1_extra or None,
    )

    manifest = run_dir / "manifest.json"
    graph_json = run_dir / "graph.json"
    gap_json = run_dir / "gap_report.json"

    build_graph_from_manifest(manifest, graph_json, top_k_terms=cfg.top_k_terms)

    mine_gaps(
        manifest,
        gap_json,
        min_chars=cfg.gap_min_chars,
        use_fulltext_triggers=True,
        distance_threshold=cfg.gap_distance_threshold,
        prefer_semantic=not cfg.gap_tfidf_only,
    )

    novelty = run_stage_4_and_5(
        graph_json,
        gap_json,
        run_dir,
        manifest_path=manifest,
        max_hypotheses=cfg.max_hypotheses,
        llm_model=cfg.llm_model,
        use_llm=cfg.use_llm,
    )

    out = {
        "run_dir": str(run_dir.resolve()),
        "arxiv_query": q,
        "manifest": str(manifest),
        "graph_json": str(graph_json),
        "gap_report": str(gap_json),
        "novelty": novelty,
    }
    sr = run_dir / "stage1_relevance.json"
    if sr.is_file():
        out["stage1_relevance"] = str(sr.resolve())
    if refined:
        out["refined_topic"] = refined.arxiv_topic
    return out


def run_from_existing_run(
    run_dir: Path,
    *,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Re-run stages 2–5 on an existing ingest folder (manifest + parsed PDFs already present)."""
    run_dir = Path(run_dir)
    manifest = run_dir / "manifest.json"
    graph_json = run_dir / "graph.json"
    gap_json = run_dir / "gap_report.json"
    cfg = config or PipelineConfig(topic="")

    build_graph_from_manifest(manifest, graph_json, top_k_terms=cfg.top_k_terms)

    mine_gaps(
        manifest,
        gap_json,
        min_chars=cfg.gap_min_chars,
        use_fulltext_triggers=True,
        distance_threshold=cfg.gap_distance_threshold,
        prefer_semantic=not cfg.gap_tfidf_only,
    )

    novelty = run_stage_4_and_5(
        graph_json,
        gap_json,
        run_dir,
        manifest_path=manifest,
        max_hypotheses=cfg.max_hypotheses,
        llm_model=cfg.llm_model,
        use_llm=cfg.use_llm,
    )

    return {
        "run_dir": str(run_dir.resolve()),
        "manifest": str(manifest),
        "novelty": novelty,
    }
