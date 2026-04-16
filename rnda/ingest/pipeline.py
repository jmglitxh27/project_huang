from __future__ import annotations

import json
import os
from pathlib import Path
from collections.abc import Callable
from typing import Any

import arxiv

from rnda.ingest.arxiv_client import (
    collect_search_results_list,
    download_pdfs_for_results,
    fetch_papers,
    result_to_metadata,
    search_arxiv,
)
from rnda.ingest.models import ArxivMetadata, ParsedPaper
from rnda.ingest.pdf_parser import parse_pdf_file
from rnda.ingest.query_builder import build_arxiv_query
from rnda.ingest.relevance import build_relevance_intent_text, score_metadata_by_semantic_similarity


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest_metadata_only(
    query: str,
    out_json: Path,
    *,
    max_results: int = 25,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> list[ArxivMetadata]:
    papers = search_arxiv(query, max_results=max_results, sort_by=sort_by, sort_order=sort_order)
    payload = {"query": query, "count": len(papers), "papers": [p.to_json_dict() for p in papers]}
    write_json(payload, out_json)
    return papers


def ingest_with_pdfs(
    query: str,
    out_dir: Path,
    *,
    max_results: int = 10,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    grobid_base_url: str | None = None,
    grobid_timeout: int = 120,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    relevance_filter: bool = True,
    relevance_text: str | None = None,
    relevance_keywords: list[str] | None = None,
    natural_language_intent: str | None = None,
    relevance_pool_size: int | None = None,
    relevance_min_score: float = 0.0,
    relevance_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    stage1_extra: dict[str, Any] | None = None,
) -> list[ParsedPaper]:
    """
    Download PDFs under out_dir/pdfs, write one JSON per paper under out_dir/parsed,
    and a manifest at out_dir/manifest.json.

    ``on_progress`` receives events such as
    ``{"stage": "arxiv_api"|"arxiv_pool"|"relevance_score"|"download"|"parse_pdf", ...}``.

    With ``relevance_filter=True`` (default), the pipeline requests a larger metadata pool,
    scores title+abstract against ``relevance_text`` (semantic similarity), then downloads
    only the top ``max_results`` matches.
    """
    pdf_dir = out_dir / "pdfs"
    parsed_dir = out_dir / "parsed"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_cb(stage: str, index: int, total: int, arxiv_id: str) -> None:
        if on_progress:
            on_progress(
                {"stage": stage, "index": index, "total": total, "arxiv_id": arxiv_id}
            )

    intent_plain = build_relevance_intent_text(
        arxiv_query_body=relevance_text or query,
        natural_language=natural_language_intent,
        extra_keywords=relevance_keywords,
    )

    raw_pool = relevance_pool_size if relevance_pool_size is not None else max(40, max_results * 4)
    effective_pool = min(300, max(raw_pool, max_results))

    if relevance_filter:
        if on_progress:
            on_progress(
                {
                    "stage": "arxiv_pool",
                    "index": 0,
                    "total": effective_pool,
                    "arxiv_id": "",
                }
            )
        try:
            pool_results = collect_search_results_list(
                query,
                max_results=effective_pool,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        except Exception:
            if on_progress:
                on_progress(
                    {
                        "stage": "arxiv_pool",
                        "index": effective_pool,
                        "total": effective_pool,
                        "arxiv_id": "",
                    }
                )
            raise
        if on_progress:
            on_progress(
                {
                    "stage": "arxiv_pool",
                    "index": len(pool_results),
                    "total": effective_pool,
                    "arxiv_id": "",
                }
            )
        if not pool_results:
            rel_payload = {
                "intent_text": intent_plain,
                "pool_requested": effective_pool,
                "pool_returned": 0,
                "embed_model": relevance_embed_model,
                "min_score": relevance_min_score,
                "selected": [],
                "warning": "arXiv returned no results for this query.",
            }
            if stage1_extra:
                rel_payload["refinement"] = stage1_extra
            write_json(rel_payload, out_dir / "stage1_relevance.json")
            pairs = []
        else:
            metas = [result_to_metadata(r) for r in pool_results]
            id_to_res = {r.get_short_id(): r for r in pool_results}
            if on_progress:
                on_progress(
                    {
                        "stage": "relevance_score",
                        "index": 0,
                        "total": len(metas),
                        "arxiv_id": "",
                    }
                )
            scored = score_metadata_by_semantic_similarity(
                metas,
                intent_plain,
                model_name=relevance_embed_model,
            )
            if relevance_min_score > 0:
                scored = [(m, s) for m, s in scored if s >= relevance_min_score]
            picked = scored[:max_results]
            if on_progress:
                on_progress(
                    {
                        "stage": "relevance_score",
                        "index": len(picked),
                        "total": len(metas),
                        "arxiv_id": "",
                    }
                )
            ordered_results: list = []
            rank_rows: list[dict[str, Any]] = []
            for rank, (meta, sc) in enumerate(picked, start=1):
                r = id_to_res.get(meta.arxiv_id)
                if r is not None:
                    ordered_results.append(r)
                rank_rows.append(
                    {
                        "rank": rank,
                        "arxiv_id": meta.arxiv_id,
                        "title": meta.title,
                        "relevance_score": round(sc, 5),
                    }
                )
            rel_payload = {
                "intent_text": intent_plain,
                "pool_requested": effective_pool,
                "pool_returned": len(pool_results),
                "embed_model": relevance_embed_model,
                "min_score": relevance_min_score,
                "selected": rank_rows,
            }
            if stage1_extra:
                rel_payload["refinement"] = stage1_extra
            write_json(rel_payload, out_dir / "stage1_relevance.json")

            if ordered_results:
                pairs = download_pdfs_for_results(
                    ordered_results,
                    pdf_dir,
                    progress=_fetch_cb if on_progress else None,
                )
            else:
                pairs = []
    else:
        pairs = fetch_papers(
            query,
            pdf_dir,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            progress=_fetch_cb if on_progress else None,
        )
        rel_payload = {
            "intent_text": intent_plain,
            "relevance_filter": False,
            "note": "Semantic reranking skipped.",
        }
        if stage1_extra:
            rel_payload["refinement"] = stage1_extra
        write_json(rel_payload, out_dir / "stage1_relevance.json")
    parsed: list[ParsedPaper] = []
    manifest_rows: list[dict[str, Any]] = []

    grobid_url = grobid_base_url if grobid_base_url is not None else os.environ.get("RNDA_GROBID_URL")
    n_total = len(pairs)

    for idx, (meta, path) in enumerate(pairs, start=1):
        if on_progress:
            on_progress(
                {
                    "stage": "parse_pdf",
                    "index": idx,
                    "total": n_total,
                    "arxiv_id": meta.arxiv_id,
                }
            )
        paper = parse_pdf_file(
            path,
            meta,
            grobid_base_url=grobid_url,
            grobid_timeout=grobid_timeout,
        )
        parsed.append(paper)
        safe_id = meta.arxiv_id.replace("/", "_")
        pj = parsed_dir / f"{safe_id}.json"
        write_json(paper.to_json_dict(), pj)
        try:
            parsed_rel = str(pj.relative_to(out_dir.resolve()))
        except ValueError:
            parsed_rel = str(pj)
        try:
            pdf_rel = str(path.relative_to(out_dir.resolve()))
        except ValueError:
            pdf_rel = str(path)
        manifest_rows.append(
            {
                "arxiv_id": meta.arxiv_id,
                "title": meta.title,
                "pdf": pdf_rel,
                "parsed_json": parsed_rel,
            }
        )

    write_json(
        {"query": query, "count": len(manifest_rows), "papers": manifest_rows},
        out_dir / "manifest.json",
    )
    return parsed


def ingest_topic(
    topic: str,
    out_dir: Path,
    *,
    max_results: int = 10,
    submitted_from: str | None = None,
    submitted_to: str | None = None,
    categories: list[str] | None = None,
    download_pdfs: bool = True,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    grobid_base_url: str | None = None,
    grobid_timeout: int = 120,
) -> list[ArxivMetadata] | list[ParsedPaper]:
    q = build_arxiv_query(
        topic,
        submitted_from_yyyymmdd=submitted_from,
        submitted_to_yyyymmdd=submitted_to,
        categories=categories,
    )
    if download_pdfs:
        return ingest_with_pdfs(
            q,
            out_dir,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            grobid_base_url=grobid_base_url,
            grobid_timeout=grobid_timeout,
        )
    ingest_metadata_only(q, out_dir / "metadata.json", max_results=max_results)
    return search_arxiv(q, max_results=max_results)
