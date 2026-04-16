from __future__ import annotations

import argparse
import os
from pathlib import Path

import arxiv

from rnda.ingest.arxiv_client import fetch_by_id, normalize_arxiv_id
from rnda.ingest.query_refinement import ensure_search_topic, refine_search_with_llm
from rnda.ingest.pipeline import ingest_metadata_only, ingest_with_pdfs
from rnda.ingest.pdf_parser import parse_pdf_file
from rnda.ingest.query_builder import build_arxiv_query


def _sort_criterion(name: str) -> arxiv.SortCriterion:
    mapping = {
        "relevance": arxiv.SortCriterion.Relevance,
        "last_updated": arxiv.SortCriterion.LastUpdatedDate,
        "submitted": arxiv.SortCriterion.SubmittedDate,
    }
    key = name.lower().replace("-", "_")
    if key not in mapping:
        raise argparse.ArgumentTypeError(f"sort must be one of {list(mapping)}")
    return mapping[key]


def main() -> None:
    p = argparse.ArgumentParser(
        description="RNDA Stage 1 — arXiv ingestion; PyMuPDF + optional GROBID parsing.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Search arXiv and optionally download + parse PDFs.")
    q.add_argument(
        "topic",
        nargs="?",
        default="",
        help="Topic / keywords for the arXiv query (optional if --nl is set).",
    )
    q.add_argument(
        "--out",
        type=Path,
        default=Path("data/arxiv_run"),
        help="Output directory (default: data/arxiv_run).",
    )
    q.add_argument("--max-results", type=int, default=5)
    q.add_argument(
        "--sort",
        type=_sort_criterion,
        default="submitted",
        help="relevance | last_updated | submitted (default: submitted).",
    )
    q.add_argument(
        "--submitted-from",
        help="Lower bound YYYYMMDDHHMMSS for submittedDate (optional).",
    )
    q.add_argument(
        "--submitted-to",
        help="Upper bound YYYYMMDDHHMMSS for submittedDate (optional).",
    )
    q.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="arXiv category prefix, e.g. cs.LG (repeatable).",
    )
    q.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only call the API (no PDF download). Writes metadata.json.",
    )
    q.add_argument(
        "--grobid-url",
        default=os.environ.get("RNDA_GROBID_URL"),
        help="GROBID base URL (e.g. http://127.0.0.1:8070). Env: RNDA_GROBID_URL.",
    )
    q.add_argument(
        "--grobid-timeout",
        type=int,
        default=120,
        help="GROBID read timeout in seconds per PDF (default: 120).",
    )
    q.add_argument(
        "--nl",
        "--natural-language",
        dest="natural_language",
        default=None,
        help="Natural-language research intent — improves semantic relevance scoring against abstracts.",
    )
    q.add_argument(
        "--refine-llm",
        action="store_true",
        help="Use OpenAI to refine topic/keywords from --nl + topic (needs OPENAI_API_KEY).",
    )
    q.add_argument(
        "--no-relevance-filter",
        action="store_true",
        help="Disable abstract/title scoring; download the first N API hits only.",
    )
    q.add_argument(
        "--relevance-pool",
        type=int,
        default=None,
        help="Metadata pool size before reranking (10–300; default: max(40, 4× max-results)).",
    )
    q.add_argument(
        "--relevance-min-score",
        type=float,
        default=0.0,
        help="Drop papers below this similarity (0–1; default 0 = keep top‑K only).",
    )

    one = sub.add_parser("fetch-one", help="Resolve one arXiv id or URL and parse its PDF.")
    one.add_argument("arxiv_id")
    one.add_argument("--out", type=Path, default=Path("data/single"))
    one.add_argument(
        "--no-download",
        action="store_true",
        help="Only print metadata (no PDF).",
    )
    one.add_argument(
        "--grobid-url",
        default=os.environ.get("RNDA_GROBID_URL"),
        help="GROBID base URL (e.g. http://127.0.0.1:8070). Env: RNDA_GROBID_URL.",
    )
    one.add_argument("--grobid-timeout", type=int, default=120)

    args = p.parse_args()

    if args.command == "query":
        topic_seed = (args.topic or "").strip()
        nl = (args.natural_language or "").strip() if args.natural_language else ""
        if not topic_seed and not nl:
            raise SystemExit("Provide a topic and/or --natural-language / --nl intent.")
        refined = None
        if args.refine_llm and (nl or topic_seed):
            refined = refine_search_with_llm(
                topic=topic_seed,
                natural_language=nl or None,
                model=os.environ.get("RNDA_LLM_MODEL", "gpt-4o-mini"),
            )
            topic_for_build = refined.arxiv_topic.strip() or topic_seed
        else:
            topic_for_build = topic_seed
        if not topic_for_build.strip() and nl:
            topic_for_build = ensure_search_topic("", nl)
        full_query = build_arxiv_query(
            topic_for_build,
            submitted_from_yyyymmdd=args.submitted_from,
            submitted_to_yyyymmdd=args.submitted_to,
            categories=args.categories,
        )
        stage1_extra: dict | None = None
        if refined:
            stage1_extra = {
                "arxiv_topic": refined.arxiv_topic,
                "keywords_for_relevance": refined.keywords,
                "rationale": refined.rationale,
            }
        elif nl:
            stage1_extra = {"natural_language_intent": nl}
        if args.metadata_only:
            out_json = args.out / "metadata.json"
            ingest_metadata_only(
                full_query,
                out_json,
                max_results=args.max_results,
                sort_by=args.sort,
                sort_order=arxiv.SortOrder.Descending,
            )
            print(f"Wrote {out_json}")
            return
        papers = ingest_with_pdfs(
            full_query,
            args.out,
            max_results=args.max_results,
            sort_by=args.sort,
            sort_order=arxiv.SortOrder.Descending,
            grobid_base_url=args.grobid_url,
            grobid_timeout=args.grobid_timeout,
            relevance_filter=not args.no_relevance_filter,
            relevance_text=topic_for_build,
            relevance_keywords=refined.keywords if refined else None,
            natural_language_intent=nl or None,
            relevance_pool_size=args.relevance_pool,
            relevance_min_score=args.relevance_min_score,
            stage1_extra=stage1_extra,
        )
        print(f"Downloaded and parsed {len(papers)} papers under {args.out}")
        return

    if args.command == "fetch-one":
        aid = normalize_arxiv_id(args.arxiv_id)
        meta, pdf_path = fetch_by_id(aid, None if args.no_download else args.out / "pdfs")
        if args.no_download or pdf_path is None:
            print(meta.to_json_dict())
            return
        parsed = parse_pdf_file(
            pdf_path,
            meta,
            grobid_base_url=args.grobid_url,
            grobid_timeout=args.grobid_timeout,
        )
        out_path = args.out / "parsed" / f"{aid.replace('/', '_')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            __import__("json").dumps(parsed.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")
        return


if __name__ == "__main__":
    main()
