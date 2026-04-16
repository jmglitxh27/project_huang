from __future__ import annotations

import argparse
from pathlib import Path

from rnda.gap.pipeline import mine_gaps


def main() -> None:
    p = argparse.ArgumentParser(
        description="RNDA Stage 3 — gap & limitation mining (extract + embed + cluster).",
        epilog="Example: python -m rnda.gap.cli data/tiny_corpus/manifest.json --out data/gap_report.json",
    )
    p.add_argument(
        "manifest",
        type=Path,
        help="Path to Stage-1 manifest.json (from rnda-ingest). Not a placeholder — must exist.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/gap_report.json"),
        help="Output JSON report (default: data/gap_report.json).",
    )
    p.add_argument("--min-chars", type=int, default=80, help="Minimum snippet length (default: 80).")
    p.add_argument(
        "--no-fulltext-triggers",
        action="store_true",
        help="Only use section-based extraction (skip paragraph/regex full-text scans).",
    )
    p.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (default: all-MiniLM-L6-v2).",
    )
    p.add_argument(
        "--distance-threshold",
        type=float,
        default=0.28,
        help="Agglomerative distance threshold in embedding space (default: 0.28). Lower = tighter clusters.",
    )
    p.add_argument(
        "--tfidf-only",
        action="store_true",
        help="Skip neural embeddings; use TF-IDF + clustering only.",
    )

    args = p.parse_args()
    if not args.manifest.is_file():
        p.error(
            f"manifest file not found: {args.manifest.resolve()}. "
            "Pass the real path to manifest.json produced after ingestion (e.g. data/tiny_corpus/manifest.json)."
        )
    rep = mine_gaps(
        args.manifest,
        args.out,
        min_chars=args.min_chars,
        use_fulltext_triggers=not args.no_fulltext_triggers,
        embedding_model=args.embedding_model,
        distance_threshold=args.distance_threshold,
        prefer_semantic=not args.tfidf_only,
    )
    print(
        f"Wrote {rep.get('snippet_count', 0)} snippets, "
        f"{rep.get('cluster_count', 0)} clusters, backend={rep.get('embedding_backend')} "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
