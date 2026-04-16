from __future__ import annotations

import argparse
from pathlib import Path

from rnda.kg.graph_builder import build_graph_from_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="RNDA Stage 2 — knowledge graph from parsed corpus.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser(
        "build",
        help="Build a NetworkX graph from a Stage-1 manifest.json and save node-link JSON.",
    )
    b.add_argument(
        "manifest",
        type=Path,
        help="Path to manifest.json (from rnda-ingest query …).",
    )
    b.add_argument(
        "--out",
        type=Path,
        default=Path("data/kg/graph.json"),
        help="Output node-link JSON (default: data/kg/graph.json).",
    )
    b.add_argument(
        "--top-k-terms",
        type=int,
        default=18,
        help="Max TF-IDF terms per paper for MENTIONS edges (default: 18).",
    )

    args = p.parse_args()
    if args.command == "build":
        if not args.manifest.is_file():
            b.error(
                f"manifest not found: {args.manifest.resolve()}. "
                "Use the path to manifest.json from ingestion (e.g. data/tiny_corpus/manifest.json)."
            )
        n = build_graph_from_manifest(args.manifest, args.out, top_k_terms=args.top_k_terms)
        print(f"Wrote graph with {n.number_of_nodes()} nodes and {n.number_of_edges()} edges -> {args.out}")


if __name__ == "__main__":
    main()
