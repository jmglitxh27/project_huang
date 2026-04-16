"""Stage 2 — knowledge graph construction from parsed arXiv corpora."""

from rnda.kg.graph_builder import build_graph_from_manifest, load_parsed_records

__all__ = ["build_graph_from_manifest", "load_parsed_records"]
