from __future__ import annotations

import math
from typing import Any

import networkx as nx


def concept_nodes(G: nx.MultiDiGraph) -> list[str]:
    return [n for n, a in G.nodes(data=True) if a.get("kind") == "concept"]


def sparse_concept_pairs(G: nx.MultiDiGraph, *, limit: int = 40) -> list[tuple[str, str, float]]:
    """
    Pairs of concepts with **low** CO_OCCURS weight (under-studied combinations in this slice).
    Sorted ascending by weight (minimum weight kept if parallel edges exist).
    """
    best: dict[tuple[str, str], float] = {}
    for u, v, _k, data in G.edges(keys=True, data=True):
        if data.get("relation") != "CO_OCCURS":
            continue
        if not (str(u).startswith("concept:") and str(v).startswith("concept:")):
            continue
        a, b = sorted([str(u), str(v)])
        w = float(data.get("weight", 0.0))
        cur = best.get((a, b))
        if cur is None or w < cur:
            best[(a, b)] = w
    pairs = [(a, b, w) for (a, b), w in best.items()]
    pairs.sort(key=lambda t: t[2])
    return pairs[:limit]


def concept_pagerank(G: nx.MultiDiGraph) -> dict[str, float]:
    """PageRank on an undirected projection of concept--concept CO_OCCURS edges."""
    UG = nx.Graph()
    for u, v, _k, data in G.edges(keys=True, data=True):
        if data.get("relation") != "CO_OCCURS":
            continue
        if not (str(u).startswith("concept:") and str(v).startswith("concept:")):
            continue
        w = float(data.get("weight", 1e-6))
        UG.add_edge(u, v, weight=max(w, 1e-9))
    if UG.number_of_nodes() == 0:
        return {}
    return nx.pagerank(UG, weight="weight")


def graph_context_summary(G: nx.MultiDiGraph, *, top_deg: int = 25) -> dict[str, Any]:
    concepts = concept_nodes(G)
    deg: list[tuple[str, int]] = []
    for c in concepts:
        deg.append((c, G.degree(c)))
    deg.sort(key=lambda x: -x[1])
    labels: dict[str, str] = {}
    for n in concepts:
        labels[n] = G.nodes[n].get("label", n)
    return {
        "concept_count": len(concepts),
        "top_concepts_by_degree": [
            {"id": n, "label": labels.get(n, n), "degree": d} for n, d in deg[:top_deg]
        ],
    }


def match_concepts_in_hypothesis(text: str, G: nx.MultiDiGraph) -> list[str]:
    """Match hypothesis text to KG concept node labels (substring, case-insensitive)."""
    t = text.lower()
    hits: list[str] = []
    for n, attr in G.nodes(data=True):
        if attr.get("kind") != "concept":
            continue
        lab = str(attr.get("label", "")).strip()
        if len(lab) < 3:
            continue
        if lab.lower() in t:
            hits.append(n)
    return hits
