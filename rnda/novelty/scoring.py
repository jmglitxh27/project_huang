from __future__ import annotations

import math
from typing import Any

import networkx as nx

from rnda.novelty.signals import concept_pagerank, match_concepts_in_hypothesis

_ALPHA = 0.50
_BETA = 0.30
_GAMMA = 0.20


def _pair_novelty_score(
    concepts: list[str],
    G: nx.MultiDiGraph,
) -> float:
    """Higher when linked concept pairs have low CO_OCCURS weight (unexplored in this slice)."""
    if len(concepts) < 2:
        return 0.62

    scores: list[float] = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            u, v = sorted([concepts[i], concepts[j]])
            ed = G.get_edge_data(u, v)
            if not ed:
                scores.append(1.0)
                continue
            w = 0.0
            found = False
            for _k, data in ed.items():
                if data.get("relation") == "CO_OCCURS":
                    w = max(w, float(data.get("weight", 0.0)))
                    found = True
            if not found:
                scores.append(1.0)
            else:
                scores.append(1.0 / (1.0 + math.log1p(w)))
    return float(sum(scores) / len(scores)) if scores else 0.7


def score_hypothesis(
    item: dict[str, Any],
    G: nx.MultiDiGraph,
    pr: dict[str, float],
) -> dict[str, Any]:
    text = str(item.get("hypothesis", ""))
    listed = item.get("concepts") or []
    concept_ids: list[str] = [str(x) for x in listed if str(x).startswith("concept:")]
    matched = match_concepts_in_hypothesis(text, G)
    for m in matched:
        if m not in concept_ids:
            concept_ids.append(m)
    concept_ids = list(dict.fromkeys(concept_ids))
    pool = list(dict.fromkeys(concept_ids + matched))[:24]
    if len(pool) >= 2:
        novelty = _pair_novelty_score(pool, G)
    else:
        novelty = 0.62

    denom = max(len(concept_ids), 1)
    in_kg = sum(1 for c in concept_ids if G.has_node(c))
    feasibility = in_kg / denom if denom else 0.5

    imp_vals: list[float] = []
    for c in concept_ids:
        if c in pr:
            imp_vals.append(pr[c])
    for c in matched:
        if c in pr:
            imp_vals.append(pr[c])
    impact = float(sum(imp_vals) / len(imp_vals)) if imp_vals else 0.45

    combined = _ALPHA * novelty + _BETA * feasibility + _GAMMA * impact

    return {
        **item,
        "scores": {
            "novelty": round(novelty, 4),
            "feasibility": round(feasibility, 4),
            "impact": round(impact, 4),
            "combined": round(combined, 4),
            "weights": {"alpha": _ALPHA, "beta": _BETA, "gamma": _GAMMA},
        },
        "matched_concept_nodes": matched[:30],
    }


def rank_hypotheses(
    raw: list[dict[str, Any]],
    G: nx.MultiDiGraph,
) -> list[dict[str, Any]]:
    pr = concept_pagerank(G)
    scored = [score_hypothesis(h, G, pr) for h in raw]
    scored.sort(key=lambda x: float(x["scores"]["combined"]), reverse=True)
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
    return scored
