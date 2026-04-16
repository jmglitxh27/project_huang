from __future__ import annotations

import json
import os
import re
from typing import Any

from rnda.novelty.signals import graph_context_summary, sparse_concept_pairs


def _heuristic_hypotheses(
    gap_data: dict[str, Any],
    sparse_pairs: list[tuple[str, str, float]],
    graph_summary: dict[str, Any],
    *,
    max_hypotheses: int = 10,
) -> tuple[list[dict[str, Any]], str]:
    """Template hypotheses when no LLM is configured."""
    hyps: list[dict[str, Any]] = []
    clusters = gap_data.get("clusters") or []
    labels = {c["id"]: c.get("label", c["id"]) for c in graph_summary.get("top_concepts_by_degree", [])}

    for cl in clusters[: max_hypotheses // 2 + 3]:
        rep = (cl.get("representative_snippet") or "")[:500]
        cid = cl.get("cluster_id")
        hyps.append(
            {
                "hypothesis": (
                    f"Conduct targeted empirical and theoretical work on the following recurring limitation "
                    f"theme identified across papers: {rep[:400]}"
                ),
                "rationale": "Emerges from automatic clustering of limitation/future-work text (Stage 3).",
                "concepts": [],
                "gap_cluster_ids": [cid],
                "related_arxiv_ids": cl.get("arxiv_ids", [])[:12],
            }
        )

    for i, (a, b, w) in enumerate(sparse_pairs[: max_hypotheses - len(hyps)]):
        la = labels.get(a, a.replace("concept:", "").replace("_", " "))
        lb = labels.get(b, b.replace("concept:", "").replace("_", " "))
        hyps.append(
            {
                "hypothesis": (
                    f"Investigate whether combining or jointly modeling '{la}' and '{lb}' can address "
                    f"under-explored interactions in this literature (empirical co-mention weight ≈ {w:.4f})."
                ),
                "rationale": "Pair ranks low in within-corpus co-occurrence — potentially novel combination in this slice.",
                "concepts": [a, b],
                "gap_cluster_ids": [],
                "related_arxiv_ids": [],
            }
        )

    if not hyps and graph_summary.get("top_concepts_by_degree"):
        top = graph_summary["top_concepts_by_degree"][:3]
        lab = ", ".join(x["label"] for x in top)
        hyps.append(
            {
                "hypothesis": f"Analyze gaps in how {lab} are evaluated and compared across this arXiv slice.",
                "rationale": "Fallback from high-centrality concepts in the induced graph.",
                "concepts": [],
                "gap_cluster_ids": [],
                "related_arxiv_ids": [],
            }
        )

    return hyps[:max_hypotheses], "heuristic"


_LLM_JSON = re.compile(r"\{[\s\S]*\}")


def _generate_llm(
    gap_data: dict[str, Any],
    graph_summary: dict[str, Any],
    sparse_pairs: list[tuple[str, str, float]],
    *,
    model: str,
    max_hypotheses: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install `openai` for LLM hypothesis generation.") from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY for LLM hypothesis generation.")

    client = OpenAI()
    gap_blob = json.dumps(
        [{"id": c.get("cluster_id"), "text": (c.get("representative_snippet") or "")[:800]} for c in gap_data.get("clusters", [])[:15]],
        ensure_ascii=False,
    )
    pair_blob = json.dumps(
        [
            {
                "a": a,
                "b": b,
                "cooc_weight": w,
            }
            for a, b, w in sparse_pairs[:25]
        ],
        ensure_ascii=False,
    )
    ctx = json.dumps(graph_summary, ensure_ascii=False)[:12000]

    prompt = f"""You are a research ideation assistant. Propose specific, falsifiable research hypotheses grounded ONLY in the evidence below (limitation clusters, sparse concept pairs from a citation/co-mention graph of arXiv papers).

Knowledge graph summary:
{ctx}

Limitation clusters (representative snippets):
{gap_blob}

Low co-occurrence concept pairs (possible unexplored combinations in this literature slice):
{pair_blob}

Return a JSON object with a single key "hypotheses" whose value is an array of at most {max_hypotheses} objects, each with:
- "hypothesis": string (one clear falsifiable claim)
- "rationale": string (why it might be novel / evidence-linked)
- "concepts": string[] (optional concept labels or ids if applicable)
- "gap_cluster_ids": number[] (optional cluster ids from limitation clusters that motivate this)
- "related_arxiv_ids": string[] (optional arxiv ids if mentioned in clusters)

Do not invent unrelated domains. Stay anchored to the provided clusters and pairs."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = _LLM_JSON.search(raw)
    if not m:
        raise ValueError("LLM did not return JSON.")
    data = json.loads(m.group(0))
    hyps = data.get("hypotheses") or []
    if not isinstance(hyps, list):
        raise ValueError("Invalid hypotheses array.")
    out: list[dict[str, Any]] = []
    for h in hyps[:max_hypotheses]:
        if not isinstance(h, dict) or "hypothesis" not in h:
            continue
        out.append(
            {
                "hypothesis": str(h.get("hypothesis", "")).strip(),
                "rationale": str(h.get("rationale", "")).strip(),
                "concepts": h.get("concepts") if isinstance(h.get("concepts"), list) else [],
                "gap_cluster_ids": h.get("gap_cluster_ids") if isinstance(h.get("gap_cluster_ids"), list) else [],
                "related_arxiv_ids": h.get("related_arxiv_ids") if isinstance(h.get("related_arxiv_ids"), list) else [],
            }
        )
    return out, f"openai:{model}"


def generate_hypotheses(
    G: Any,
    gap_data: dict[str, Any],
    *,
    max_hypotheses: int = 10,
    llm_model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """
    Stage 4 — produce ranked candidate hypotheses using graph signals + optional LLM.
    Returns (hypotheses, backend_note).
    """
    sp = sparse_concept_pairs(G, limit=50)
    summary = graph_context_summary(G)

    if use_llm:
        try:
            return _generate_llm(gap_data, summary, sp, model=llm_model, max_hypotheses=max_hypotheses)
        except Exception:
            pass

    return _heuristic_hypotheses(gap_data, sp, summary, max_hypotheses=max_hypotheses)
