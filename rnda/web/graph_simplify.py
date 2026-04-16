from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# vis-network groups for coloring
_GROUP = {"paper": 1, "category": 2, "concept": 3, "unknown": 4}


def load_node_link(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def simplify_for_vis(
    data: dict[str, Any],
    *,
    max_nodes: int = 280,
    max_edges: int = 3500,
) -> dict[str, Any]:
    """
    Convert NetworkX ``node_link_data`` JSON into a smaller graph for vis-network.
    Keeps all paper and category nodes; retains highest-degree concepts if over budget.
    """
    raw_nodes = data.get("nodes") or []
    raw_links = data.get("links") or data.get("edges") or []

    def nid(n: dict[str, Any]) -> Any:
        return n.get("id")

    def kind_of(n: dict[str, Any]) -> str:
        return (n.get("kind") or "unknown") or "unknown"

    papers = [n for n in raw_nodes if kind_of(n) == "paper"]
    cats = [n for n in raw_nodes if kind_of(n) == "category"]
    concepts = [n for n in raw_nodes if kind_of(n) == "concept"]

    budget = max_nodes - len(papers) - len(cats)
    if budget < 50:
        budget = max(30, max_nodes // 3)

    if len(concepts) > budget:
        deg: dict[Any, int] = {nid(n): 0 for n in concepts}
        for e in raw_links:
            s, t = e.get("source"), e.get("target")
            if s in deg:
                deg[s] = deg.get(s, 0) + 1
            if t in deg:
                deg[t] = deg.get(t, 0) + 1
        concepts.sort(key=lambda n: deg.get(nid(n), 0), reverse=True)
        concepts = concepts[:budget]

    keep_ids = {nid(n) for n in papers + cats + concepts}
    kept_nodes = papers + cats + concepts

    out_edges: list[dict[str, Any]] = []
    for e in raw_links:
        s, t = e.get("source"), e.get("target")
        if s in keep_ids and t in keep_ids:
            w = e.get("weight")
            label = e.get("relation", "")
            out_edges.append(
                {
                    "from": s,
                    "to": t,
                    "title": str(label) + (f" ({w:.3f})" if isinstance(w, (int, float)) else ""),
                    "value": float(w) if isinstance(w, (int, float)) else 0.3,
                }
            )

    if len(out_edges) > max_edges:
        out_edges.sort(key=lambda e: -e.get("value", 0))
        out_edges = out_edges[:max_edges]

    vis_nodes: list[dict[str, Any]] = []
    for n in kept_nodes:
        i = nid(n)
        k = kind_of(n)
        label = n.get("label") or n.get("title") or str(i)
        if k == "paper":
            label = (n.get("title") or str(i))[:80]
        elif k == "concept":
            label = str(label)[:60]
        vis_nodes.append(
            {
                "id": i,
                "label": label,
                "group": _GROUP.get(k, 4),
                "title": f"{k}: {n.get('title', n.get('arxiv_id', i))}"[:500],
            }
        )

    return {
        "nodes": vis_nodes,
        "edges": out_edges,
        "multigraph": data.get("multigraph", True),
        "stats": {
            "raw_nodes": len(raw_nodes),
            "raw_edges": len(raw_links),
            "shown_nodes": len(vis_nodes),
            "shown_edges": len(out_edges),
        },
    }


def vis_network_payload_from_run(run_dir: Path, *, max_nodes: int = 280) -> dict[str, Any]:
    gpath = run_dir / "graph.json"
    if not gpath.is_file():
        raise FileNotFoundError("graph.json not found — run the pipeline with graph building first.")
    data = load_node_link(gpath)
    return simplify_for_vis(data, max_nodes=max_nodes)
