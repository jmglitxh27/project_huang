from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph
from sklearn.feature_extraction.text import TfidfVectorizer

_SLUG = re.compile(r"[^a-z0-9]+")


def load_parsed_records(manifest_path: Path | str) -> list[dict[str, Any]]:
    mp = Path(manifest_path)
    if not mp.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {mp.resolve()}\n"
            "Use the real path to manifest.json from a Stage-1 ingest run, e.g.:\n"
            "  data/arxiv_run/manifest.json\n"
            "  data/tiny_corpus/manifest.json"
        )
    data = json.loads(mp.read_text(encoding="utf-8"))
    base = mp.parent.resolve()
    rows: list[dict[str, Any]] = []
    for p in data.get("papers", []):
        pj = Path(p["parsed_json"])
        if not pj.is_absolute():
            pj = (base / pj).resolve()
        rows.append(json.loads(pj.read_text(encoding="utf-8")))
    return rows


def _paper_blob(rec: dict[str, Any]) -> str:
    m = rec["metadata"]
    parts = [m.get("title") or "", m.get("abstract") or ""]
    ft = rec.get("full_text") or ""
    parts.append(ft[:25000])
    return "\n\n".join(parts)


def _concept_node_id(term: str) -> str:
    t = _SLUG.sub("_", term.lower().strip())[:140].strip("_")
    return f"concept:{t}" if t else "concept:empty"


def build_knowledge_graph(
    records: list[dict[str, Any]],
    *,
    top_k_terms: int = 18,
    min_df_ratio: float = 0.05,
    max_features: int = 6000,
) -> nx.MultiDiGraph:
    """
    Build a directed multigraph:
    - ``paper:{arxiv_id}`` --HAS_PRIMARY_CATEGORY--> ``category:{cs.LG}``
    - ``paper:*`` --MENTIONS--> ``concept:*``  (weight = TF-IDF mass on that term)
    - ``concept:a`` --CO_OCCURS--> ``concept:b`` (weight aggregated across papers)
    """
    if not records:
        raise ValueError("No records to graph.")

    G: nx.MultiDiGraph = nx.MultiDiGraph()
    texts = [_paper_blob(r) for r in records]
    min_df = max(1, int(len(texts) * min_df_ratio))

    vectorizer = TfidfVectorizer(
        max_df=0.9,
        min_df=min_df,
        ngram_range=(1, 2),
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()

    for rec in records:
        m = rec["metadata"]
        pid = m["arxiv_id"]
        pn = f"paper:{pid}"
        G.add_node(
            pn,
            kind="paper",
            title=m.get("title", ""),
            arxiv_id=pid,
            published=m.get("published"),
        )
        cat = (m.get("primary_category") or "").strip() or "unknown"
        cn = f"category:{cat}"
        if not G.has_node(cn):
            G.add_node(cn, kind="category", name=cat)
        G.add_edge(pn, cn, relation="HAS_PRIMARY_CATEGORY")

    cooc_acc: defaultdict[tuple[str, str], float] = defaultdict(float)

    for row_idx, rec in enumerate(records):
        row = X.getrow(row_idx)
        nz = list(zip(row.indices, row.data))
        if not nz:
            continue
        nz.sort(key=lambda t: -t[1])
        chosen = nz[:top_k_terms]
        pid = rec["metadata"]["arxiv_id"]
        pn = f"paper:{pid}"
        doc_concepts: list[tuple[str, float]] = []
        for j, w in chosen:
            term = str(names[j])
            if len(term) < 3:
                continue
            cid = _concept_node_id(term)
            if not G.has_node(cid):
                G.add_node(cid, kind="concept", label=term)
            G.add_edge(pn, cid, relation="MENTIONS", weight=float(w))
            doc_concepts.append((cid, float(w)))
        for i in range(len(doc_concepts)):
            for j in range(i + 1, len(doc_concepts)):
                a, wa = doc_concepts[i]
                b, wb = doc_concepts[j]
                x, y = sorted([a, b])
                cooc_acc[(x, y)] += wa * wb

    for (a, b), w in cooc_acc.items():
        if w <= 0:
            continue
        G.add_edge(a, b, relation="CO_OCCURS", weight=float(w))

    return G


def graph_to_json(G: nx.MultiDiGraph) -> dict[str, Any]:
    return json_graph.node_link_data(G)


def build_graph_from_manifest(
    manifest_path: Path,
    out_json: Path,
    *,
    top_k_terms: int = 18,
) -> nx.MultiDiGraph:
    records = load_parsed_records(manifest_path)
    G = build_knowledge_graph(records, top_k_terms=top_k_terms)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = graph_to_json(G)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return G
