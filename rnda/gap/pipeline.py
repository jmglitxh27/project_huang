from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rnda.gap.clustering import (
    cluster_embeddings,
    cluster_tfidf_fallback,
    embed_semantic,
    representative_index,
)
from rnda.gap.extractors import collect_snippets
from rnda.kg.graph_builder import load_parsed_records


def mine_gaps(
    manifest_path: Path,
    out_json: Path,
    *,
    min_chars: int = 80,
    use_fulltext_triggers: bool = True,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    distance_threshold: float = 0.28,
    prefer_semantic: bool = True,
) -> dict[str, Any]:
    """
    Load parsed papers, extract limitation/future-work snippets, cluster, write report.
    """
    records = load_parsed_records(manifest_path)
    snippets = collect_snippets(
        records,
        min_chars=min_chars,
        use_fulltext_triggers=use_fulltext_triggers,
    )
    if not snippets:
        payload = {
            "manifest": str(manifest_path.resolve()),
            "snippet_count": 0,
            "clusters": [],
            "note": "No limitation/future-work snippets found; widen triggers or ingest more papers.",
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    texts = [s.text for s in snippets]
    used_backend = "semantic"

    if prefer_semantic:
        try:
            X = embed_semantic(texts, model_name=embedding_model)
            labels = cluster_embeddings(X, distance_threshold=distance_threshold)
        except Exception:
            labels, X = cluster_tfidf_fallback(texts)
            used_backend = "tfidf_fallback"
    else:
        labels, X = cluster_tfidf_fallback(texts)
        used_backend = "tfidf"

    uniq = sorted(set(labels.tolist()))
    clusters_out: list[dict[str, Any]] = []
    X_np = np.asarray(X, dtype=np.float32)

    for cid in uniq:
        mask = labels == cid
        idx = np.where(mask)[0]
        size = int(mask.sum())
        rep_i = representative_index(mask, X_np)
        rep_snip = snippets[rep_i]
        papers = sorted({snippets[i].arxiv_id for i in idx})
        previews = [snippets[int(i)].text[:400] for i in idx[:8]]
        clusters_out.append(
            {
                "cluster_id": int(cid),
                "size": size,
                "representative_snippet": rep_snip.text[:1200],
                "representative_source": rep_snip.source,
                "arxiv_ids": papers,
                "snippet_previews": previews,
            }
        )

    clusters_out.sort(key=lambda c: -c["size"])

    payload: dict[str, Any] = {
        "manifest": str(manifest_path.resolve()),
        "embedding_backend": used_backend,
        "embedding_model": embedding_model if used_backend == "semantic" else None,
        "distance_threshold": distance_threshold if used_backend == "semantic" else None,
        "snippet_count": len(snippets),
        "cluster_count": len(clusters_out),
        "clusters": clusters_out,
        "snippets": [asdict(s) for s in snippets],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
