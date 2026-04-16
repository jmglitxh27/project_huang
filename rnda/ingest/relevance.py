from __future__ import annotations

import numpy as np

from rnda.ingest.models import ArxivMetadata


def build_relevance_intent_text(
    *,
    arxiv_query_body: str,
    natural_language: str | None,
    extra_keywords: list[str] | None,
) -> str:
    """Combine user phrasing and search terms for semantic similarity scoring."""
    parts: list[str] = []
    if natural_language and natural_language.strip():
        parts.append(natural_language.strip())
    if extra_keywords:
        parts.append(" ".join(k.strip() for k in extra_keywords if k and k.strip()))
    body = (arxiv_query_body or "").strip()
    if body:
        parts.append(body)
    return " ".join(parts).strip() or "research paper"


def score_metadata_by_semantic_similarity(
    papers: list[ArxivMetadata],
    intent_text: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[tuple[ArxivMetadata, float]]:
    """
    Embed ``intent_text`` and each paper's title+abstract; return pairs sorted by
    cosine similarity (descending). Uses the same MiniLM model as gap semantic clustering.
    """
    if not papers:
        return []
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    doc_texts = [f"{p.title}. {p.abstract}"[:8000] for p in papers]
    texts = [intent_text[:8000]] + doc_texts
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 32,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    q = emb[0:1]
    docs = emb[1:]
    sims = (docs @ q.T).ravel()
    scored = [(papers[i], float(sims[i])) for i in range(len(papers))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
