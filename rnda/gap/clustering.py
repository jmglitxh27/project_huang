from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def embed_semantic(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Dense embeddings (sentence-transformers / Hugging Face)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def embed_tfidf(texts: list[str], *, max_features: int = 4096) -> np.ndarray:
    """Fallback bag-of-ngrams vectors."""
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )
    return vec.fit_transform(texts).astype(np.float32).toarray()


def cluster_embeddings(
    X: np.ndarray,
    *,
    distance_threshold: float = 0.28,
    linkage: str = "average",
) -> np.ndarray:
    """
    Hierarchical clustering in embedding space.
    On L2-normalized dense vectors, Euclidean distance relates to cosine similarity.
    """
    n = X.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int)
    if n == 2:
        return np.array([0, 1], dtype=int)
    Xn = _l2_normalize(X) if X.shape[1] > 0 else X
    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="euclidean",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    return cl.fit_predict(Xn)


def cluster_tfidf_fallback(
    texts: list[str],
    *,
    n_clusters: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """TF-IDF + disjoint clusters from Agglomerative on cosine distance matrix."""
    vec = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
        sublinear_tf=True,
    )
    M = vec.fit_transform(texts).astype(np.float32)
    S = cosine_similarity(M)
    dist = 1.0 - np.clip(S, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    n = dist.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=int), embed_tfidf(texts)
    if n_clusters is None:
        n_clusters = max(2, min(12, int(np.sqrt(n))))
    cl = AgglomerativeClustering(
        n_clusters=min(n_clusters, n),
        metric="precomputed",
        linkage="average",
    )
    labels = cl.fit_predict(dist)
    return labels, embed_tfidf(texts)


def representative_index(cluster_mask: np.ndarray, X: np.ndarray) -> int:
    """Index of snippet closest to the cluster centroid (cosine on normalized X)."""
    idx = np.where(cluster_mask)[0]
    if len(idx) == 0:
        return 0
    if len(idx) == 1:
        return int(idx[0])
    sub = X[idx]
    sub = _l2_normalize(sub)
    centroid = sub.mean(axis=0, keepdims=True)
    centroid = _l2_normalize(centroid)
    sim = (sub @ centroid.T).ravel()
    best = int(np.argmax(sim))
    return int(idx[best])
