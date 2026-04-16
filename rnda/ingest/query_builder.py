from __future__ import annotations


def build_arxiv_query(
    topic: str,
    *,
    submitted_from_yyyymmdd: str | None = None,
    submitted_to_yyyymmdd: str | None = None,
    categories: list[str] | None = None,
) -> str:
    """
    Compose an arXiv API query string.
    Date bounds use arXiv `submittedDate` field per https://info.arxiv.org/help/api/user-manual.html
    """
    parts: list[str] = [f"({topic.strip()})"]
    if submitted_from_yyyymmdd or submitted_to_yyyymmdd:
        start = submitted_from_yyyymmdd or "199101010000"
        end = submitted_to_yyyymmdd or "300012312359"
        parts.append(f"AND submittedDate:[{start} TO {end}]")
    if categories:
        cat_q = " OR ".join(f"cat:{c}" for c in categories)
        parts.append(f"AND ({cat_q})")
    return " ".join(parts)

