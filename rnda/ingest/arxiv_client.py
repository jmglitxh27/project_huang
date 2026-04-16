from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Iterator

ProgressCallback = Callable[[str, int, int, str], None]

import arxiv

from rnda.ingest.models import ArxivMetadata

_ARXIV_DELAY = 4.0
_MAX_RETRIES = 6


def _arxiv_client() -> arxiv.Client:
    return arxiv.Client(
        page_size=100,
        delay_seconds=_ARXIV_DELAY,
        num_retries=5,
    )


def _collect_results(search: arxiv.Search) -> list[arxiv.Result]:
    """Run a query with extra backoff on 429 / 503 (arXiv rate limits)."""
    client = _arxiv_client()
    last: arxiv.HTTPError | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return list(client.results(search))
        except arxiv.HTTPError as e:
            last = e
            if getattr(e, "status", None) not in (403, 429, 500, 502, 503):
                raise
            wait = min(120.0, 8.0 * (2**attempt))
            time.sleep(wait)
    assert last is not None
    raise last

_ARXIV_SHORT_ID = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(?P<id>[\w._/-]+?)(?:\.pdf)?(?:$|[?#])",
    re.IGNORECASE,
)


def normalize_arxiv_id(raw: str) -> str:
    """Accept URLs or bare ids (e.g. 2401.12345v1) and return a short id without leading 'arxiv:'."""
    s = raw.strip()
    m = _ARXIV_SHORT_ID.search(s)
    if m:
        return m.group("id").rstrip("/")
    if s.lower().startswith("arxiv:"):
        return s.split(":", 1)[1].strip()
    return s


def result_to_metadata(r: arxiv.Result) -> ArxivMetadata:
    short = r.get_short_id()
    authors = [a.name for a in r.authors]
    entry_url = f"https://arxiv.org/abs/{short}"
    return ArxivMetadata(
        arxiv_id=short,
        title=(r.title or "").replace("\n", " ").strip(),
        authors=authors,
        abstract=(r.summary or "").replace("\n", " ").strip(),
        primary_category=r.primary_category or "",
        categories=list(r.categories or []),
        published=r.published.isoformat() if r.published else "",
        updated=r.updated.isoformat() if r.updated else "",
        pdf_url=r.pdf_url,
        entry_url=entry_url,
        comment=r.comment,
        journal_ref=r.journal_ref,
        doi=r.doi,
    )


def search_arxiv(
    query: str,
    *,
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> list[ArxivMetadata]:
    """Query the arXiv API and return structured metadata (no PDF download)."""
    arxiv_search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return [result_to_metadata(r) for r in _collect_results(arxiv_search)]


def iter_search_results(
    query: str,
    *,
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> Iterator[arxiv.Result]:
    arxiv_search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    yield from _collect_results(arxiv_search)


def download_pdf(
    result: arxiv.Result,
    dest_dir: Path,
    *,
    filename: str | None = None,
) -> Path:
    """Download the paper PDF; returns path to the saved file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = filename or f"{result.get_short_id().replace('/', '_')}.pdf"
    returned = result.download_pdf(dirpath=str(dest_dir), filename=stem)
    return Path(returned)


def collect_search_results_list(
    query: str,
    *,
    max_results: int = 50,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
) -> list[arxiv.Result]:
    """Return up to ``max_results`` arXiv ``Result`` objects (no PDF download)."""
    arxiv_search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    return _collect_results(arxiv_search)


def download_pdfs_for_results(
    results: list[arxiv.Result],
    pdf_dir: Path,
    *,
    delay_between_downloads: float = 3.0,
    progress: ProgressCallback | None = None,
) -> list[tuple[ArxivMetadata, Path]]:
    """Download PDFs for pre-selected ``Result`` objects (e.g. after relevance ranking)."""
    out: list[tuple[ArxivMetadata, Path]] = []
    n = len(results)
    last_err: arxiv.HTTPError | None = None
    for attempt in range(_MAX_RETRIES):
        out.clear()
        try:
            if progress:
                progress("download", 0, n, "")
            for idx, result in enumerate(results, start=1):
                aid = result.get_short_id()
                meta = result_to_metadata(result)
                if progress:
                    progress("download", idx, n, aid)
                path = download_pdf(result, pdf_dir)
                out.append((meta, path))
                time.sleep(delay_between_downloads)
            return out
        except arxiv.HTTPError as e:
            last_err = e
            if getattr(e, "status", None) not in (403, 429, 500, 502, 503):
                raise
            wait = min(120.0, 8.0 * (2**attempt))
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def fetch_papers(
    query: str,
    pdf_dir: Path,
    *,
    max_results: int = 10,
    delay_between_downloads: float = 3.0,
    sort_by: arxiv.SortCriterion = arxiv.SortCriterion.SubmittedDate,
    sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
    progress: ProgressCallback | None = None,
) -> list[tuple[ArxivMetadata, Path]]:
    """
    Search arXiv, download each PDF into pdf_dir, return (metadata, path) pairs.
    Respects arXiv rate limits via Client delay + pauses between downloads.

    ``progress`` is optional ``callback(stage: str, index: int, total: int, arxiv_id: str)``
    where stage is ``arxiv_api`` (waiting on first API response), ``download`` (PDF saved).
    """
    out: list[tuple[ArxivMetadata, Path]] = []
    last_err: arxiv.HTTPError | None = None
    for attempt in range(_MAX_RETRIES):
        out.clear()
        arxiv_search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        try:
            client = _arxiv_client()
            it = client.results(arxiv_search)
            if progress:
                progress("arxiv_api", 0, max_results, "")
            idx = 0
            while True:
                try:
                    result = next(it)
                except StopIteration:
                    break
                idx += 1
                if idx > max_results:
                    break
                aid = result.get_short_id()
                meta = result_to_metadata(result)
                if progress:
                    progress("download", idx, max_results, aid)
                path = download_pdf(result, pdf_dir)
                out.append((meta, path))
                time.sleep(delay_between_downloads)
            return out
        except arxiv.HTTPError as e:
            last_err = e
            if getattr(e, "status", None) not in (403, 429, 500, 502, 503):
                raise
            wait = min(120.0, 8.0 * (2**attempt))
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def fetch_by_id(
    arxiv_id: str,
    pdf_dir: Path | None = None,
) -> tuple[ArxivMetadata, Path | None]:
    """Resolve a single arXiv id (or URL) to metadata, optionally downloading the PDF."""
    aid = normalize_arxiv_id(arxiv_id)
    arxiv_by_id = arxiv.Search(id_list=[aid])
    results = _collect_results(arxiv_by_id)
    if not results:
        raise ValueError(f"No arXiv entry found for id: {arxiv_id!r}")
    r = results[0]
    meta = result_to_metadata(r)
    if pdf_dir is None:
        return meta, None
    path = download_pdf(r, pdf_dir)
    return meta, path
