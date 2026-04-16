from __future__ import annotations

from pathlib import Path

import requests

DEFAULT_GROBID_PATH = "/api/processFulltextDocument"


class GrobidError(Exception):
    """Raised when the GROBID service returns an error or invalid response."""


def _post_pdf(url: str, pdf_path: Path, timeout: tuple[int, int] | int) -> requests.Response:
    path = Path(pdf_path)
    with path.open("rb") as f:
        try:
            return requests.post(
                url,
                files={"input": (path.name, f, "application/pdf")},
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise GrobidError(str(e)) from e


def process_fulltext_document(
    pdf_path: Path,
    base_url: str,
    *,
    timeout: tuple[int, int] | int = (30, 600),
) -> str:
    """
    Send a PDF to GROBID's processFulltextDocument endpoint and return TEI XML.

    ``base_url`` should be like ``http://127.0.0.1:8070`` (no trailing slash).

    See: https://grobid.readthedocs.io/en/latest/GROBID-service/
    """
    url = base_url.rstrip("/") + DEFAULT_GROBID_PATH
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(pdf_path)
    resp = _post_pdf(url, path, timeout)
    if resp.status_code >= 400:
        raise GrobidError(f"GROBID HTTP {resp.status_code}: {resp.text[:500]}")
    text = resp.text
    if not text.strip().startswith("<?xml") and "<TEI" not in text[:2000]:
        raise GrobidError("GROBID response does not look like TEI/XML.")
    return text
