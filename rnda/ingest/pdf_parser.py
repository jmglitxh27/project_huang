from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

import xml.etree.ElementTree as ET

from rnda.ingest.grobid_client import GrobidError, process_fulltext_document
from rnda.ingest.grobid_tei import parse_tei_xml
from rnda.ingest.models import ArxivMetadata, ParsedPaper

# Common CS/ML paper section headers (first match wins in document order).
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^\s*abstract\s*$", re.IGNORECASE | re.MULTILINE)),
    ("introduction", re.compile(r"^\s*1\.?\s+introduction\s*$", re.IGNORECASE | re.MULTILINE)),
    ("related_work", re.compile(r"^\s*(?:2\.?\s+)?related\s+work\s*$", re.IGNORECASE | re.MULTILINE)),
    ("background", re.compile(r"^\s*(?:\d+\.?\s+)?background\s*$", re.IGNORECASE | re.MULTILINE)),
    ("method", re.compile(r"^\s*(?:3\.?\s+)?method(?:ology)?\s*$", re.IGNORECASE | re.MULTILINE)),
    ("methods", re.compile(r"^\s*(?:\d+\.?\s+)?methods\s*$", re.IGNORECASE | re.MULTILINE)),
    ("experiments", re.compile(r"^\s*(?:\d+\.?\s+)?experiments?\s*$", re.IGNORECASE | re.MULTILINE)),
    ("results", re.compile(r"^\s*(?:\d+\.?\s+)?results?\s*$", re.IGNORECASE | re.MULTILINE)),
    ("discussion", re.compile(r"^\s*discussion\s*$", re.IGNORECASE | re.MULTILINE)),
    ("limitations", re.compile(r"^\s*limitations\s*$", re.IGNORECASE | re.MULTILINE)),
    ("conclusion", re.compile(r"^\s*(?:\d+\.?\s+)?conclusions?\s*$", re.IGNORECASE | re.MULTILINE)),
    ("references", re.compile(r"^\s*references\s*$", re.IGNORECASE | re.MULTILINE)),
]

# Fallback: split body into coarse blocks if no numbered sections.
_FALLBACK_SPLIT = re.compile(
    r"\n\s*(?=(?:Abstract|Introduction|Methods?|Experiments?|Results?|Discussion|Conclusion|References)\b)",
    re.IGNORECASE,
)


def _clean_line_breaks(text: str) -> str:
    # Join hyphenated line wraps; normalize spaces.
    t = text.replace("-\n", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_full_text(pdf_path: Path | str) -> tuple[str, int]:
    doc = fitz.open(pdf_path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text(sort=True))
        return "\n\n".join(parts), doc.page_count
    finally:
        doc.close()


def extract_full_text_from_bytes(data: bytes) -> tuple[str, int]:
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text(sort=True))
        return "\n\n".join(parts), doc.page_count
    finally:
        doc.close()


def split_sections(full_text: str) -> tuple[dict[str, str], list[str]]:
    """
    Heuristic sectioning for camera-ready PDFs. Quality varies with publisher layout.
    Returns (sections dict, notes).
    """
    notes: list[str] = []
    text = _clean_line_breaks(full_text)
    if len(text) < 200:
        notes.append("Very little text extracted; PDF may be scan-only or two-column layout degraded.")
        return {}, notes

    spans: list[tuple[int, int, str]] = []
    for name, pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), name))
    spans.sort(key=lambda x: x[0])

    if not spans:
        chunks = [c.strip() for c in _FALLBACK_SPLIT.split(text) if c.strip()]
        if len(chunks) > 1:
            notes.append("Used fallback keyword splits; verify sections manually.")
            return {"body": "\n\n".join(chunks[:20])}, notes
        notes.append("Could not detect section headers; stored as single body block.")
        return {"body": text}, notes

    # Dedupe: keep first occurrence per section name for stable boundaries.
    seen: set[str] = set()
    ordered: list[tuple[int, int, str]] = []
    for start, end, name in spans:
        if name in seen:
            continue
        seen.add(name)
        ordered.append((start, end, name))
    ordered.sort(key=lambda x: x[0])

    sections: dict[str, str] = {}
    for i, (start, _end, name) in enumerate(ordered):
        nxt = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        chunk = text[start:nxt].strip()
        if chunk:
            if name in sections:
                sections[name] = sections[name] + "\n\n" + chunk
            else:
                sections[name] = chunk

    notes.append("Sections detected via line-header heuristics (not GROBID).")
    return sections, notes


def parse_pdf_bytes(
    data: bytes,
    metadata: ArxivMetadata,
    *,
    pdf_path: str | None = None,
    grobid_base_url: str | None = None,
    grobid_timeout: int = 120,
) -> ParsedPaper:
    import tempfile

    pymupdf_full_text, page_count = extract_full_text_from_bytes(data)
    pymupdf_sections, notes = split_sections(pymupdf_full_text)
    sections = pymupdf_sections
    full_text = pymupdf_full_text
    grobid_payload: dict | None = None

    if grobid_base_url:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tpath = Path(tmp.name)
        try:
            return parse_pdf_file(
                tpath,
                metadata,
                grobid_base_url=grobid_base_url,
                grobid_timeout=grobid_timeout,
            )
        finally:
            try:
                tpath.unlink(missing_ok=True)
            except OSError:
                pass

    return ParsedPaper(
        metadata=metadata,
        pdf_path=pdf_path,
        full_text=full_text,
        sections=sections,
        page_count=page_count,
        extraction_notes=notes,
        grobid=grobid_payload,
        pymupdf_sections=pymupdf_sections,
        pymupdf_full_text=pymupdf_full_text,
    )


def parse_pdf_file(
    pdf_path: Path,
    metadata: ArxivMetadata,
    *,
    grobid_base_url: str | None = None,
    grobid_timeout: int = 120,
) -> ParsedPaper:
    pymupdf_full_text, page_count = extract_full_text(pdf_path)
    pymupdf_sections, notes = split_sections(pymupdf_full_text)
    grobid_payload: dict | None = None
    sections = pymupdf_sections
    full_text = pymupdf_full_text

    if grobid_base_url:
        try:
            tei = process_fulltext_document(
                pdf_path,
                grobid_base_url,
                timeout=(30, grobid_timeout),
            )
            grobid_payload = parse_tei_xml(tei)
            if grobid_payload.get("sections"):
                sections = grobid_payload["sections"]
            if grobid_payload.get("full_text"):
                full_text = grobid_payload["full_text"]
            elif sections:
                full_text = "\n\n\n".join(sections.values())
            notes.append("GROBID full-text OK; sections preferred from TEI when present.")
        except (GrobidError, OSError, ValueError, ET.ParseError) as e:
            notes.append(f"GROBID unavailable or failed ({e}); using PyMuPDF sections only.")
            grobid_payload = None
        else:
            if grobid_payload:
                notes = [n for n in notes if "line-header heuristics (not GROBID)" not in n]

    return ParsedPaper(
        metadata=metadata,
        pdf_path=str(pdf_path.resolve()),
        full_text=full_text,
        sections=sections,
        page_count=page_count,
        extraction_notes=notes,
        grobid=grobid_payload,
        pymupdf_sections=pymupdf_sections,
        pymupdf_full_text=pymupdf_full_text,
    )
