from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ArxivMetadata:
    """Structured fields from the arXiv API (no PDF required)."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    published: str
    updated: str
    pdf_url: str
    entry_url: str
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedPaper:
    """arXiv metadata plus PDF text; PyMuPDF is baseline, optional GROBID enriches structure."""

    metadata: ArxivMetadata
    pdf_path: str | None
    full_text: str
    sections: dict[str, str]
    page_count: int
    extraction_notes: list[str] = field(default_factory=list)
    grobid: dict[str, Any] | None = None
    pymupdf_sections: dict[str, str] | None = None
    pymupdf_full_text: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "metadata": self.metadata.to_json_dict(),
            "pdf_path": self.pdf_path,
            "full_text": self.full_text,
            "sections": self.sections,
            "page_count": self.page_count,
            "extraction_notes": self.extraction_notes,
        }
        if self.grobid is not None:
            out["grobid"] = self.grobid
        if self.pymupdf_sections is not None:
            out["pymupdf_sections"] = self.pymupdf_sections
        if self.pymupdf_full_text is not None:
            out["pymupdf_full_text"] = self.pymupdf_full_text
        return out
