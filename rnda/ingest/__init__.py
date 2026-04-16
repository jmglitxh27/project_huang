"""Stage 1: arXiv ingestion and PDF parsing."""

from rnda.ingest.arxiv_client import fetch_papers, search_arxiv
from rnda.ingest.grobid_client import GrobidError, process_fulltext_document
from rnda.ingest.grobid_tei import parse_tei_xml
from rnda.ingest.models import ArxivMetadata, ParsedPaper
from rnda.ingest.pdf_parser import parse_pdf_bytes, parse_pdf_file

__all__ = [
    "ArxivMetadata",
    "ParsedPaper",
    "GrobidError",
    "fetch_papers",
    "parse_pdf_bytes",
    "parse_pdf_file",
    "parse_tei_xml",
    "process_fulltext_document",
    "search_arxiv",
]
