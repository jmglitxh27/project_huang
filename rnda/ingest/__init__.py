"""Stage 1: arXiv ingestion and PDF parsing.

Submodules (``arxiv_client``, ``pdf_parser``, …) are imported from their modules
directly. This package ``__init__`` stays lightweight so optional imports
(e.g. ``rnda.ingest.query_refinement``) do not pull in PyMuPDF or arXiv.
"""
