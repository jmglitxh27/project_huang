from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Section keys / titles that often carry limitations, future work, or caveats.
_KEY_HINTS = re.compile(
    r"(limitation|future\s*work|future\s*research|discussion|conclusion|"
    r"threat|weakness|negative|failure|remaining\s*challenges|open\s*(problems|questions)|"
    r"broader\s*impact|appendix)",
    re.IGNORECASE,
)

_PARA_TRIGGER = re.compile(
    r"(?is)\b("
    r"limitation|limitations|future\s+work|future\s+research|open\s+problem|open\s+question|"
    r"we\s+leave\s+to\s+future|directions\s+for\s+future|negative\s+result|"
    r"unfortunately|does\s+not\s+address|failed\s+to|"
    r"room\s+for\s+improvement|remains\s+(?:an\s+)?open|"
    r"importantly,?\s*we\s+did\s+not"
    r")\b",
)

# Short windows around hedging / failure language (negative signals).
_NEG_WINDOW = re.compile(
    r"(?is).{0,40}\b("
    r"however|unfortunately|limitations?|does\s+not|failed|failure|"
    r"not\s+able\s+to|struggles?\s+to|does\s+not\s+generalize|"
    r"negative\s+results?|did\s+not\s+find"
    r")\b.{0,350}",
)


@dataclass(frozen=True)
class Snippet:
    text: str
    arxiv_id: str
    source: str


def _clean(s: str) -> str:
    t = re.sub(r"\s+", " ", s).strip()
    return t


def _chunk_text(s: str, *, max_len: int = 1600) -> str:
    s = _clean(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rsplit(" ", 1)[0] + "…"


def extract_from_sections(record: dict[str, Any], *, min_chars: int = 80) -> list[Snippet]:
    aid = record.get("metadata", {}).get("arxiv_id", "unknown")
    sections = record.get("sections") or {}
    out: list[Snippet] = []
    if isinstance(sections, dict):
        for key, body in sections.items():
            if not isinstance(body, str) or len(body.strip()) < min_chars:
                continue
            lk = str(key).lower()
            first_line = body.split("\n", 1)[0].lower() if body else ""
            if _KEY_HINTS.search(lk) or _KEY_HINTS.search(first_line):
                out.append(
                    Snippet(
                        text=_chunk_text(body),
                        arxiv_id=aid,
                        source=f"section:{key}",
                    )
                )
    grobid = record.get("grobid")
    if isinstance(grobid, dict) and isinstance(grobid.get("sections"), dict):
        for key, body in grobid["sections"].items():
            if not isinstance(body, str) or len(body.strip()) < min_chars:
                continue
            lk = str(key).lower()
            first_line = body.split("\n", 1)[0].lower() if body else ""
            if _KEY_HINTS.search(lk) or _KEY_HINTS.search(first_line):
                out.append(
                    Snippet(
                        text=_chunk_text(body),
                        arxiv_id=aid,
                        source=f"grobid_section:{key}",
                    )
                )
    return out


def extract_from_fulltext(record: dict[str, Any], *, min_chars: int = 80) -> list[Snippet]:
    """Pull paragraph-sized units triggered by limitation / future-work phrasing."""
    aid = record.get("metadata", {}).get("arxiv_id", "unknown")
    full = record.get("full_text") or ""
    if len(full) < min_chars:
        return []
    out: list[Snippet] = []
    paragraphs = re.split(r"\n\s*\n+", full)
    for para in paragraphs:
        p = _clean(para)
        if len(p) < min_chars:
            continue
        if _PARA_TRIGGER.search(p):
            out.append(
                Snippet(
                    text=_chunk_text(p),
                    arxiv_id=aid,
                    source="fulltext:paragraph_trigger",
                )
            )
    for m in _NEG_WINDOW.finditer(full):
        chunk = _clean(m.group(0))
        if len(chunk) >= min_chars:
            out.append(
                Snippet(
                    text=_chunk_text(chunk),
                    arxiv_id=aid,
                    source="fulltext:negation_window",
                )
            )
    return out


def collect_snippets(
    records: list[dict[str, Any]],
    *,
    min_chars: int = 80,
    use_fulltext_triggers: bool = True,
) -> list[Snippet]:
    """Gather snippets from all records; de-duplicate exact duplicates per paper."""
    seen: set[tuple[str, str, str]] = set()
    collected: list[Snippet] = []
    for rec in records:
        chunks = extract_from_sections(rec, min_chars=min_chars)
        if use_fulltext_triggers:
            chunks.extend(extract_from_fulltext(rec, min_chars=min_chars))
        for sn in chunks:
            key = (sn.arxiv_id, sn.source, sn.text[:200])
            if key in seen:
                continue
            seen.add(key)
            collected.append(sn)
    return collected
