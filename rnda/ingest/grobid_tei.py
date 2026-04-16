from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any
from xml.etree.ElementTree import Element

TEI_NS = "http://www.tei-c.org/ns/1.0"


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _all_text(elem: Element) -> str:
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if _local_name(child.tag) in ("ref", "figure", "formula"):
            t = _all_text(child)
            if t:
                parts.append(t)
        else:
            parts.append(_all_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _normalize_ws(s: str) -> str:
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _slug_key(head: str, used: dict[str, int]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_")[:80] or "section"
    n = used.get(base, 0)
    used[base] = n + 1
    return f"{base}_{n}" if n else base


def _collect_div_sections(body: Element) -> dict[str, str]:
    """Collect section text from body divs (prefer top-level sections with a head)."""
    sections: dict[str, str] = {}
    used: dict[str, int] = {}
    candidates: list[Element] = []
    for child in list(body):
        if _local_name(child.tag) == "div":
            candidates.append(child)
    if not candidates:
        candidates = [d for d in body.findall(".//{%s}div" % TEI_NS) if d.find("{%s}head" % TEI_NS) is not None]
    for div in candidates:
        head_el = div.find("{%s}head" % TEI_NS)
        head = _normalize_ws(_all_text(head_el)) if head_el is not None else ""
        if not head:
            continue
        key = _slug_key(head, used)
        buf: list[str] = []
        for p in div.findall(".//{%s}p" % TEI_NS):
            t = _normalize_ws(_all_text(p))
            if t:
                buf.append(t)
        if not buf:
            raw = _normalize_ws(_all_text(div))
            if head:
                raw = raw.replace(head, "", 1).strip()
            if raw:
                buf.append(raw)
        if buf:
            sections[key] = f"{head}\n\n" + "\n\n".join(buf)
    return sections


def _authors_from_header(root: Element) -> list[str]:
    names: list[str] = []
    for pers in root.findall(".//{%s}sourceDesc//{%s}author" % (TEI_NS, TEI_NS)):
        name_el = pers.find(".//{%s}persName" % TEI_NS)
        if name_el is not None:
            forename = name_el.find("{%s}forename" % TEI_NS)
            surname = name_el.find("{%s}surname" % TEI_NS)
            bits = []
            if forename is not None and forename.text:
                bits.append(forename.text.strip())
            if surname is not None and surname.text:
                bits.append(surname.text.strip())
            if bits:
                names.append(" ".join(bits))
                continue
        t = _normalize_ws(_all_text(pers))
        if t:
            names.append(t)
    return names


def _title_from_header(root: Element) -> str | None:
    for xp in (
        ".//{%s}titleStmt/{%s}title[@type='main']" % (TEI_NS, TEI_NS),
        ".//{%s}titleStmt/{%s}title" % (TEI_NS, TEI_NS),
        ".//{%s}analytic//{%s}title[@level='a']" % (TEI_NS, TEI_NS),
    ):
        el = root.find(xp)
        if el is not None:
            t = _normalize_ws(_all_text(el))
            if t:
                return t
    return None


def _abstract_from_header(root: Element) -> str | None:
    abs_el = root.find(".//{%s}profileDesc/{%s}abstract" % (TEI_NS, TEI_NS))
    if abs_el is None:
        return None
    t = _normalize_ws(_all_text(abs_el))
    return t or None


def _references(root: Element) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for bib in root.findall(".//{%s}back//{%s}biblStruct" % (TEI_NS, TEI_NS)):
        title = None
        analytic = bib.find("{%s}analytic" % TEI_NS)
        if analytic is not None:
            te = analytic.find(".//{%s}title[@level='a']" % TEI_NS)
            if te is None:
                te = analytic.find(".//{%s}title" % TEI_NS)
            if te is not None:
                title = _normalize_ws(_all_text(te))
        if not title:
            monogr = bib.find("{%s}monogr" % TEI_NS)
            if monogr is not None:
                te = monogr.find(".//{%s}title" % TEI_NS)
                if te is not None:
                    title = _normalize_ws(_all_text(te))
        authors: list[str] = []
        for ath in bib.findall(".//{%s}author" % TEI_NS):
            t = _normalize_ws(_all_text(ath))
            if t:
                authors.append(t)
        year = None
        for d in bib.findall(".//{%s}date" % TEI_NS):
            if d.text:
                year = d.text.strip()[:4]
                break
        refs.append({"title": title, "authors": authors, "year": year})
    return refs


def parse_tei_xml(tei_xml: str) -> dict[str, Any]:
    """
    Extract title, authors, abstract, body sections, and bibliography from GROBID TEI.
    Keys are stable for JSON serialization.
    """
    root = ET.fromstring(tei_xml)
    text_el = root.find(".//{%s}text" % TEI_NS)
    body = None if text_el is None else text_el.find("{%s}body" % TEI_NS)
    sections: dict[str, str] = {}
    if body is not None:
        sections = _collect_div_sections(body)
    if not sections and body is not None:
        raw = _normalize_ws(_all_text(body))
        if raw:
            sections = {"body": raw}

    title = _title_from_header(root)
    authors = _authors_from_header(root)
    abstract = _abstract_from_header(root)
    references = _references(root)

    full_pieces: list[str] = []
    if abstract:
        full_pieces.append("Abstract\n\n" + abstract)
    for k, v in sections.items():
        full_pieces.append(v)
    full_text = _normalize_ws("\n\n\n".join(full_pieces)) if full_pieces else ""

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "sections": sections,
        "references": references,
        "full_text": full_text,
    }
