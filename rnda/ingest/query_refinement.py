from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


@dataclass
class RefinedSearch:
    """LLM- or heuristic-produced search hints for Stage 1."""

    arxiv_topic: str
    keywords: list[str]
    rationale: str


_LLM_JSON = re.compile(r"\{[\s\S]*\}")


def ensure_search_topic(topic: str, natural_language: str | None) -> str:
    """If ``topic`` is empty, derive a minimal arXiv query fragment from natural language."""
    t = (topic or "").strip()
    if t:
        return t
    nl = (natural_language or "").strip()
    if not nl:
        return ""
    return _heuristic_refine("", nl).arxiv_topic


def _heuristic_refine(topic: str, natural_language: str | None) -> RefinedSearch:
    base = (topic or "").strip()
    nl = (natural_language or "").strip()
    if not base and nl:
        words = re.findall(r"[A-Za-z][A-Za-z0-9+.-]{1,}", nl.lower())
        stop = frozenset(
            "the a an and or for to of in on at with from by as is are was were be been being "
            "this that these those we our it its they their not no yes which what how when into "
            "about papers paper study studies research work recent using use model models".split()
        )
        kws = [w for w in words if w not in stop and len(w) > 2][:12]
        arxiv_topic = " ".join(kws) if kws else nl[:200]
        return RefinedSearch(
            arxiv_topic=arxiv_topic,
            keywords=kws,
            rationale="Heuristic tokenization from natural-language intent (no API key or LLM disabled).",
        )
    raw = f"{base} {nl}".strip()
    tokens = re.findall(r"[\w.+-]+", raw.lower())
    stop = frozenset(
        "the and or for of to in on at with from by a an is are was were be been".split()
    )
    kws = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) < 3 or t in stop:
            continue
        if t not in seen:
            seen.add(t)
            kws.append(t)
        if len(kws) >= 16:
            break
    return RefinedSearch(
        arxiv_topic=base or " ".join(kws[:8]),
        keywords=kws or (base.split() if base else []),
        rationale="Heuristic keyword extraction (LLM refinement disabled or unavailable).",
    )


def refine_search_with_llm(
    *,
    topic: str,
    natural_language: str | None = None,
    model: str = "gpt-4o-mini",
) -> RefinedSearch:
    """
    Use OpenAI to propose an arXiv ``ti``/``abs``-friendly keyword string plus extra
    relevance keywords. Falls back to :func:`_heuristic_refine` if no key or failure.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return _heuristic_refine(topic, natural_language)

    try:
        from openai import OpenAI
    except ImportError:
        return _heuristic_refine(topic, natural_language)

    nl = (natural_language or "").strip()
    to = (topic or "").strip()
    prompt = (
        "You help researchers query arXiv. Given a short keyword topic and/or a natural-language "
        "research goal, output a compact arXiv query fragment (boolean-ish: AND/OR terms, quoted "
        "phrases when needed) for the `query` API field, plus a list of 6–20 keywords/phrases "
        "for relevance checking against titles and abstracts.\n"
        f"Topic / keywords: {to or '(none)'}\n"
        f"Natural-language intent: {nl or '(none)'}\n"
        "Respond with JSON only: "
        '{"arxiv_topic": "...", "keywords": ["...", ...], "rationale": "one sentence"}'
    )

    client = OpenAI()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output only valid JSON for literature search."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = _LLM_JSON.search(raw)
        if not m:
            return _heuristic_refine(topic, natural_language)
        data = json.loads(m.group(0))
        at = str(data.get("arxiv_topic") or to or nl).strip()
        kws = data.get("keywords") or []
        if not isinstance(kws, list):
            kws = []
        kws = [str(x).strip() for x in kws if str(x).strip()][:24]
        rationale = str(data.get("rationale") or "LLM refinement.").strip()
        if not at:
            return _heuristic_refine(topic, natural_language)
        return RefinedSearch(arxiv_topic=at, keywords=kws, rationale=rationale)
    except Exception:
        return _heuristic_refine(topic, natural_language)
