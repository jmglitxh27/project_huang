from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _fallback_markdown(manifest: dict, gap: dict | None, novelty: dict | None) -> str:
    lines: list[str] = []
    lines.append("# Literature corpus report")
    lines.append("")
    lines.append(f"**Query:** `{manifest.get('query', '')}`")
    lines.append(f"**Papers:** {manifest.get('count', 0)}")
    lines.append("")
    lines.append("## Included papers")
    lines.append("")
    for p in manifest.get("papers") or []:
        lines.append(f"- **{p.get('title', '')[:200]}** — `{p.get('arxiv_id', '')}`")
    lines.append("")
    if gap and gap.get("clusters"):
        lines.append("## Gap / limitation themes (auto-clustered)")
        lines.append("")
        for c in gap["clusters"][:12]:
            lines.append(f"### Cluster {c.get('cluster_id')}")
            lines.append("")
            lines.append((c.get("representative_snippet") or "")[:800])
            lines.append("")
    if novelty and novelty.get("hypotheses_ranked"):
        lines.append("## Ranked novelty hypotheses")
        lines.append("")
        for h in novelty["hypotheses_ranked"][:15]:
            sc = h.get("scores", {})
            lines.append(f"1. {h.get('hypothesis', '')[:500]}")
            lines.append(f"   - *Scores:* combined {sc.get('combined')} (novelty {sc.get('novelty')}, feasibility {sc.get('feasibility')}, impact {sc.get('impact')})")
            lines.append("")
    lines.append("---")
    lines.append("*Generated without LLM (set OPENAI_API_KEY for a narrative synthesis).*")
    return "\n".join(lines)


_MD_BLOCK = re.compile(r"```(?:markdown)?\s*([\s\S]*?)```", re.I)


def _llm_markdown(
    manifest: dict,
    gap: dict | None,
    novelty: dict | None,
    *,
    model: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package required for LLM report") from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    papers_blob = json.dumps(
        [
            {"arxiv_id": p.get("arxiv_id"), "title": p.get("title")}
            for p in (manifest.get("papers") or [])[:80]
        ],
        ensure_ascii=False,
    )
    gap_blob = json.dumps(
        [{"id": c.get("cluster_id"), "snippet": (c.get("representative_snippet") or "")[:600]} for c in (gap or {}).get("clusters") or []][:20],
        ensure_ascii=False,
    )
    hyps = (novelty or {}).get("hypotheses_ranked") or (novelty or {}).get("hypotheses") or []
    hyp_blob = json.dumps(
        [
            {
                "rank": h.get("rank"),
                "hypothesis": h.get("hypothesis"),
                "rationale": h.get("rationale"),
                "scores": h.get("scores"),
            }
            for h in hyps[:25]
        ],
        ensure_ascii=False,
    )

    prompt = f"""You are writing a concise technical report for researchers who sampled an arXiv sub-corpus and ran automated gap + novelty analysis.

Context JSON fragments:

Papers in corpus:
{papers_blob}

Limitation / gap clusters (representative snippets):
{gap_blob}

Ranked novelty hypotheses (with scores):
{hyp_blob}

Write a single **Markdown** document with these sections (use ## headings):
1. Executive summary (5–8 sentences): what was collected and what the analysis suggests.
2. Corpus description: scope, how many papers, implied sub-field from titles.
3. Recurring limitations & open directions: synthesize clusters (not a bullet list of raw text only — interpret).
4. Novelty hypotheses: discuss top hypotheses and what would falsify them; connect to corpus themes.
5. Methods note: what is automated vs. what requires human expert follow-up.
6. References: numbered list of arXiv ids from the corpus cited in the narrative.

Use calm, precise academic tone. Do not invent papers not in the corpus list."""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write clear Markdown with ## headings only, no YAML front matter."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.45,
        max_tokens=4096,
    )
    raw = (resp.choices[0].message.content or "").strip()
    m = _MD_BLOCK.search(raw)
    if m:
        return m.group(1).strip()
    return raw


def markdown_to_pdf_reportlab(md_text: str, out_pdf: Path) -> None:
    """Render Markdown-ish text to PDF using ReportLab (best-effort, UTF-8)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        rightMargin=inch * 0.75,
        leftMargin=inch * 0.75,
        topMargin=inch * 0.75,
        bottomMargin=inch * 0.75,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=13, spaceAfter=8)
    body = ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=13)

    story: list[Any] = []
    blocks = re.split(r"\n{2,}", md_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            story.append(Paragraph(escape(block[2:].strip()), h1))
        elif block.startswith("## "):
            story.append(Paragraph(escape(block[3:].strip()), h2))
        elif block.startswith("### "):
            story.append(Paragraph(escape(block[4:].strip()), styles["Heading3"]))
        elif block.startswith("- ") or block.startswith("* "):
            for line in block.split("\n"):
                line = line.strip()
                if line.startswith(("- ", "* ")):
                    story.append(Paragraph("• " + escape(line[2:].strip()), body))
                elif line:
                    story.append(Paragraph(escape(line), body))
        else:
            # preserve single newlines as <br/>
            safe = escape(block).replace("\n", "<br/>")
            story.append(Paragraph(safe, body))
        story.append(Spacer(1, 6))

    doc.build(story)


def generate_literature_report(
    run_dir: Path | str,
    *,
    model: str = "gpt-4o-mini",
    prefer_llm: bool = True,
) -> dict[str, Any]:
    """
    Write ``literature_report.md`` and ``literature_report.pdf`` under ``run_dir``.
    Falls back to heuristic Markdown if LLM is unavailable.
    """
    run_dir = Path(run_dir)
    man_path = run_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(man_path)

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    gap: dict[str, Any] | None = None
    novelty_path = run_dir / "novelty_full.json"
    nov_path2 = run_dir / "novelty_report.json"
    if (run_dir / "gap_report.json").is_file():
        gap = json.loads((run_dir / "gap_report.json").read_text(encoding="utf-8"))
    novelty: dict[str, Any] | None = None
    if novelty_path.is_file():
        novelty = json.loads(novelty_path.read_text(encoding="utf-8"))
    elif nov_path2.is_file():
        novelty = json.loads(nov_path2.read_text(encoding="utf-8"))
        if novelty.get("hypotheses") and not novelty.get("hypotheses_ranked"):
            novelty["hypotheses_ranked"] = novelty["hypotheses"]

    md_text: str
    backend: str
    if prefer_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            md_text = _llm_markdown(manifest, gap, novelty, model=model)
            backend = f"openai:{model}"
        except Exception:
            md_text = _fallback_markdown(manifest, gap, novelty)
            backend = "fallback_template"
    else:
        md_text = _fallback_markdown(manifest, gap, novelty)
        backend = "fallback_template"

    md_out = run_dir / "literature_report.md"
    pdf_out = run_dir / "literature_report.pdf"
    md_out.write_text(md_text, encoding="utf-8")
    try:
        markdown_to_pdf_reportlab(md_text, pdf_out)
    except Exception as e:
        if pdf_out.is_file():
            pdf_out.unlink(missing_ok=True)
        (run_dir / "literature_report_pdf_error.txt").write_text(str(e), encoding="utf-8")

    return {
        "run_dir": str(run_dir.resolve()),
        "markdown_path": str(md_out),
        "pdf_path": str(pdf_out),
        "backend": backend,
    }
