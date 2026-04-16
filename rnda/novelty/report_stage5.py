from __future__ import annotations

from typing import Any


def markdown_report(payload: dict[str, Any]) -> str:
    """Human-readable Stage 5 report (Markdown)."""
    lines: list[str] = []
    lines.append("# RNDA novelty report")
    lines.append("")
    lines.append(f"**Run directory:** `{payload.get('run_dir', '')}`")
    lines.append(f"**Hypothesis backend:** {payload.get('hypothesis_backend', '')}")
    lines.append("")
    hyps = payload.get("hypotheses_ranked") or []
    if not hyps:
        lines.append("_No hypotheses produced._")
        return "\n".join(lines)

    for h in hyps:
        r = h.get("rank", "?")
        sc = h.get("scores", {})
        lines.append(f"## {r}. {h.get('hypothesis', '')[:200]}")
        lines.append("")
        lines.append(
            f"**Scores** — combined: {sc.get('combined')} "
            f"(novelty {sc.get('novelty')}, feasibility {sc.get('feasibility')}, impact {sc.get('impact')})"
        )
        lines.append("")
        if h.get("rationale"):
            lines.append(f"**Rationale:** {h['rationale']}")
            lines.append("")
        if h.get("related_arxiv_ids"):
            lines.append(f"**Related arXiv IDs:** {', '.join(h['related_arxiv_ids'][:15])}")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def build_json_export(payload: dict[str, Any]) -> dict[str, Any]:
    """Structured JSON suitable for UI / downstream tools."""
    return {
        "version": 1,
        "run_dir": payload.get("run_dir"),
        "query": payload.get("query"),
        "hypothesis_backend": payload.get("hypothesis_backend"),
        "hypotheses": payload.get("hypotheses_ranked", []),
    }
