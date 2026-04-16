from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from rnda.kg.io import load_graph_from_json
from rnda.novelty.generator import generate_hypotheses
from rnda.novelty.report_stage5 import build_json_export, markdown_report
from rnda.novelty.scoring import rank_hypotheses


def run_stage_4_and_5(
    graph_json: Path | str,
    gap_report_json: Path | str,
    out_dir: Path | str,
    *,
    manifest_path: Path | str | None = None,
    max_hypotheses: int = 10,
    llm_model: str = "gpt-4o-mini",
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Stage 4 — hypothesis generation; Stage 5 — scoring + reports.

    Writes ``novelty_report.json`` and ``novelty_report.md`` under ``out_dir``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    G: nx.MultiDiGraph = load_graph_from_json(graph_json)
    gap_data = json.loads(Path(gap_report_json).read_text(encoding="utf-8"))

    raw, backend_note = generate_hypotheses(
        G,
        gap_data,
        max_hypotheses=max_hypotheses,
        llm_model=llm_model,
        use_llm=use_llm,
    )
    ranked = rank_hypotheses(raw, G)

    query = ""
    if manifest_path:
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        query = str(m.get("query", ""))

    payload: dict[str, Any] = {
        "run_dir": str(out.resolve()),
        "query": query,
        "hypothesis_backend": backend_note,
        "hypotheses_ranked": ranked,
        "gap_summary": {
            "snippet_count": gap_data.get("snippet_count"),
            "cluster_count": gap_data.get("cluster_count"),
        },
    }

    (out / "novelty_report.json").write_text(
        json.dumps(build_json_export(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "novelty_report.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    # Full detail including scores inline
    (out / "novelty_full.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return payload
