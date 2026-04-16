from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


def load_graph_from_json(path: Path | str) -> nx.MultiDiGraph:
    """Load a ``MultiDiGraph`` previously saved with :func:`rnda.kg.graph_builder.graph_to_json`."""
    mp = Path(path)
    if not mp.is_file():
        raise FileNotFoundError(f"Graph JSON not found: {mp.resolve()}")
    data: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(data)
