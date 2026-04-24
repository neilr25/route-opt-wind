"""API helpers extracted from api.py to keep main file under 300 lines."""

from typing import Dict, List, Tuple

import networkx as nx


def _serialize_mesh(G):
    """Return compact JSON-ready dict of mesh nodes + edges."""
    nodes = {}
    for n, d in G.nodes(data=True):
        nodes[n] = {
            "id": n,
            "lat": d["lat"],
            "lon": d["lon"],
            "stage_idx": d.get("stage_idx", None),
            "lane_idx": d.get("lane_idx", None),
            "is_land": bool(d.get("is_land", False)),
        }
    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            "u": u,
            "v": v,
            "u_lat": nodes[u]["lat"],
            "u_lon": nodes[u]["lon"],
            "v_lat": nodes[v]["lat"],
            "v_lon": nodes[v]["lon"],
            "bearing": round(d.get("bearing", 0), 2),
            "distance_nm": round(d.get("distance_nm", 0), 2),
            "crosses_land": bool(d.get("crosses_land", False)),
        })
    return {"nodes": list(nodes.values()), "edges": edges, "edge_count": len(edges)}


def _parse_ll(val: str) -> Tuple[float, float]:
    try:
        parts = val.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        raise ValueError(f"Invalid lat,lon: {val}")


_PORT_MAP = {
    "ROTTERDAM": (51.9244, 4.4777),
    "NEW YORK": (40.7128, -74.0060),
    "SINGAPORE": (1.3521, 103.8198),
    "SHANGHAI": (31.2304, 121.4737),
    "LOS ANGELES": (33.7362, -118.2922),
    "HAMBURG": (53.5488, 9.9872),
    "VALENCIA": (39.4699, -0.3763),
    "TOKYO": (35.6762, 139.6503),
    "BUSAN": (35.1145, 129.0403),
    "ALGECIRAS": (36.1333, -5.4500),
}


def _named_port(name: str) -> Tuple[float, float]:
    key = name.strip().upper()
    if key not in _PORT_MAP:
        raise ValueError(f"Unknown port: {name}. Known: {list(_PORT_MAP.keys())}")
    return _PORT_MAP[key]


__all__ = ["_serialize_mesh", "_parse_ll", "_named_port", "_PORT_MAP"]
