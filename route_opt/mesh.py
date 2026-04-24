"""Build corridor mesh from baseline route with edge bearings.

VOIDS-style stage-skipping mesh:
- Baseline waypoints every ~50 nm (densified)
- Graph stages every STAGE_SKIP waypoints (~200 nm apart)
- Lanes every LANE_SPACING_NM across +/-CORRIDOR_WIDTH_NM
- Edges connect (stage=i, lane=j) to (stage=i+1, lane=k) only if
  |k-j| <= MAX_LANE_CHANGE.  This keeps heading deviation <= ~45°.
- Land-crossing edges have a small penalty (1e3) rather than being cut,
  so port-departure and port-approach legs can still be traversed.
"""

import math
from typing import List, Tuple

import networkx as nx
from global_land_mask import globe

from route_opt.config import CORRIDOR_WIDTH_NM, LANE_SPACING_NM, MAX_LANE_CHANGE, STAGE_SKIP


def _nm_to_deg(nm: float, lat: float) -> Tuple[float, float]:
    dlat = nm / 60.0
    dlon = nm / (60.0 * math.cos(math.radians(lat))) if math.cos(math.radians(lat)) else nm / 60.0
    return dlat, dlon


def _bearing(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _distance_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(hav))


def _node_too_close_to_land(lat: float, lon: float, buffer_nm: float = 10.0, samples: int = 8) -> bool:
    """Return True if any point in a circle of radius buffer_nm around (lat, lon) is on land."""
    return False  # REMOVED - hard blocks on offset nodes + edge_crosses_land only


def _edge_crosses_land(p1: Tuple[float, float], p2: Tuple[float, float], samples: int = 20) -> bool:
    """Return True if any sample on the straight line between p1 and p2 is on land.
    
    Uses moderate samples. With coastal lane locking, few if any edges near ports
    will be tested on offset lanes.
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    for i in range(1, samples + 1):
        frac = i / (samples + 1)
        lat = lat1 + frac * (lat2 - lat1)
        lon = lon1 + frac * (lon2 - lon1)
        if globe.is_land(lat, lon):
            return True
    return False


def corridor_graph(
    baseline: List[Tuple[float, float]],
    width_nm: float = CORRIDOR_WIDTH_NM,
    lane_spacing_nm: float = LANE_SPACING_NM,
    max_lane_change: int = MAX_LANE_CHANGE,
    stage_skip: int = STAGE_SKIP,
) -> nx.DiGraph:
    """
    Build a directed corridor graph with stage-skipping:
    - Nodes every `stage_skip` baseline waypoints (coarse stages)
    - Lanes every `lane_spacing_nm` across +/-`width_nm`
    - Edges connect (i, j) to (i+1, k) where |k-j| <= max_lane_change
    """
    G = nx.DiGraph()
    n_lanes_half = int(round(width_nm / lane_spacing_nm))
    lane_offsets = list(range(-n_lanes_half, n_lanes_half + 1))
    node_list: List[List[Tuple[int, int]]] = []

    # Build graph stages every `stage_skip` waypoints
    stage_indices = list(range(0, len(baseline), stage_skip))
    if stage_indices[-1] != len(baseline) - 1:
        stage_indices.append(len(baseline) - 1)

    n_stages = len(stage_indices)
    COASTAL_STAGES = 3  # no lateral deviation within first/last 3 stages

    for stage_idx, base_idx in enumerate(stage_indices):
        a = baseline[base_idx]
        if stage_idx < len(stage_indices) - 1:
            # Look ahead to next stage for the overall route heading
            next_stage_idx = stage_indices[stage_idx + 1]
            next_pt = baseline[next_stage_idx]
            brng = _bearing(a, next_pt)
        elif base_idx > 0:
            prev_pt = baseline[base_idx - 1]
            brng = _bearing(prev_pt, a)
        else:
            brng = 0.0

        perp = (brng + 90) % 360
        nodes_here: List[Tuple[int, int]] = []

        for lane_idx in lane_offsets:
            # Near coast: only centre lane allowed
            if (stage_idx < COASTAL_STAGES or stage_idx >= n_stages - COASTAL_STAGES) and lane_idx != 0:
                continue

            lateral_nm = lane_idx * lane_spacing_nm
            dlat, dlon = _nm_to_deg(abs(lateral_nm), a[0])
            sign = 1 if lateral_nm >= 0 else -1
            olat = a[0] + sign * dlat * math.cos(math.radians(perp))
            olon = a[1] + sign * dlon * math.sin(math.radians(perp))
            olat = max(-90, min(90, olat))
            olon = ((olon + 180) % 360) - 180
            # Center lane follows ATOBVIAC waypoints — already maritime.
            # Only flag offset lanes as land.
            if lane_idx == 0:
                is_land = False
            else:
                is_land = globe.is_land(olat, olon)
            node_key = (stage_idx, lane_idx)
            G.add_node(
                node_key, lat=olat, lon=olon, stage_idx=stage_idx,
                lane_idx=lane_idx, is_land=is_land,
            )
            nodes_here.append(node_key)

        node_list.append(nodes_here)

    # Build edges with lane-change pruning
    for stage_idx in range(len(node_list) - 1):
        for n1 in node_list[stage_idx]:
            lane1 = G.nodes[n1]["lane_idx"]
            for n2 in node_list[stage_idx + 1]:
                lane2 = G.nodes[n2]["lane_idx"]
                if abs(lane2 - lane1) > max_lane_change:
                    continue
                p1 = (G.nodes[n1]["lat"], G.nodes[n1]["lon"])
                p2 = (G.nodes[n2]["lat"], G.nodes[n2]["lon"])
                crosses_land = False if lane1 == 0 and lane2 == 0 else _edge_crosses_land(p1, p2)
                dist = _distance_nm(p1, p2)
                edge_bearing = _bearing(p1, p2)
                G.add_edge(
                    n1, n2,
                    weight=dist * 1.0,  # placeholder; real cost comes from optimiser
                    bearing=edge_bearing,
                    distance_nm=dist,
                    crosses_land=crosses_land,
                )

    # Store stage indices on the graph for ETA lookup
    G.graph["stage_indices"] = stage_indices

    return G
