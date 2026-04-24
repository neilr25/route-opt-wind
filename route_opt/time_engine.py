"""Compute voyage ETAs for waypoints and mesh nodes.

Each mesh node maps to an "equivalent base node" — the baseline waypoint at the
same stage.  The node's ETA = start_time + (cumulative NM to equivalent base node
/ ship_speed_kts).
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def _haversine_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(hav))


def baseline_cumdist(baseline: List[Tuple[float, float]]) -> List[float]:
    """Return cumulative NM from start for each baseline waypoint."""
    cd = [0.0]
    for i in range(1, len(baseline)):
        cd.append(cd[-1] + _haversine_nm(baseline[i - 1], baseline[i]))
    return cd


def baseline_etas(
    baseline: List[Tuple[float, float]],
    start: datetime,
    speed_kts: float,
) -> List[datetime]:
    """ETA for each baseline waypoint."""
    cd = baseline_cumdist(baseline)
    return [start + timedelta(hours=d / speed_kts) for d in cd]


def node_etas(
    G,
    baseline: List[Tuple[float, float]],
    stage_indices: List[int],
    start: datetime,
    speed_kts: float,
) -> Dict[Tuple[int, int], datetime]:
    """ETA for every mesh node using its equivalent base waypoint distance."""
    cd = baseline_cumdist(baseline)
    max_stage = len(stage_indices) - 1

    def _eta_for_stage(stage_idx: int) -> datetime:
        idx = min(stage_idx, max_stage)
        base_idx = stage_indices[idx]
        base_idx = min(base_idx, len(cd) - 1)
        return start + timedelta(hours=cd[base_idx] / speed_kts)

    return {
        n: _eta_for_stage(d["stage_idx"])
        for n, d in G.nodes(data=True)
        if "stage_idx" in d
    }
