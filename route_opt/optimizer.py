"""State-space A* optimizer with weather-aware edge costs."""

import heapq
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from route_opt.cost_engine import edge_cost, fuel_no_wind, fuel_with_wind, fuel_without_wingsail
from route_opt.weather_client import wind_at_points


def _haversine_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Heuristic distance in nautical miles."""
    R = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(hav))


State = Tuple[Tuple[int, int], Optional[Tuple[int, int]]]  # (node, prev_node)


def _bearing_seg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Bearing from a to b in degrees (0-360)."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _route_distance_nm(path: List[Tuple[float, float]]) -> float:
    """Total route distance in nautical miles."""
    total = 0.0
    for i in range(len(path) - 1):
        total += _haversine_nm(path[i], path[i + 1])
    return total


def optimise(
    G: nx.DiGraph,
    baseline: List[Tuple[float, float]],
    start_ll: Tuple[float, float],
    goal_ll: Tuple[float, float],
    date: str,
    ship_speed_kts: float,
    max_detour_pct: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], float, float, float, float, float]:
    """
    A* search through corridor graph.
    Returns:
        - optimised path as list of (lat, lon)
        - cost_standard_no_wind (tonnes — baseline route, no wind assist)
        - cost_standard_with_wind (tonnes — baseline route, with wind)
        - cost_optimised (tonnes — optimised route, with wind)
        - baseline_distance_nm (nautical miles)
        - optimised_distance_nm (nautical miles)
    
    If max_detour_pct is set, the optimised route distance is capped:
    routes longer than baseline * (1 + max_detour_pct/100) are rejected and
    the baseline is returned instead.
    """
    # ---- Find closest nodes to start/goal ----
    nodes = list(G.nodes(data=True))
    start_node = min(nodes, key=lambda n: _haversine_nm(start_ll, (n[1]["lat"], n[1]["lon"])))[0]
    goal_node = min(nodes, key=lambda n: _haversine_nm(goal_ll, (n[1]["lat"], n[1]["lon"])))[0]

    # ---- Pre-fetch weather for all graph nodes ----
    points = [(d["lat"], d["lon"]) for _, d in nodes]
    winds = wind_at_points(points, date)
    wind_map: Dict[Tuple[int, int], Tuple[float, float]] = {n[0]: w for n, w in zip(nodes, winds)}

    # ---- Pre-compute edge metadata dicts for fast lookup ----
    edge_bearings: Dict[Tuple[int, int], float] = {}
    edge_distances_nm: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        edge_bearings[(u, v)] = data["bearing"]
        edge_distances_nm[(u, v)] = data["distance_nm"]

    # ---- Pre-fetch weather for baseline waypoints ----
    baseline_winds = wind_at_points(baseline, date)

    # ---- Compute baseline distance ----
    baseline_dist_nm = _route_distance_nm(baseline)

    # ---- Standard route costs (baseline waypoints) ----
    cost_std_no_wind = 0.0
    cost_std_with_wind = 0.0
    for i in range(len(baseline) - 1):
        a = baseline[i]
        b = baseline[i + 1]
        dist = _haversine_nm(a, b)
        hours = dist / ship_speed_kts
        bearing = _bearing_seg(a, b)
        ws_kmh, wd = baseline_winds[i + 1]
        ws_ms = ws_kmh  # data is already m/s
        f_no = fuel_without_wingsail(ws_ms, wd, bearing, hours)
        cost_std_no_wind += f_no
        f_wi = fuel_with_wind(ws_ms, wd, bearing, hours)
        cost_std_with_wind += min(f_no, f_wi)

    # ---- A* heuristic cache ----
    goal_coords = (G.nodes[goal_node]["lat"], G.nodes[goal_node]["lon"])
    h_cache: Dict[Tuple[int, int], float] = {}
    for n, d in nodes:
        dist = _haversine_nm((d["lat"], d["lon"]), goal_coords)
        hours = dist / ship_speed_kts
        h_cache[n] = fuel_no_wind(hours)

    # ---- A* init ----
    open_set: List[Tuple[float, State]] = []
    initial_state: State = (start_node, None)
    heapq.heappush(open_set, (h_cache[start_node], initial_state))
    g_score: Dict[State, float] = {initial_state: 0.0}
    came_from: Dict[State, State] = {}
    visited: set = set()

    while open_set:
        _, state = heapq.heappop(open_set)
        current, prev = state

        if current == goal_node:
            # Reconstruct path
            path = []
            tmp_state = state
            while True:
                path.append(tmp_state[0])
                if tmp_state[1] is None:
                    break
                prev_state = came_from.get(tmp_state)
                if prev_state is None:
                    break
                tmp_state = prev_state
            path.reverse()
            path_ll = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in path]
            cost_opt = g_score[state]

            # Compute optimised route distance
            opt_dist_nm = _route_distance_nm(path_ll)

            # Apply detour cap if requested
            if max_detour_pct is not None and opt_dist_nm > baseline_dist_nm * (1 + max_detour_pct / 100):
                # Optimised route is too long — fall back to baseline
                return baseline, cost_std_no_wind, cost_std_with_wind, cost_std_with_wind, baseline_dist_nm, baseline_dist_nm

            return path_ll, cost_std_no_wind, cost_std_with_wind, cost_opt, baseline_dist_nm, opt_dist_nm

        if state in visited:
            continue
        visited.add(state)

        current_bearing = None
        if prev is not None and G.has_edge(prev, current):
            current_bearing = G.edges[prev, current]["bearing"]

        for neighbor in G.successors(current):
            edge_data = G.edges[current, neighbor]

            # HARD BLOCK: Skip land nodes (except center lane)
            if G.nodes[neighbor].get("is_land", False) and G.nodes[neighbor].get("lane_idx", 0) != 0:
                continue

            # HARD BLOCK: Skip edges that cross land
            if edge_data.get("crosses_land", False):
                continue

            b = edge_data["bearing"]
            dist = edge_data["distance_nm"]
            ws, wd = wind_map.get(neighbor, (0.0, 0.0))
            ws_ms = ws  # data is already m/s
            hours = dist / ship_speed_kts
            c = edge_cost(ws_ms, wd, b, current_bearing, hours)
            next_state: State = (neighbor, current)
            tentative = g_score[state] + c
            if tentative < g_score.get(next_state, float("inf")):
                came_from[next_state] = state
                g_score[next_state] = tentative
                f = tentative + h_cache.get(neighbor, 0)
                heapq.heappush(open_set, (f, next_state))

    # ---- Fallback: if A* exhausts, return baseline as optimised route ----
    cost_baseline_opt = sum(
        edge_cost(
            baseline_winds[i + 1][0],
            baseline_winds[i + 1][1],
            _bearing_seg(baseline[i], baseline[i + 1]),
            None,
            _haversine_nm(baseline[i], baseline[i + 1]) / ship_speed_kts,
        )
        for i in range(len(baseline) - 1)
    )
    return baseline, cost_std_no_wind, cost_std_with_wind, cost_baseline_opt, baseline_dist_nm, baseline_dist_nm
