"""State-space A* optimizer with hourly weather-aware edge costs."""

import heapq
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx

from route_opt.cost_engine import edge_cost, fuel_no_wind, fuel_with_wind, fuel_without_wingsail
from route_opt.weather_client import wind_at_points
from route_opt.hourly_weather import wind_at_points_hourly
from route_opt.time_engine import baseline_etas, node_etas


def _haversine_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(hav))


def _bearing_seg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _route_distance_nm(path: List[Tuple[float, float]]) -> float:
    return sum(_haversine_nm(path[i], path[i + 1]) for i in range(len(path) - 1))


State = Tuple[Tuple[int, int], Optional[Tuple[int, int]]]


def _baseline_costs(baseline, voyage_dt, speed_kts):
    """Return baseline costs using hourly weather."""
    etas = baseline_etas(baseline, voyage_dt, speed_kts)
    baseline_winds = wind_at_points_hourly(baseline, etas)
    cost_std_no_wind = 0.0
    cost_std_with_wind = 0.0
    for i in range(len(baseline) - 1):
        a = baseline[i]
        b = baseline[i + 1]
        dist = _haversine_nm(a, b)
        hours = dist / speed_kts
        bearing = _bearing_seg(a, b)
        ws_ms, wd = baseline_winds[i]
        f_no = fuel_without_wingsail(ws_ms, wd, bearing, hours)
        cost_std_no_wind += f_no
        f_wi = fuel_with_wind(ws_ms, wd, bearing, hours)
        cost_std_with_wind += min(f_no, f_wi)
    return cost_std_no_wind, cost_std_with_wind


def _fetch_hourly_winds(G, nodes, node_etas_map):
    """Fetch hourly wind for all nodes, falling back to daily if hourly unavailable."""
    points = [(d["lat"], d["lon"]) for _, d in nodes]
    etas = [node_etas_map.get(n[0], datetime.min) for n in nodes]
    if datetime.min in etas:
        # Fallback: use daily weather if ETAs can't be built
        return wind_at_points(points, etas[0].strftime("%Y-%m-%d"))
    return wind_at_points_hourly(points, etas)


def optimise(
    G: nx.DiGraph,
    baseline: List[Tuple[float, float]],
    start_ll: Tuple[float, float],
    goal_ll: Tuple[float, float],
    date: str,
    ship_speed_kts: float,
    voyage_datetime: Optional[datetime] = None,
    max_detour_pct: Optional[float] = None,
) -> Tuple[List[Tuple[float, float]], float, float, float, float, float]:
    """
    A* search through corridor graph.
    Returns:
        - optimised path as list of (lat, lon)
        - cost_standard_no_wind (tonnes — baseline route, no wind assist)
        - cost_standard_with_wind (tonnes — baseline route, with wind)
        - cost_optimised (tonnes — optimised route, with wind)
        - baseline_distance_nm
        - optimised_distance_nm
    """
    # Parse voyage start datetime
    if voyage_datetime is None:
        voyage_datetime = datetime.strptime(date, "%Y-%m-%d")

    nodes = list(G.nodes(data=True))
    start_node = min(nodes, key=lambda n: _haversine_nm(start_ll, (n[1]["lat"], n[1]["lon"])))[0]
    goal_node = min(nodes, key=lambda n: _haversine_nm(goal_ll, (n[1]["lat"], n[1]["lon"])))[0]

    # ---- Compute node ETAs using time_engine ----
    stage_indices = G.graph.get("stage_indices")
    if stage_indices:
        node_etas_map = node_etas(G, baseline, stage_indices, voyage_datetime, ship_speed_kts)
    else:
        # Fallback to daily if no stage indices available
        node_etas_map = {}

    # ---- Pre-fetch hourly weather for all graph nodes ----
    winds = _fetch_hourly_winds(G, nodes, node_etas_map)
    wind_map: Dict[Tuple[int, int], Tuple[float, float]] = {n[0]: w for n, w in zip(nodes, winds)}

    # ---- Pre-compute edge metadata dicts for fast lookup ----
    edge_bearings: Dict[Tuple[int, int], float] = {}
    edge_distances_nm: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        edge_bearings[(u, v)] = data["bearing"]
        edge_distances_nm[(u, v)] = data["distance_nm"]

    # ---- Compute baseline distance ----
    baseline_dist_nm = _route_distance_nm(baseline)

    # ---- Standard route costs with hourly weather ----
    cost_std_no_wind, cost_std_with_wind = _baseline_costs(baseline, voyage_datetime, ship_speed_kts)

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
            opt_dist_nm = _route_distance_nm(path_ll)

            if max_detour_pct is not None and opt_dist_nm > baseline_dist_nm * (1 + max_detour_pct / 100):
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

    # Fallback: return baseline as optimised route
    cost_baseline_opt = sum(
        edge_cost(*wind_map.get(baseline[i + 1], (0.0, 0.0)),
                  _bearing_seg(baseline[i], baseline[i + 1]), None,
                  _haversine_nm(baseline[i], baseline[i + 1]) / ship_speed_kts)
        for i in range(len(baseline) - 1)
    )
    return baseline, cost_std_no_wind, cost_std_with_wind, cost_baseline_opt, baseline_dist_nm, baseline_dist_nm
