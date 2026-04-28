"""State-space A* optimizer with hourly weather-aware edge costs."""

import heapq
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx

from route_opt.cost_engine import edge_cost, fuel_no_wind, fuel_without_wingsail, fuel_with_wind, _twa, _sog
from route_opt.weather_client import wind_at_points
from route_opt.hourly_weather import wind_at_points_hourly
from route_opt.time_engine import baseline_etas, node_etas
from route_opt.current_client import current_at_points as _current_at_points
from route_opt.unified_weather import weather_and_current_at_points as _weather_and_current


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


def _densify_route(route, max_gap_nm=22.0):
    """Insert intermediate points along each segment longer than max_gap_nm.
    
    Returns densified route with ~25nm spacing for accurate weather sampling.
    Uses 22nm threshold so that haversine nonlinearity keeps segments under ~25nm.
    """
    if len(route) < 2:
        return route
    out = [route[0]]
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        dist = _haversine_nm(a, b)
        if dist <= max_gap_nm:
            out.append(b)
            continue
        steps = max(1, int(dist / max_gap_nm))
        for s in range(1, steps + 1):
            frac = s / (steps + 1)
            out.append((
                a[0] + frac * (b[0] - a[0]),
                a[1] + frac * (b[1] - a[1]),
            ))
        out.append(b)
    return out


def _baseline_costs(baseline, etas, speed_kts, date_str, use_hourly, baseline_currents=None):
    """Return baseline costs using hourly weather and optional currents.
    
    Densifies baseline to ~22nm segments so each sub-segment samples
    the correct weather at the right time — matching the optimiser's resolution.
    Uses 22nm threshold to keep all haversine segments under ~25nm.
    """
    # Densify to ~22nm for accurate weather/current sampling per segment
    densified = _densify_route(baseline, max_gap_nm=22.0)
    
    # Re-compute ETAs for the densified baseline
    from route_opt.time_engine import baseline_etas as _baseline_etas
    densified_etas = _baseline_etas(densified, etas[0], speed_kts)
    
    # Interpolate currents from original baseline waypoints to densified points
    if baseline_currents:
        densified_currents = _interpolate_currents(baseline, baseline_currents, densified)
    else:
        densified_currents = None
    
    # Fetch wind for densified points (for accurate cost)
    if use_hourly:
        densified_winds = wind_at_points_hourly(densified, densified_etas)
        # Also fetch wind at original baseline points (for edge_meta display)
        baseline_winds = wind_at_points_hourly(baseline, etas)
    else:
        densified_winds = wind_at_points(densified, date_str)
        baseline_winds = wind_at_points(baseline, date_str)
    
    cost_std_no_wind = 0.0
    cost_std_with_wind = 0.0
    for i in range(len(densified) - 1):
        a = densified[i]
        b = densified[i + 1]
        dist = _haversine_nm(a, b)
        bearing = _bearing_seg(a, b)
        ws_ms, wd = densified_winds[i]
        cu, cv = densified_currents[i] if densified_currents else (0.0, 0.0)
        # SOG-adjusted hours (same physics as optimiser)
        stw_mps = speed_kts * 0.514444
        sog_mps = _sog(stw_mps, cu, cv, bearing)
        sog_kts = sog_mps / 0.514444 if sog_mps > 0 else speed_kts
        hours = dist / sog_kts
        f_no = fuel_without_wingsail(ws_ms, wd, bearing, hours)
        cost_std_no_wind += f_no
        f_wi = fuel_with_wind(ws_ms, wd, bearing, hours)
        cost_std_with_wind += min(f_no, f_wi)
    return cost_std_no_wind, cost_std_with_wind, baseline_winds


def _interpolate_currents(original_route, original_currents, densified_route):
    """Interpolate current (u, v) from original waypoints to densified waypoints.
    
    For each densified point, find which original segment it lies on
    and linearly interpolate current vectors based on position along that segment.
    """
    densified_currents = []
    
    for pt in densified_route:
        # Find which original segment this point is on (nearest segment)
        best_dist = float('inf')
        best_idx = 0
        
        for j in range(len(original_route) - 1):
            a = original_route[j]
            b = original_route[j + 1]
            # Project point onto segment a-b
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            px, py = pt[0], pt[1]
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                t = 0.0
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
            proj_x = ax + t * dx
            proj_y = ay + t * dy
            d = _haversine_nm(pt, (proj_x, proj_y))
            if d < best_dist:
                best_dist = d
                best_idx = j
                best_t = max(0.0, min(1.0, ((px - a[0]) * (b[0] - a[0]) + (py - a[1]) * (b[1] - a[1])) / max((b[0] - a[0])**2 + (b[1] - a[1])**2, 1e-12)))
        
        # Linearly interpolate current between original waypoints
        u_a, v_a = original_currents[best_idx]
        u_b, v_b = original_currents[min(best_idx + 1, len(original_currents) - 1)]
        u = u_a + best_t * (u_b - u_a)
        v = v_a + best_t * (v_b - v_a)
        densified_currents.append((u, v))
    
    return densified_currents


def _fetch_hourly_winds(G, nodes, node_etas_map, date_str, use_hourly):
    """Fetch hourly wind for all nodes, falling back to daily if hourly unavailable."""
    points = [(d["lat"], d["lon"]) for _, d in nodes]
    if not use_hourly:
        return wind_at_points(points, date_str)
    etas = [node_etas_map.get(n[0], datetime.min) for n in nodes]
    if datetime.min in etas:
        return wind_at_points(points, date_str)
    return wind_at_points_hourly(points, etas)


def _compute_edge_meta_for_graph(G, path_nodes, wind_map, current_map, node_etas_map, voyage_dt, speed_kts):
    meta = []
    for i in range(len(path_nodes) - 1):
        n_u = path_nodes[i]
        n_v = path_nodes[i + 1]
        u_data = G.nodes[n_u]
        v_data = G.nodes[n_v]
        e_data = G.edges[n_u, n_v]
        dist = e_data["distance_nm"]
        bearing = e_data["bearing"]
        hours = dist / speed_kts
        eta = node_etas_map.get(n_u, voyage_dt).isoformat()
        ws, wd = wind_map.get(n_v, (0.0, 0.0))
        current_u, current_v = current_map.get(n_v, (0.0, 0.0))
        current_speed = math.sqrt(current_u ** 2 + current_v ** 2)
        prev_b = G.edges[path_nodes[i - 1], n_u]["bearing"] if i > 0 else None
        fuel_t = edge_cost(ws, wd, bearing, prev_b, dist, speed_kts, current_u, current_v)
        twa = _twa(wd, bearing)
        meta.append({
            "from_lat": u_data["lat"],
            "from_lon": u_data["lon"],
            "to_lat": v_data["lat"],
            "to_lon": v_data["lon"],
            "bearing": round(bearing, 2),
            "distance_nm": round(dist, 2),
            "eta": eta,
            "wind_speed_ms": round(ws, 2),
            "wind_direction_deg": round(wd, 1),
            "twa_deg": round(twa, 1),
            "current_u_ms": round(current_u, 3),
            "current_v_ms": round(current_v, 3),
            "current_speed_ms": round(current_speed, 3),
            "fuel_tonnes": round(fuel_t, 3),
        })
    return meta


def _compute_edge_meta_for_waypoints(waypoints, winds, currents, etas, speed_kts):
    meta = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        dist = _haversine_nm(a, b)
        bearing = _bearing_seg(a, b)
        hours = dist / speed_kts
        eta = etas[i].isoformat()
        ws, wd = winds[i]
        cu, cv = currents[i] if i < len(currents) else (0.0, 0.0)
        current_speed = math.sqrt(cu ** 2 + cv ** 2)
        fuel_t = edge_cost(ws, wd, bearing, None, dist, speed_kts, cu, cv)
        twa = _twa(wd, bearing)
        meta.append({
            "from_lat": a[0],
            "from_lon": a[1],
            "to_lat": b[0],
            "to_lon": b[1],
            "bearing": round(bearing, 2),
            "distance_nm": round(dist, 2),
            "eta": eta,
            "wind_speed_ms": round(ws, 2),
            "wind_direction_deg": round(wd, 1),
            "twa_deg": round(twa, 1),
            "current_u_ms": round(cu, 3),
            "current_v_ms": round(cv, 3),
            "current_speed_ms": round(current_speed, 3),
            "fuel_tonnes": round(fuel_t, 3),
        })
    return meta


def optimise(
    G: nx.DiGraph,
    baseline: List[Tuple[float, float]],
    start_ll: Tuple[float, float],
    goal_ll: Tuple[float, float],
    date: str,
    ship_speed_kts: float,
    voyage_datetime: Optional[datetime] = None,
    max_detour_pct: Optional[float] = None,
    use_hourly: bool = True,
    use_currents: bool = True,
) -> Tuple[List[Tuple[float, float]], float, float, float, float, float, List[Dict], List[Dict]]:
    """
    A* search through corridor graph.
    Returns:
        - optimised path as list of (lat, lon)
        - cost_standard_no_wind (tonnes -- baseline route, no wind assist)
        - cost_standard_with_wind (tonnes -- baseline route, with wind)
        - cost_optimised (tonnes -- optimised route, with wind)
        - baseline_distance_nm
        - optimised_distance_nm
        - edge_metadata list of dicts for each traversed edge
    """
    # Parse voyage start datetime
    if voyage_datetime is None:
        voyage_datetime = datetime.strptime(date, "%Y-%m-%d")
    date_str = voyage_datetime.strftime("%Y-%m-%d")

    nodes = list(G.nodes(data=True))
    start_node = min(nodes, key=lambda n: _haversine_nm(start_ll, (n[1]["lat"], n[1]["lon"])))[0]
    goal_node = min(nodes, key=lambda n: _haversine_nm(goal_ll, (n[1]["lat"], n[1]["lon"])))[0]

    # ---- Compute node ETAs using time_engine ----
    stage_indices = G.graph.get("stage_indices")
    if stage_indices:
        node_etas_map = node_etas(G, baseline, stage_indices, voyage_datetime, ship_speed_kts)
    else:
        node_etas_map = {}

    # ---- Compute baseline ETAs and costs ----
    baseline_etas_list = baseline_etas(baseline, voyage_datetime, ship_speed_kts)

    # ---- Pre-fetch ocean currents (prefer unified weather for speed) ----
    if use_currents:
        # Use unified weather (compact numpy cache, O(1) lookup, handles wind+current together)
        baseline_pts = list(baseline)
        baseline_etas_uniform = list(baseline_etas_list)
        node_pts = [(d["lat"], d["lon"]) for _, d in nodes]
        node_etas_list = [node_etas_map.get(n[0], voyage_datetime) for n in nodes]
        
        # Fetch baseline wind+current via unified
        baseline_unified = _weather_and_current(baseline_pts, baseline_etas_uniform)
        baseline_currents = [
            (r[2], r[3]) if r is not None else (0.0, 0.0)
            for r in baseline_unified
        ]
        
        # Fetch node wind+current via unified
        node_unified = _weather_and_current(node_pts, node_etas_list)
        wind_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
        current_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for n, r in zip(nodes, node_unified):
            if r is not None:
                wind_map[n[0]] = (r[0], r[1])
                current_map[n[0]] = (r[2], r[3])
            else:
                wind_map[n[0]] = (0.0, 0.0)
                current_map[n[0]] = (0.0, 0.0)
        
        # Wind already fetched from unified — skip separate wind fetch
        _wind_already_fetched = True
    else:
        baseline_currents = None
        current_map: Dict[Tuple[int, int], Tuple[float, float]] = {n[0]: (0.0, 0.0) for n in nodes}
        _wind_already_fetched = False

    cost_std_no_wind, cost_std_with_wind, baseline_winds = _baseline_costs(
        baseline, baseline_etas_list, ship_speed_kts, date_str, use_hourly, baseline_currents
    )

    # ---- Pre-fetch weather for all graph nodes ----
    if not _wind_already_fetched:
        winds = _fetch_hourly_winds(G, nodes, node_etas_map, date_str, use_hourly)
        wind_map: Dict[Tuple[int, int], Tuple[float, float]] = {n[0]: w for n, w in zip(nodes, winds)}

    # ---- Pre-compute edge metadata dicts for fast lookup ----
    edge_bearings: Dict[Tuple[int, int], float] = {}
    edge_distances_nm: Dict[Tuple[int, int], float] = {}
    for u, v, data in G.edges(data=True):
        edge_bearings[(u, v)] = data["bearing"]
        edge_distances_nm[(u, v)] = data["distance_nm"]

    # ---- Compute baseline distance ----
    baseline_dist_nm = _route_distance_nm(baseline)

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
            path_nodes = path
            path_ll = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in path]
            cost_opt = g_score[state]
            opt_dist_nm = _route_distance_nm(path_ll)

            if max_detour_pct is not None and opt_dist_nm > baseline_dist_nm * (1 + max_detour_pct / 100):
                edge_meta = _compute_edge_meta_for_waypoints(
                    baseline, baseline_winds, baseline_currents or [(0.0, 0.0)] * len(baseline), baseline_etas_list, ship_speed_kts
                )
                baseline_edge_meta = edge_meta
                return baseline, cost_std_no_wind, cost_std_with_wind, cost_std_with_wind, baseline_dist_nm, baseline_dist_nm, edge_meta, baseline_edge_meta

            edge_meta = _compute_edge_meta_for_graph(
                G, path_nodes, wind_map, current_map, node_etas_map, voyage_datetime, ship_speed_kts
            )
            baseline_edge_meta = _compute_edge_meta_for_waypoints(
                baseline, baseline_winds, baseline_currents or [(0.0, 0.0)] * len(baseline), baseline_etas_list, ship_speed_kts
            )
            return path_ll, cost_std_no_wind, cost_std_with_wind, cost_opt, baseline_dist_nm, opt_dist_nm, edge_meta, baseline_edge_meta

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
            current_u, current_v = current_map.get(neighbor, (0.0, 0.0))
            c = edge_cost(ws_ms, wd, b, None, dist, ship_speed_kts,
                           current_u, current_v)
            next_state: State = (neighbor, current)
            tentative = g_score[state] + c
            if tentative < g_score.get(next_state, float("inf")):
                came_from[next_state] = state
                g_score[next_state] = tentative
                f = tentative + h_cache.get(neighbor, 0)
                heapq.heappush(open_set, (f, next_state))

    # Fallback: return baseline as optimised route
    cost_baseline_opt = cost_std_with_wind
    edge_meta = _compute_edge_meta_for_waypoints(
        baseline, baseline_winds, baseline_currents or [(0.0, 0.0)] * len(baseline), baseline_etas_list, ship_speed_kts
    )
    baseline_edge_meta = edge_meta
    return baseline, cost_std_no_wind, cost_std_with_wind, cost_baseline_opt, baseline_dist_nm, baseline_dist_nm, edge_meta, baseline_edge_meta
