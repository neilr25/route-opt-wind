"""FastAPI wrapper around the route optimisation engine."""

import json
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
import networkx as nx
import pandas as pd

from route_opt.baseline import baseline_route as _baseline_route
from route_opt.mesh import corridor_graph
from route_opt.optimizer import optimise
from route_opt.weather_client import preload_year
from route_opt.visualizer import plot_routes

# Global mesh cache: {(start, end) tuple hash: graph}
_mesh_cache: dict[int, nx.DiGraph] = {}

def _graph_for_route(baseline: List[Tuple[float, float]]) -> nx.DiGraph:
    key = hash(tuple(baseline))
    if key in _mesh_cache:
        return _mesh_cache[key]
    G = corridor_graph(baseline)
    _mesh_cache[key] = G
    return G

app = FastAPI(title="SGS Route Optimiser", version="1.0.0")

_DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


def _serialize_mesh(G):
    """Return compact JSON-ready dict of mesh nodes + edges with full debug attrs."""
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the route optimisation dashboard."""
    if _DASHBOARD_PATH.exists():
        return _DASHBOARD_PATH.read_text(encoding="utf-8")
    return "<h1>Dashboard not found. Run from repo root.</h1>"


@app.get("/optimize")
def optimize(
    start: str = Query(..., description="Start lat,lon or named port (e.g. '51.0,3.7' or 'ROTTERDAM')"),
    end: str = Query(..., description="End lat,lon or named port"),
    speed: float = Query(default=12.0, ge=1, le=25, description="Ship speed in knots"),
    voyage_date: str = Query(default="2025-06-01", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    max_detour_pct: Optional[float] = Query(default=None, description="Max route detour as % of baseline distance (e.g. 10 for 10%)"),
    viz: bool = Query(default=False, description="Return Plotly HTML snippet?"),
):
    """Run optimisation and return baseline + optimised routes with fuel estimates."""
    try:
        start_ll = _parse_ll(start)
    except ValueError:
        start_ll = _named_port(start)
    try:
        end_ll = _parse_ll(end)
    except ValueError:
        end_ll = _named_port(end)

    baseline = _baseline_route(start_ll, end_ll)
    G = _graph_for_route(baseline)
    try:
        result_tuple = optimise(
            G, baseline, start_ll, end_ll, voyage_date, speed,
            max_detour_pct=max_detour_pct,
        )
        path, cost_std_no_wind, cost_std_wind, cost_opt, baseline_dist_nm, opt_dist_nm = result_tuple
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimisation failed: {exc}")

    # Compute distance metrics
    detour_pct = ((opt_dist_nm - baseline_dist_nm) / baseline_dist_nm * 100) if baseline_dist_nm > 0 else 0
    detour_hours = (opt_dist_nm - baseline_dist_nm) / speed if opt_dist_nm > baseline_dist_nm else 0
    baseline_hours = baseline_dist_nm / speed
    opt_hours = opt_dist_nm / speed

    result = {
        "baseline_route": baseline,
        "optimised_route": path,
        "standard_no_wind_tonnes": round(cost_std_no_wind, 2),
        "standard_with_wind_tonnes": round(cost_std_wind, 2),
        "optimised_with_wind_tonnes": round(cost_opt, 2),
        "baseline_distance_nm": round(baseline_dist_nm, 1),
        "optimised_distance_nm": round(opt_dist_nm, 1),
        "detour_pct": round(detour_pct, 1),
        "detour_hours": round(detour_hours, 1),
        "baseline_hours": round(baseline_hours, 1),
        "optimised_hours": round(opt_hours, 1),
        "mesh": _serialize_mesh(G),
    }

    return result


@app.get("/batch")
def batch_optimise(
    start: str = Query(default="ROTTERDAM"),
    end: str = Query(default="NEW YORK"),
    speed: float = Query(default=12.0, ge=1, le=25),
    year: int = Query(default=2025),
    max_detour_pct: Optional[float] = Query(default=None),
):
    """Run batch optimisation for a full year. Returns JSON array of results."""
    try:
        start_ll = _parse_ll(start)
    except ValueError:
        start_ll = _named_port(start)
    try:
        end_ll = _parse_ll(end)
    except ValueError:
        end_ll = _named_port(end)

    baseline = _baseline_route(start_ll, end_ll)
    G = _graph_for_route(baseline)

    # Pre-load weather
    mesh_points = [(d["lat"], d["lon"]) for _, d in G.nodes(data=True)]
    all_points = list(set(mesh_points + list(baseline)))
    preload_year(year, sample_points=all_points)

    d = date_cls(year, 1, 1)
    total_days = (date_cls(year, 12, 31) - d).days + 1
    records = []

    for day_num in range(total_days):
        date_str = d.strftime("%Y-%m-%d")
        try:
            result_tuple = optimise(
                G, baseline, start_ll, end_ll, date_str, speed,
                max_detour_pct=max_detour_pct,
            )
            path, cost_no, cost_wind, cost_opt, base_dist, opt_dist = result_tuple
            savings_std = cost_no - cost_wind
            savings_opt = cost_wind - cost_opt
            detour_pct = ((opt_dist - base_dist) / base_dist * 100) if base_dist > 0 else 0
            detour_hours = (opt_dist - base_dist) / speed if opt_dist > base_dist else 0
            records.append({
                "date": date_str,
                "standard_no_wind_t": round(cost_no, 2),
                "standard_with_wind_t": round(cost_wind, 2),
                "optimised_with_wind_t": round(cost_opt, 2),
                "wind_savings_t": round(savings_std, 2),
                "optimisation_extra_t": round(savings_opt, 2),
                "total_savings_t": round(savings_std + savings_opt, 2),
                "baseline_distance_nm": round(base_dist, 1),
                "optimised_distance_nm": round(opt_dist, 1),
                "detour_pct": round(detour_pct, 1),
                "detour_hours": round(detour_hours, 1),
            })
        except Exception as exc:
            records.append({
                "date": date_str,
                "error": str(exc),
            })
        d += timedelta(days=1)

    return records


@app.get("/batch_stream")
def batch_stream(
    start: str = Query(default="ROTTERDAM"),
    end: str = Query(default="NEW YORK"),
    speed: float = Query(default=12.0, ge=1, le=25),
    year: int = Query(default=2025),
    max_detour_pct: Optional[float] = Query(default=None),
):
    """Stream batch results via Server-Sent Events (SSE)."""
    try:
        start_ll = _parse_ll(start)
    except ValueError:
        start_ll = _named_port(start)
    try:
        end_ll = _parse_ll(end)
    except ValueError:
        end_ll = _named_port(end)

    baseline = _baseline_route(start_ll, end_ll)
    G = _graph_for_route(baseline)

    mesh_points = [(d["lat"], d["lon"]) for _, d in G.nodes(data=True)]
    all_points = list(set(mesh_points + list(baseline)))

    def generate():
        # Send pre-load event
        yield f"data: {json.dumps({'type': 'preload', 'year': year})}\n\n"

        preload_year(year, sample_points=all_points)

        yield f"data: {json.dumps({'type': 'preload_done', 'year': year})}\n\n"

        d = date_cls(year, 1, 1)
        total_days = (date_cls(year, 12, 31) - d).days + 1

        for day_num in range(total_days):
            date_str = d.strftime("%Y-%m-%d")
            try:
                result_tuple = optimise(
                    G, baseline, start_ll, end_ll, date_str, speed,
                    max_detour_pct=max_detour_pct,
                )
                path, cost_no, cost_wind, cost_opt, base_dist, opt_dist = result_tuple
                savings_std = cost_no - cost_wind
                savings_opt = cost_wind - cost_opt
                detour_pct = ((opt_dist - base_dist) / base_dist * 100) if base_dist > 0 else 0
                detour_hours = (opt_dist - base_dist) / speed if opt_dist > base_dist else 0
                result = {
                    "type": "result",
                    "day": day_num + 1,
                    "total": total_days,
                    "date": date_str,
                    "standard_no_wind_t": round(cost_no, 2),
                    "standard_with_wind_t": round(cost_wind, 2),
                    "optimised_with_wind_t": round(cost_opt, 2),
                    "wind_savings_t": round(savings_std, 2),
                    "optimisation_extra_t": round(savings_opt, 2),
                    "baseline_distance_nm": round(base_dist, 1),
                    "optimised_distance_nm": round(opt_dist, 1),
                    "detour_pct": round(detour_pct, 1),
                    "detour_hours": round(detour_hours, 1),
                }
            except Exception as exc:
                result = {"type": "error", "date": date_str, "error": str(exc)}

            yield f"data: {json.dumps(result)}\n\n"
            d += timedelta(days=1)

        yield f"data: {json.dumps({'type': 'done', 'total': total_days})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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
        raise HTTPException(status_code=400, detail=f"Unknown port: {name}. Known: {list(_PORT_MAP.keys())}")
    return _PORT_MAP[key]
