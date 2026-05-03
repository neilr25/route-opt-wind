"""FastAPI wrapper around the route optimisation engine."""

import json
from datetime import date as date_cls, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
import networkx as nx
import pandas as pd

from route_opt.baseline import baseline_route as _baseline_route
from route_opt.mesh import corridor_graph as _graph_for_route
from route_opt.optimizer import optimise
from route_opt.corridor_weather import ensure_month_loaded, clear_cache, get_cache_status
from route_opt.visualizer import plot_routes
from route_opt.api_helpers import _serialize_mesh, _parse_ll, _named_port
from route_opt.weather_api import router as weather_router


app = FastAPI(title="SGS Route Optimiser", version="1.0.0")
app.include_router(weather_router)

_DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cache/clear")
def clear_cache_endpoint():
    """Wipe all in-memory bbox and DataFrame caches. Safe to call anytime."""
    clear_cache()
    return {"cleared": True}


@app.get("/cache/status")
def cache_status_endpoint():
    """Return bbox cache stats for dashboard display."""
    return get_cache_status()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the route optimisation dashboard."""
    if _DASHBOARD_PATH.exists():
        return _DASHBOARD_PATH.read_text(encoding="utf-8")
    return "<h1>Dashboard not found. Run from repo root.</h1>"


@app.get("/optimize")
def optimize(
    start: str = Query(..., description="Start lat,lon or named port (e.g. '51.0,3.7' or 'COPENHAGEN'). Demo ports: CHIBA, COPENHAGEN, LOOP TERMINAL, MELBOURNE, NOVOROSSIYSK, PORT RASHID"),
    end: str = Query(..., description="End lat,lon or named port"),
    speed: float = Query(default=12.0, ge=1, le=25, description="Ship speed in knots"),
    voyage_date: str = Query(default="2025-06-01", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    voyage_time: str = Query(default="12:00", pattern=r"^\d{2}:\d{2}$", description="UTC time (HH:MM) — defaults to 12:00 UTC (matches daily ERA5 snapshot)"),
    max_detour_pct: Optional[float] = Query(default=None, description="Max route detour as % of baseline distance (e.g. 10 for 10%)"),
    viz: bool = Query(default=False, description="Return Plotly HTML snippet?"),
    use_currents: bool = Query(default=True, description="Include ocean current effects on speed-over-ground (default true). Set false for wind-only routing."),
    corridor_width_nm: Optional[float] = Query(default=None, description="Corridor width in nm (default 200)"),
    lane_spacing_nm: Optional[float] = Query(default=None, description="Lane spacing in nm (default 25)"),
    stage_skip: Optional[int] = Query(default=None, description="Stage skip — connect every Nth waypoint (default 4)"),
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

    # Override corridor config if specified
    if corridor_width_nm is not None or lane_spacing_nm is not None or stage_skip is not None:
        from route_opt.config import CORRIDOR_WIDTH_NM as _CW, LANE_SPACING_NM as _LS, STAGE_SKIP as _SS
        from route_opt.mesh import corridor_graph as _corridor_graph_fn
        G = _corridor_graph_fn(
            baseline,
            width_nm=corridor_width_nm if corridor_width_nm is not None else _CW,
            lane_spacing_nm=lane_spacing_nm if lane_spacing_nm is not None else _LS,
            stage_skip=stage_skip if stage_skip is not None else _SS,
        )

    # Parse voyage datetime
    try:
        voyage_dt = datetime.strptime(f"{voyage_date} {voyage_time}", "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid voyage time: {voyage_time}. Expected HH:MM")

    try:
        result_tuple = optimise(
            G, baseline, start_ll, end_ll, voyage_date, speed,
            voyage_datetime=voyage_dt,
            max_detour_pct=max_detour_pct,
            use_currents=use_currents,
        )
        path, cost_std_no_wind, cost_std_wind, cost_opt, baseline_dist_nm, opt_dist_nm, edge_meta, baseline_edge_meta = result_tuple
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
        "wind_savings_t": round(cost_std_no_wind - cost_std_wind, 2),
        "optimisation_extra_t": round(cost_std_wind - cost_opt, 2),
        "total_savings_t": round(cost_std_no_wind - cost_opt, 2),
        "wind_savings_pct": round((cost_std_no_wind - cost_std_wind) / cost_std_no_wind * 100, 1) if cost_std_no_wind > 0 else 0,
        "total_savings_pct": round((cost_std_no_wind - cost_opt) / cost_std_no_wind * 100, 1) if cost_std_no_wind > 0 else 0,
        "baseline_distance_nm": round(baseline_dist_nm, 1),
        "optimised_distance_nm": round(opt_dist_nm, 1),
        "detour_pct": round(detour_pct, 1),
        "detour_hours": round(detour_hours, 1),
        "baseline_hours": round(baseline_hours, 1),
        "optimised_hours": round(opt_hours, 1),
        "use_currents": use_currents,
        "mesh": _serialize_mesh(G),
        "edge_meta": edge_meta,
        "baseline_edge_meta": baseline_edge_meta,
    }

    return result


@app.get("/batch")
def batch_optimise(
    start: str = Query(default="COPENHAGEN"),
    end: str = Query(default="LOOP TERMINAL"),
    speed: float = Query(default=12.0, ge=1, le=25),
    start_date: str = Query(default="2025-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: str = Query(default="2025-12-31", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    max_detour_pct: Optional[float] = Query(default=None),
):
    """Run batch optimisation for a date range. Returns JSON array of results."""
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

    # Parse date range
    try:
        d = date_cls.fromisoformat(start_date)
        end_d = date_cls.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    total_days = (end_d - d).days + 1
    records = []

    # Track which months we've loaded
    _loaded_months = set()

    for day_num in range(total_days):
        current_date = d + timedelta(days=day_num) if day_num > 0 else d
        date_str = current_date.strftime("%Y-%m-%d")
        # Lazy-load corridor data on first encounter of each month
        month_key = (current_date.year, current_date.month)
        if month_key not in _loaded_months:
            ensure_month_loaded(current_date.year, current_date.month)
            _loaded_months.add(month_key)
        voyage_dt = datetime(current_date.year, current_date.month, current_date.day, 12, 0)
        try:
            result_tuple = optimise(
                G, baseline, start_ll, end_ll, date_str, speed,
                voyage_datetime=voyage_dt,
                max_detour_pct=max_detour_pct,
            )
            path, cost_no, cost_wind, cost_opt, base_dist, opt_dist, _, _ = result_tuple
            savings_std = cost_no - cost_wind
            savings_opt = cost_wind - cost_opt
            wind_savings_pct = (savings_std / cost_no * 100) if cost_no > 0 else 0
            total_savings_pct = ((cost_no - cost_opt) / cost_no * 100) if cost_no > 0 else 0
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
                "wind_savings_pct": round(wind_savings_pct, 1),
                "total_savings_pct": round(total_savings_pct, 1),
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
    start: str = Query(default="COPENHAGEN"),
    end: str = Query(default="LOOP TERMINAL"),
    speed: float = Query(default=12.0, ge=1, le=25),
    start_date: str = Query(default="2025-01-01", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    end_date: str = Query(default="2025-12-31", pattern=r"^\d{4}-\d{2}-\d{2}$"),
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

    try:
        d = date_cls.fromisoformat(start_date)
        end_d = date_cls.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    total_days = (end_d - d).days + 1

    def generate():
        _loaded_months = set()
        yield f"data: {json.dumps({'type': 'preload', 'total': total_days, 'start': start_date, 'end': end_date})}\n\n"
        yield f"data: {json.dumps({'type': 'preload_done', 'total': total_days})}\n\n"

        for day_num in range(total_days):
            try:
                current_date = d + timedelta(days=day_num)
                date_str = current_date.strftime("%Y-%m-%d")
                # Lazy-load corridor data on first encounter of each month
                month_key = (current_date.year, current_date.month)
                if month_key not in _loaded_months:
                    ensure_month_loaded(current_date.year, current_date.month)
                    _loaded_months.add(month_key)
                voyage_dt = datetime(current_date.year, current_date.month, current_date.day, 12, 0)
                result_tuple = optimise(
                    G, baseline, start_ll, end_ll, date_str, speed,
                    voyage_datetime=voyage_dt,
                    max_detour_pct=max_detour_pct,
                )
                path, cost_no, cost_wind, cost_opt, base_dist, opt_dist, edge_meta, baseline_edge_meta = result_tuple
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
                    "wind_savings_t": round(cost_no - cost_wind, 2),
                    "optimisation_extra_t": round(cost_wind - cost_opt, 2),
                    "total_savings_t": round(cost_no - cost_opt, 2),
                    "wind_savings_pct": round((cost_no - cost_wind) / cost_no * 100, 1) if cost_no > 0 else 0,
                    "total_savings_pct": round((cost_no - cost_opt) / cost_no * 100, 1) if cost_no > 0 else 0,
                    "baseline_distance_nm": round(base_dist, 1),
                    "optimised_distance_nm": round(opt_dist, 1),
                    "detour_pct": round(detour_pct, 1),
                    "detour_hours": round(detour_hours, 1),
                    "edge_meta": edge_meta,
                    "baseline_edge_meta": baseline_edge_meta,
                }
            except Exception as exc:
                result = {"type": "error", "day": day_num + 1, "total": total_days, "date": date_str, "error": str(exc)}

            yield f"data: {json.dumps(result)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'total': total_days})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/weather_hourly")
def weather_hourly(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """Return 24 hourly wind readings for a single lat/lon on a given date."""
    from route_opt.hourly_direct import read_hourly_for_point
    try:
        winds = read_hourly_for_point(lat, lon, date)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No hourly data for {date}. Generate it first via scripts/generate_hourly_for_route.py")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "date": date,
        "lat": lat,
        "lon": lon,
        "hours": [
            {"hour": h, "wind_speed_ms": round(ws, 2), "wind_direction_deg": round(wd, 1)}
            for h, (ws, wd) in enumerate(winds)
        ]
    }



