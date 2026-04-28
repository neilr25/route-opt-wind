"""Weather-only API — fast local ERA5 lookup, drop-in replacement for Open-Meteo.

Endpoints mirror Open-Meteo archive API:
    /weather           → single point, single day, hourly
    /weather_range     → single point, date range, hourly
    /weather_batch     → multiple points, single day, hourly (batched lookup)

All lookups are local Parquet reads — no network latency.
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query

from route_opt.hourly_weather import wind_at_points_hourly

router = APIRouter(prefix="/weather", tags=["weather"])


def _snap_to_grid(val: float) -> float:
    return round(round(val * 4) / 4, 2)


def _parse_date_range(start_date: str, end_date: str) -> Tuple[date, date]:
    try:
        d_start = date.fromisoformat(start_date)
        d_end = date.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    if d_end < d_start:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")
    return d_start, d_end


def _make_datetimes(start: date, end: date, hour: Optional[int] = None) -> List[datetime]:
    """Generate hourly datetimes from start_date to end_date inclusive."""
    d = start
    hours = []
    while d <= end:
        if hour is None:
            for h in range(24):
                hours.append(datetime(d.year, d.month, d.day, h, 0))
        else:
            hours.append(datetime(d.year, d.month, d.day, hour, 0))
        d += timedelta(days=1)
    return hours


def _ms_to_kn(ws_ms: float) -> float:
    return round(ws_ms * 1.94384, 2)


@router.get("")
def weather(
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude (or use latitude alias)"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude (or use longitude alias)"),
    # Drop-in Open-Meteo aliases (single or comma-separated)
    latitude: Optional[str] = Query(None, description="Open-Meteo alias for lat (single or comma-separated)"),
    longitude: Optional[str] = Query(None, description="Open-Meteo alias for lon (single or comma-separated)"),
    start_date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="End date (YYYY-MM-DD), defaults to start_date"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Filter to single hour (0–23). Omit for all 24 hours."),
    wind_speed_unit: str = Query("ms", pattern=r"^(ms|kn)$", description="Wind speed unit: ms (m/s) or kn (knots)"),
    hourly: Optional[str] = Query(None, description="[Open-Meteo compat] Ignored — always returns wind_speed_10m + wind_direction_10m"),
    timezone: str = Query("GMT", description="Timezone (ignored — all data is UTC)"),
    apikey: Optional[str] = Query(None, description="API key (ignored — no auth required)"),
):
    """Return hourly wind for a single lat/lon over a date range.
    
    Mirrors Open-Meteo archive API response shape for easy client swaps.
    Supports comma-separated latitude/longitude for multi-point queries.
    """
    # Resolve coordinates
    if latitude and longitude:
        lats = [float(x.strip()) for x in latitude.split(",")]
        lons = [float(x.strip()) for x in longitude.split(",")]
        if len(lats) != len(lons):
            raise HTTPException(status_code=400, detail="latitude and longitude must have same number of values")
        points = list(zip(lats, lons))
        multi_point = len(points) > 1
    elif lat is not None and lon is not None:
        points = [(lat, lon)]
        multi_point = False
    else:
        raise HTTPException(status_code=400, detail="Provide lat+lon OR latitude+longitude")

    end = end_date or start_date
    d_start, d_end = _parse_date_range(start_date, end)
    datetimes = _make_datetimes(d_start, d_end, hour)
    if not datetimes:
        raise HTTPException(status_code=400, detail="Date range produced no hours")

    # For single-point queries, extract lat/lon from points (works whether lat/lon or latitude/longitude was used)
    if not multi_point:
        lat, lon = points[0]

    # Build parallel look-up lists for wind_at_points_hourly
    if multi_point:
        lookup_points = []
        lookup_datetimes = []
        for dt in datetimes:
            for p in points:
                lookup_points.append(p)
                lookup_datetimes.append(dt)
    else:
        lookup_points = points * len(datetimes)
        lookup_datetimes = datetimes

    # Look up
    t0 = datetime.now()
    results = wind_at_points_hourly(lookup_points, lookup_datetimes)
    elapsed_ms = (datetime.now() - t0).total_seconds() * 1000

    convert = _ms_to_kn if wind_speed_unit == "kn" else lambda x: round(x, 2)
    speed_unit_label = "kn" if wind_speed_unit == "kn" else "m/s"

    # Pre-compute ISO time strings without seconds (Open-Meteo format: 2025-06-01T00:00)
    iso_times = [dt.strftime("%Y-%m-%dT%H:%M") for dt in datetimes]

    if multi_point:
        # Build columnar arrays matching Open-Meteo format
        # Data is concatenated: point 1 all hours, then point 2 all hours, ...
        all_times = []
        all_speeds = []
        all_dirs = []
        for p_idx in range(len(points)):
            for dt_idx in range(len(datetimes)):
                flat_idx = dt_idx * len(points) + p_idx
                ws, wd = results[flat_idx]
                all_times.append(iso_times[dt_idx])
                all_speeds.append(convert(ws))
                all_dirs.append(round(wd, 1))

        return {
            "latitude": [_snap_to_grid(p[0]) for p in points],
            "longitude": [_snap_to_grid(p[1]) for p in points],
            "generationtime_ms": round(elapsed_ms, 2),
            "utc_offset_seconds": 0,
            "timezone": timezone,
            "timezone_abbreviation": timezone,
            "elevation": [0.0] * len(points),
            "hourly_units": {
                "time": "iso8601",
                "wind_speed_10m": speed_unit_label,
                "wind_direction_10m": "°",
            },
            "hourly": {
                "time": all_times,
                "wind_speed_10m": all_speeds,
                "wind_direction_10m": all_dirs,
            },
        }

    # Columnar Open-Meteo-compatible format (single point)
    iso_times = [dt.strftime("%Y-%m-%dT%H:%M") for dt in datetimes]
    all_speeds = [convert(ws) for ws, wd in results]
    all_dirs   = [round(wd, 1) for ws, wd in results]

    return {
        "latitude": _snap_to_grid(lat),
        "longitude": _snap_to_grid(lon),
        "generationtime_ms": round(elapsed_ms, 2),
        "utc_offset_seconds": 0,
        "timezone": timezone,
        "timezone_abbreviation": timezone,
        "elevation": 0.0,
        "hourly_units": {
            "time": "iso8601",
            "wind_speed_10m": speed_unit_label,
            "wind_direction_10m": "°",
        },
        "hourly": {
            "time": iso_times,
            "wind_speed_10m": all_speeds,
            "wind_direction_10m": all_dirs,
        },
    }


@router.get("/batch")
def weather_batch(
    coords: str = Query(..., description="Comma-separated lat,lon pairs. E.g. '51.92,4.48|40.71,-74.01'"),
    query_date: str = Query(..., alias="date", pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date (YYYY-MM-DD)"),
    hour: Optional[int] = Query(None, ge=0, le=23, description="Single hour filter"),
    wind_speed_unit: str = Query("ms", pattern=r"^(ms|kn)$", description="Wind speed unit: ms (m/s) or kn (knots)"),
):
    """Return hourly wind for many coordinates on the same day — batched in one lookup.
    
    Much faster than N separate /weather calls because all coordinates share
    the same month cache load.
    """
    # Parse coordinate string: "51.92,4.48|40.71,-74.01|..."
    points = []
    for pair in coords.split("|"):
        try:
            lat, lon = [float(x.strip()) for x in pair.split(",")]
            points.append((lat, lon))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid coordinate pair: {pair}. Use lat,lon")

    if not points:
        raise HTTPException(status_code=400, detail="No valid coordinates provided")

    d = date.fromisoformat(query_date)
    if hour is None:
        datetimes = [datetime(d.year, d.month, d.day, h, 0) for h in range(24)]
    else:
        datetimes = [datetime(d.year, d.month, d.day, hour, 0)] * len(points)

    # Pad points to match datetimes length if single hour mode
    if hour is not None:
        lookup_points = points
    else:
        # Repeat each point 24x so zip(points, datetimes) lines up
        lookup_points = []
        for p in points:
            lookup_points.extend([p] * 24)
        # Also flatten datetimes to match
        datetimes_flat = datetimes * len(points)
        datetimes = datetimes_flat

    t0 = datetime.now()
    results = wind_at_points_hourly(lookup_points, datetimes)
    elapsed_ms = (datetime.now() - t0).total_seconds() * 1000

    convert = _ms_to_kn if wind_speed_unit == "kn" else lambda x: round(x, 2)
    speed_unit_label = "kn" if wind_speed_unit == "kn" else "m/s"

    # Re-shape results back into per-point arrays
    if hour is None:
        per_point = []
        idx = 0
        for _ in points:
            hours_data = []
            for h in range(24):
                ws, wd = results[idx]
                hours_data.append({
                    "hour": h,
                    "wind_speed_10m": convert(ws),
                    "wind_direction_10m": round(wd, 1),
                })
                idx += 1
            per_point.append(hours_data)
    else:
        per_point = [
            [{"hour": hour, "wind_speed_10m": convert(ws), "wind_direction_10m": round(wd, 1)}]
            for ws, wd in results
        ]

    return {
        "date": query_date,
        "coordinates": len(points),
        "generation_time_ms": round(elapsed_ms, 2),
        "units": {
            "wind_speed_10m": speed_unit_label,
            "wind_direction_10m": "°",
        },
        "results": [
            {
                "lat": _snap_to_grid(lat),
                "lon": _snap_to_grid(lon),
                "hours": hours_data,
            }
            for (lat, lon), hours_data in zip(points, per_point)
        ],
    }
