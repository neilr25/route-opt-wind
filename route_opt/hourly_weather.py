"""Hourly weather lookup with time-based interpolation.

Reads ERA5 hourly Parquet files (`C:\\app\\data\\hourly\\weather_YYYY-MM_hourly.parquet`)
and returns wind at (lat, lon, datetime) with bilinear time interpolation.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_hourly_cache: Dict[int, dict] = {}


def _find_hourly_parquet(year: int, month: int) -> Optional[Path]:
    p = Path(rf"C:\app\data\hourly\weather_{year}-{month:02d}_hourly.parquet")
    return p if p.exists() else None


def _build_hourly_cache(year: int, points: List[Tuple[float, float]]) -> dict:
    n_pts = len(points)
    t0 = time.time()

    snapped = [(round(round(lat * 4) / 4, 2), round(round(lon * 4) / 4, 2)) for lat, lon in points]
    pts_df = pd.DataFrame(snapped, columns=["latitude", "longitude"])
    pts_df["pt_idx"] = range(n_pts)

    lat_min = float(pts_df["latitude"].min()) - 1.0
    lat_max = float(pts_df["latitude"].max()) + 1.0
    lon_min = float(pts_df["longitude"].min()) - 1.0
    lon_max = float(pts_df["longitude"].max()) + 1.0

    dfs = []
    for month in range(1, 13):
        p = _find_hourly_parquet(year, month)
        if not p:
            continue
        df = pd.read_parquet(p, columns=["time", "latitude", "longitude", "wind_speed_10m", "wind_direction_10m"])
        df = df[
            (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        ]
        if not df.empty:
            df["latitude"] = df["latitude"].round(2)
            df["longitude"] = df["longitude"].round(2)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No hourly weather data for {year}")

    all_data = pd.concat(dfs, ignore_index=True)
    all_data["hour"] = pd.to_datetime(all_data["time"]).dt.floor("h")

    merged = all_data.merge(pts_df, on=["latitude", "longitude"], how="inner")
    if merged.empty:
        raise FileNotFoundError(f"No hourly data for requested points in {year}")
    hours = sorted(merged["hour"].unique())
    hour_index = {h: i for i, h in enumerate(hours)}
    n_hours = len(hours)

    ws = np.zeros((n_pts, n_hours), dtype=np.float32)
    wd = np.zeros((n_pts, n_hours), dtype=np.float32)

    pt_idxs = merged["pt_idx"].values
    hour_idxs = np.array([hour_index[h] for h in merged["hour"]], dtype=np.int32)
    ws[pt_idxs, hour_idxs] = merged["wind_speed_10m"].values.astype(np.float32)
    wd[pt_idxs, hour_idxs] = merged["wind_direction_10m"].values.astype(np.float32)

    print(f"  [hourly_weather] Built {year} cache: {n_pts} pts x {n_hours} hrs, "
          f"{(ws.nbytes+wd.nbytes)/1e6:.1f} MB in {time.time()-t0:.1f}s")

    result = {"key": tuple(points), "ws": ws, "wd": wd, "hours": hours, "hour_index": hour_index}
    _hourly_cache[year] = result
    return result


def _get_cache(year: int, points: List[Tuple[float, float]]) -> Optional[dict]:
    cache = _hourly_cache.get(year)
    if cache and set(points).issubset(set(cache.get("key", []))):
        return cache
    try:
        return _build_hourly_cache(year, list(set(points)))
    except FileNotFoundError:
        return None


def wind_at_points_hourly(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float]]:
    """Return [(ws_ms, wd_deg), ...] for each point/datetime pair with bilinear interpolation.

    Falls back to daily weather_client.wind_at_points if hourly data is not available.
    """
    if not points:
        return []

    if len(points) == 1 and len(datetimes) >= 1:
        # Single point queried for multiple times — expand to match lengths
        points = points * len(datetimes)
    if not datetimes:
        return []

    year = datetimes[0].year

    cache = _get_cache(year, points)
    if cache is None:
        # Fall back to daily data (no time interpolation)
        from route_opt.weather_client import wind_at_points as _daily
        date_str = datetimes[0].strftime("%Y-%m-%d")
        return _daily(points, date_str)

    pt_map = {pt: i for i, pt in enumerate(cache["key"])}
    ws_arr = cache["ws"]
    wd_arr = cache["wd"]
    hours = cache["hours"]
    hour_index = cache["hour_index"]
    n_hours = len(hours)

    result = []
    for pt, dt in zip(points, datetimes):
        pi = pt_map.get(pt, -1)
        if pi < 0 or pi >= ws_arr.shape[0]:
            result.append((0.0, 0.0))
            continue

        target = dt.replace(minute=0, second=0, microsecond=0)
        if target in hour_index:
            hi = hour_index[target]
            result.append((float(ws_arr[pi, hi]), float(wd_arr[pi, hi])))
            continue

        # Bilinear time interpolation between neighbouring hours
        # Find floor and ceil hours
        if target < hours[0] or target > hours[-1]:
            # Outside cache range — clip to nearest
            nearest = min(hours, key=lambda h: abs((h - target).total_seconds()))
            hi = hour_index[nearest]
            result.append((float(ws_arr[pi, hi]), float(wd_arr[pi, hi])))
            continue

        # Find hour just before and just after target
        before = max((h for h in hours if h <= target), default=hours[0])
        after = min((h for h in hours if h >= target), default=hours[-1])
        if before == after:
            hi = hour_index[before]
            result.append((float(ws_arr[pi, hi]), float(wd_arr[pi, hi])))
            continue

        hi_b = hour_index[before]
        hi_a = hour_index[after]
        total_s = (after - before).total_seconds()
        frac = (target - before).total_seconds() / total_s if total_s > 0 else 0.0

        ws_b = float(ws_arr[pi, hi_b])
        ws_a = float(ws_arr[pi, hi_a])
        wd_b = float(wd_arr[pi, hi_b])
        wd_a = float(wd_arr[pi, hi_a])

        # Interpolate wind speed linearly
        ws_interp = ws_b + frac * (ws_a - ws_b)

        # Interpolate direction with circular handling
        delta = ((wd_a - wd_b + 180) % 360) - 180
        wd_interp = (wd_b + frac * delta + 360) % 360

        result.append((ws_interp, wd_interp))

    return result


def preload_year_hourly(year: int, points: List[Tuple[float, float]]) -> None:
    """Pre-load an entire year of hourly weather into memory."""
    _get_cache(year, points)
