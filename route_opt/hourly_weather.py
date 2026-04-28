"""Hourly weather lookup with time-based interpolation.

Reads ERA5 hourly Parquet files on demand and caches by (year, month).
For route optimization: loads compact arrays for only the needed points (~10 MB/month per corridor).
For weather API: loads full DataFrame once (~1.6 GB/month global), shared across all API calls.

Trade-off: first month-load is ~2s, but subsequent lookups are instant.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import duckdb
from route_opt.weather_client import preload_year

# ---------------------------------------------------------------------------
# Per-month FULL DataFrame cache for weather API  (O(1) lookups for any point)
# ---------------------------------------------------------------------------
_monthly_df_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Per-month COMPACT array cache for route optimizer (tiny footprint)
# ---------------------------------------------------------------------------
_monthly_cache: Dict[Tuple[int, int], dict] = {}


# ── Compatibility alias ─────────────────────────────────────────────────────

preload_year_hourly = preload_year
"""Alias for route optimizer that expects this name."""



def _find_hourly_parquet(year: int, month: int) -> Optional[Path]:
    p = Path(rf"C:\app\data\hourly\weather_{year}-{month:02d}_hourly.parquet")
    return p if p.exists() else None


# ── COMPACT cache: builds a dense array for specific points ──────────────────

def _build_monthly_cache(year: int, month: int, points: List[Tuple[float, float]]) -> dict:
    """Load a single month's hourly data into a (n_points, n_hours) array."""
    t0 = time.time()
    p = _find_hourly_parquet(year, month)
    if not p:
        raise FileNotFoundError(f"No hourly data for {year}-{month:02d}")

    snapped = [(round(round(lat * 4) / 4, 2), round(round(lon * 4) / 4, 2)) for lat, lon in points]
    pts_df = pd.DataFrame(snapped, columns=["latitude", "longitude"])
    pts_df["pt_idx"] = range(len(points))

    lat_min = float(pts_df["latitude"].min()) - 1.0
    lat_max = float(pts_df["latitude"].max()) + 1.0
    lon_min = float(pts_df["longitude"].min()) - 1.0
    lon_max = float(pts_df["longitude"].max()) + 1.0

    # DuckDB with spatial pushdown for compact datasets
    query = (
        f"SELECT time, latitude, longitude, wind_speed_10m, wind_direction_10m "
        f"FROM read_parquet('{str(p)}') "
        f"WHERE latitude >= {lat_min} AND latitude <= {lat_max} "
        f"  AND longitude >= {lon_min} AND longitude <= {lon_max}"
    )

    df = duckdb.query(query).to_df()
    if df.empty:
        raise FileNotFoundError(f"No hourly data for {year}-{month:02d} in bounding box")

    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)
    df["hour"] = pd.to_datetime(df["time"]).dt.floor("h")

    merged = df.merge(pts_df, on=["latitude", "longitude"], how="inner")
    if merged.empty:
        raise FileNotFoundError(f"No hourly data for requested points in {year}-{month:02d}")

    hours = sorted(merged["hour"].unique())
    hour_index = {h: i for i, h in enumerate(hours)}
    n_hours = len(hours)

    ws = np.zeros((len(points), n_hours), dtype=np.float32)
    wd = np.zeros((len(points), n_hours), dtype=np.float32)

    pt_idx = merged["pt_idx"].values
    hi = np.array([hour_index[h] for h in merged["hour"]], dtype=np.int32)
    ws[pt_idx, hi] = merged["wind_speed_10m"].values.astype(np.float32)
    wd[pt_idx, hi] = merged["wind_direction_10m"].values.astype(np.float32)

    print(f"  [hourly_weather] Compact {year}-{month:02d}: {len(points)} pts x {n_hours} hrs, "
          f"{(ws.nbytes + wd.nbytes) / 1e6:.1f} MB in {time.time() - t0:.1f}s")

    return {
        "key": tuple(points),
        "ws": ws,
        "wd": wd,
        "hours": hours,
        "hour_index": hour_index,
    }


def _get_monthly_cache(year: int, month: int, points: List[Tuple[float, float]]) -> Optional[dict]:
    cache_key = (year, month)
    cache = _monthly_cache.get(cache_key)
    if cache and set(points).issubset(set(cache["key"])):
        return cache
    # Merge with existing cache points to avoid rebuilds from different subsets
    if cache is not None:
        merged_points = list(set(points) | set(cache["key"]))
    else:
        merged_points = list(set(points))
    try:
        cache = _build_monthly_cache(year, month, merged_points)
        _monthly_cache[cache_key] = cache
        return cache
    except FileNotFoundError:
        return None


# ── FULL DataFrame cache: loads entire month once for weather API ──────────────

def _build_monthly_df(year: int, month: int) -> pd.DataFrame:
    """Load full month into memory."""
    t0 = time.time()
    p = _find_hourly_parquet(year, month)
    if not p:
        raise FileNotFoundError(f"No hourly data for {year}-{month:02d}")

    df = duckdb.query(
        f"SELECT time, latitude, longitude, wind_speed_10m, wind_direction_10m "
        f"FROM read_parquet('{str(p)}')"
    ).to_df()

    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)
    df["hour"] = pd.to_datetime(df["time"]).dt.floor("h")

    print(f"  [hourly_weather] Full {year}-{month:02d}: {len(df):,} rows "
          f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB) in {time.time() - t0:.1f}s")
    return df


def _get_monthly_df(year: int, month: int) -> pd.DataFrame:
    key = (year, month)
    if key not in _monthly_df_cache:
        _monthly_df_cache[key] = _build_monthly_df(year, month)
    return _monthly_df_cache[key]


# ── API entrypoint ──────────────────────────────────────────────────────────

def wind_at_points_hourly(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float]]:
    """Return [(ws_ms, wd_deg), ...] for each point/datetime pair."""
    if not points or not datetimes:
        return []
    if len(points) == 1 and len(datetimes) > 1:
        points = points * len(datetimes)

    snapped = [(round(round(lat * 4) / 4, 2), round(round(lon * 4) / 4, 2))
               for lat, lon in points]

    # Group by month
    month_groups: Dict[Tuple[int, int], dict] = {}
    for i, (pt, dt) in enumerate(zip(snapped, datetimes)):
        key = (dt.year, dt.month)
        month_groups.setdefault(key, {"indices": [], "pts": [], "dts": []})
        month_groups[key]["indices"].append(i)
        month_groups[key]["pts"].append(pt)
        month_groups[key]["dts"].append(dt)

    result = [None] * len(datetimes)

    # Compact cache handles any number of points efficiently (~1 MB per month).
    # Full DataFrame cache is reserved for the weather API (arbitrary global queries).
    total_points = len(points)

    for (year, month), group in month_groups.items():
        try:
            if total_points <= 10_000:
                # Use compact array cache (route optimizer, corridors up to ~10k points)
                cache = _get_monthly_cache(year, month, group["pts"])
                if cache is None:
                    raise FileNotFoundError
                _batch_lookup_hour(cache, group["pts"], group["dts"], group["indices"], result)
            else:
                # Use full DataFrame cache (weather API, massive queries)
                df = _get_monthly_df(year, month)
                _batch_lookup_df(df, group["pts"], group["dts"], group["indices"], result)
        except FileNotFoundError:
            for i in group["indices"]:
                result[i] = (0.0, 0.0)

    return result


# ── Compact array lookup (fast) ─────────────────────────────────────────────

def _batch_lookup_hour(cache, points, datetimes, indices, result):
    ws_arr = cache["ws"]
    wd_arr = cache["wd"]
    hours = cache["hours"]
    hour_index = cache["hour_index"]
    cache_points = cache["key"]

    for pt, dt, idx in zip(points, datetimes, indices):
        try:
            pi = cache_points.index(pt)
        except ValueError:
            result[idx] = (0.0, 0.0)
            continue

        target = dt.replace(minute=0, second=0, microsecond=0)

        if target in hour_index:
            hi = hour_index[target]
            result[idx] = (float(ws_arr[pi, hi]), float(wd_arr[pi, hi]))
            continue

        if target < hours[0]:
            hi = hour_index[hours[0]]
            result[idx] = (float(ws_arr[pi, hi]), float(wd_arr[pi, hi]))
            continue
        if target > hours[-1]:
            hi = hour_index[hours[-1]]
            result[idx] = (float(ws_arr[pi, hi]), float(wd_arr[pi, hi]))
            continue

        before = max((h for h in hours if h <= target), default=hours[0])
        after = min((h for h in hours if h >= target), default=hours[-1])
        if before == after:
            hi = hour_index[before]
            result[idx] = (float(ws_arr[pi, hi]), float(wd_arr[pi, hi]))
            continue

        hi_b, hi_a = hour_index[before], hour_index[after]
        total_s = (after - before).total_seconds()
        frac = (target - before).total_seconds() / total_s if total_s > 0 else 0.0

        ws_b, ws_a = float(ws_arr[pi, hi_b]), float(ws_arr[pi, hi_a])
        wd_b, wd_a = float(wd_arr[pi, hi_b]), float(wd_arr[pi, hi_a])

        ws = ws_b + frac * (ws_a - ws_b)
        delta = ((wd_a - wd_b + 180) % 360) - 180
        wd = (wd_b + frac * delta + 360) % 360
        result[idx] = (ws, wd)


# ── Full DataFrame lookup (flexible, handles many points) ─────────────────────

def _batch_lookup_df(df, points, datetimes, indices, result):
    # Build lookup DataFrame
    pts = pd.DataFrame(points, columns=["latitude", "longitude"])
    pts["hour"] = [dt.replace(minute=0, second=0, microsecond=0) for dt in datetimes]
    pts["idx"] = indices

    merged = pts.merge(df, on=["latitude", "longitude", "hour"], how="left")

    for _, row in merged.iterrows():
        i = int(row["idx"])
        result[i] = (
            float(row["wind_speed_10m"]) if pd.notna(row["wind_speed_10m"]) else 0.0,
            float(row["wind_direction_10m"]) if pd.notna(row["wind_direction_10m"]) else 0.0,
        )
