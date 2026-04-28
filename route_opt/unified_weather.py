"""Unified wind + current lookup with compact numpy cache.

Reads unified Parquet files (wind + current merged).
Single function: weather_and_current_at_points(points, datetimes)
Returns: [(wind_speed_ms, wind_direction_deg, current_u_ms, current_v_ms), ...]

Cache structure per month has ws, wd, cu, cv arrays keyed by (lat, lon, hour).
Lookup is O(1) array index.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Cache keyed by (year, month, point_tuple_hash)
_unified_cache: Dict[Tuple[int, int, int], dict] = {}

DATA_DIR = Path(r"C:\app\data\unified")


def _find_unified_file(year: int, month: int) -> Optional[Path]:
    p = DATA_DIR / f"unified_{year}-{month:02d}.parquet"
    return p if p.exists() else None


def _build_cache(
    year: int,
    month: int,
    points: List[Tuple[float, float]],
) -> Optional[dict]:
    """Load unified month into compact (n_points, n_hours) arrays."""
    t0 = time.time()
    p = _find_unified_file(year, month)
    if not p:
        raise FileNotFoundError(f"No unified data for {year}-{month:02d}")

    # Snap to wind grid (0.25-degree increments)
    snapped = [(round(round(lat * 4) / 4, 2), round(round(lon * 4) / 4, 2))
               for lat, lon in points]
    pts_df = pd.DataFrame(snapped, columns=["latitude", "longitude"])
    pts_df["pt_idx"] = range(len(points))

    lat_min = float(pts_df["latitude"].min()) - 1.0
    lat_max = float(pts_df["latitude"].max()) + 1.0
    lon_min = float(pts_df["longitude"].min()) - 1.0
    lon_max = float(pts_df["longitude"].max()) + 1.0

    # DuckDB spatial pushdown (fast filter)
    query = (
        f"SELECT time, latitude, longitude, wind_speed_10m, wind_direction_10m, "
        f"current_u_ms, current_v_ms "
        f"FROM read_parquet('{str(p)}') "
        f"WHERE latitude >= {lat_min} AND latitude <= {lat_max} "
        f"  AND longitude >= {lon_min} AND longitude <= {lon_max}"
    )

    import duckdb
    df = duckdb.query(query).to_df()

    if df.empty:
        raise FileNotFoundError(f"No unified data in bounding box for {year}-{month:02d}")

    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)
    df["hour"] = pd.to_datetime(df["time"]).dt.floor("h")

    # Drop exact duplicate rows per (lat, lon, hour).
    # Unified files may have duplicates where ERA5 hourly + ESPC 3-hourly overlap.
    # Keep rows with non-zero current data (ESPC/GLBy timesteps) over zero-current rows.
    df = df.sort_values("current_u_ms", ascending=False, kind="mergesort")
    df = df.drop_duplicates(subset=["latitude", "longitude", "hour"], keep="first")

    # ESPC/GLBy currents are 3-hourly; ERA5 wind is hourly.
    # Forward-fill currents so non-ESPC hours inherit the nearest previous 3-hourly value.
    df = df.sort_values(["latitude", "longitude", "hour"])
    for col in ["current_u_ms", "current_v_ms"]:
        df[col] = df.groupby(["latitude", "longitude"])[col].ffill()

    merged = df.merge(pts_df, on=["latitude", "longitude"], how="inner")
    if merged.empty:
        raise FileNotFoundError(f"No data for requested points in {year}-{month:02d}")

    hours = sorted(merged["hour"].unique())
    hour_index = {h: i for i, h in enumerate(hours)}
    n_hours = len(hours)

    ws = np.zeros((len(points), n_hours), dtype=np.float32)
    wd = np.zeros((len(points), n_hours), dtype=np.float32)
    cu = np.zeros((len(points), n_hours), dtype=np.float32)
    cv = np.zeros((len(points), n_hours), dtype=np.float32)

    pt_idx = merged["pt_idx"].values.astype(np.int32)
    hi = np.array([hour_index[h] for h in merged["hour"]], dtype=np.int32)

    ws[pt_idx, hi] = merged["wind_speed_10m"].values.astype(np.float32)
    wd[pt_idx, hi] = merged["wind_direction_10m"].values.astype(np.float32)
    cu[pt_idx, hi] = merged["current_u_ms"].values.astype(np.float32)
    cv[pt_idx, hi] = merged["current_v_ms"].values.astype(np.float32)

    print(f"  [unified] Cache {year}-{month:02d}: {len(points)} pts x {n_hours} hrs, "
          f"{(ws.nbytes + wd.nbytes + cu.nbytes + cv.nbytes) / 1e6:.1f} MB "
          f"in {time.time() - t0:.1f}s")

    return {
        "key": tuple(points),
        "ws": ws,
        "wd": wd,
        "cu": cu,
        "cv": cv,
        "hours": hours,
        "hour_index": hour_index,
    }


def _get_cache(year: int, month: int, points: List[Tuple[float, float]]) -> Optional[dict]:
    # Use (year, month) as cache key so different point subsets share the same cache.
    # Merge new points into existing cache rather than rebuilding from scratch.
    cache_key = (year, month)
    cache = _unified_cache.get(cache_key)
    if cache and set(points).issubset(set(cache["key"])):
        return cache
    if cache is not None:
        merged_points = list(set(points) | set(cache["key"]))
    else:
        merged_points = list(set(points))
    try:
        cache = _build_cache(year, month, merged_points)
        _unified_cache[cache_key] = cache
        return cache
    except FileNotFoundError:
        return None


def weather_and_current_at_points(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float, float, float]]:
    """Return [(ws_ms, wd_deg, cu_ms, cv_ms), ...] for each point/datetime pair."""
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

    for (year, month), group in month_groups.items():
        cache = _get_cache(year, month, group["pts"])
        if cache is None:
            for i in group["indices"]:
                result[i] = (0.0, 0.0, 0.0, 0.0)
            continue
        _batch_lookup(cache, group["pts"], group["dts"], group["indices"], result)

    return result


def _batch_lookup(cache, points, datetimes, indices, result):
    ws_arr = cache["ws"]
    wd_arr = cache["wd"]
    cu_arr = cache["cu"]
    cv_arr = cache["cv"]
    hours = cache["hours"]
    hour_index = cache["hour_index"]
    cache_points = cache["key"]

    for pt, dt, idx in zip(points, datetimes, indices):
        try:
            pi = cache_points.index(pt)
        except ValueError:
            result[idx] = (0.0, 0.0, 0.0, 0.0)
            continue

        target = dt.replace(minute=0, second=0, microsecond=0)

        if target in hour_index:
            hi = hour_index[target]
        elif target < hours[0]:
            hi = hour_index[hours[0]]
        elif target > hours[-1]:
            hi = hour_index[hours[-1]]
        else:
            before = max((h for h in hours if h <= target), default=hours[0])
            after = min((h for h in hours if h >= target), default=hours[-1])
            if before == after:
                hi = hour_index[before]
            else:
                hi_b, hi_a = hour_index[before], hour_index[after]
                total_s = (after - before).total_seconds()
                frac = (target - before).total_seconds() / total_s if total_s > 0 else 0.0

                ws_b, ws_a = float(ws_arr[pi, hi_b]), float(ws_arr[pi, hi_a])
                wd_b, wd_a = float(wd_arr[pi, hi_b]), float(wd_arr[pi, hi_a])
                cu_b, cu_a = float(cu_arr[pi, hi_b]), float(cu_arr[pi, hi_a])
                cv_b, cv_a = float(cv_arr[pi, hi_b]), float(cv_arr[pi, hi_a])

                ws = ws_b + frac * (ws_a - ws_b)
                delta = ((wd_a - wd_b + 180) % 360) - 180
                wd = (wd_b + frac * delta + 360) % 360
                cu = cu_b + frac * (cu_a - cu_b)
                cv = cv_b + frac * (cv_a - cv_b)
                result[idx] = (ws, wd, cu, cv)
                continue

        result[idx] = (
            float(ws_arr[pi, hi]),
            float(wd_arr[pi, hi]),
            float(cu_arr[pi, hi]),
            float(cv_arr[pi, hi]),
        )


def clear_cache():
    _unified_cache.clear()
