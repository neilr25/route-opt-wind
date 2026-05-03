"""Corridor weather lookup using integer-encoded unified Parquet + DuckDB.
Replaces old ESPC-based corridor_weather.py.

API-compatible with optimizer.py expectations:
  weather_and_current_at_points(points, datetimes) -> [(ws, wd, cu, cv)]
  wind_at_points_hourly(points, datetimes) -> [(ws, wd)]
  ensure_month_loaded(year, month) -> bool
  clear_cache()
  preload_bounding_box(year, month, lat_min, lat_max, lon_min, lon_max, hr_min, hr_max) -> bool
"""
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import duckdb
import numpy as np

DATA_DIR = r"C:\app\data\unified"
EPOCH_2025 = 1735689600

# Per-month FULL DataFrame cache for weather API
_cache: Dict[Tuple[int, int], Dict] = {}
_mtime_cache: Dict[Tuple[int, int], float] = {}

# Per-month bounding-box pre-loaded cache for route optimizer
# Key: (year, month), Value: dict mapping (lat_idx, lon_idx, hour_rel) -> (ws, wd, cu, cv)
_bbox_cache: Dict[Tuple[int, int], Dict[Tuple[int, int, int], Tuple[float, float, float, float]]] = {}


def _file_for(year: int, month: int) -> Optional[str]:
    from pathlib import Path
    p = Path(DATA_DIR) / f"unified_{year}-{month:02d}_int.parquet"
    if p.exists():
        return str(p)
    # Fall back to monthly float file (for any month not yet rebuilt)
    p2 = Path(DATA_DIR) / f"unified_{year}-{month:02d}.parquet"
    return str(p2) if p2.exists() else None


def preload_bounding_box(year: int, month: int, points: List[Tuple[float, float]], hr_min: int, hr_max: int) -> bool:
    """Pre-load weather data for exact grid cells used by a route into memory.

    Call once before optimise() to eliminate per-call DuckDB round-trips.
    Collects only the unique snapped grid cells from points, loads ALL hours
    for those cells. Lookup is then O(1) dict access.
    """
    f = _file_for(year, month)
    if f is None:
        return False

    # Snap points to ERA5 0.25 deg grid and deduplicate
    snapped = set()
    for lat, lon in points:
        li = int(round((lat + 90.0) / 0.25))
        lj = int(round((lon + 180.0) / 0.25))
        li = max(0, min(719, li))
        lj = max(0, min(1439, lj))
        snapped.add((li, lj))
    snapped = sorted(snapped)

    # Already cached?
    cache_key = (year, month)
    if cache_key in _bbox_cache:
        return True

    val_parts = [f"({li},{lj})" for li, lj in snapped]
    val_clause = "VALUES " + ",".join(val_parts)
    duck_path = f.replace(chr(92), '/')

    query = f"""
        SELECT p.lat_idx, p.lon_idx, w.hour_rel,
               w.wind_speed, w.wind_dir, w.current_u, w.current_v
        FROM ({val_clause}) AS p(lat_idx, lon_idx)
        INNER JOIN read_parquet('{duck_path}') w
          ON p.lat_idx = w.lat_idx AND p.lon_idx = w.lon_idx
        WHERE w.hour_rel BETWEEN {hr_min} AND {hr_max}
    """

    t0 = time.time()
    con = duckdb.connect()
    con.execute("SET temp_directory=''")
    try:
        rows = con.execute(query).fetchall()
    finally:
        con.close()

    lookup = {}
    for row in rows:
        key = (row[0], row[1], row[2])
        lookup[key] = (
            float(row[3]) / 100.0,
            float(row[4]) / 10.0,
            float(row[5]) / 1000.0,
            float(row[6]) / 1000.0,
        )

    _bbox_cache[cache_key] = lookup
    print(f"  [corridor_weather] Cell preload {year}-{month:02d}: "
          f"{len(snapped)} cells, hr {hr_min}-{hr_max}, {len(rows)} rows in {time.time() - t0:.2f}s")
    return True


def _weather_and_current_bbox(year: int, month: int, points: List[Tuple[float, float]],
                               datetimes: List[datetime]) -> List[Tuple[float, float, float, float]]:
    """O(1) dict lookup using pre-loaded bounding box cache."""
    cache = _bbox_cache.get((year, month))
    if cache is None:
        raise FileNotFoundError(f"No bounding box cache for {year}-{month:02d}. "
                                f"Call preload_bounding_box first.")

    result = []
    for (lat, lon), dt in zip(points, datetimes):
        li = int(round((lat + 90.0) / 0.25))
        lj = int(round((lon + 180.0) / 0.25))
        li = max(0, min(719, li))
        lj = max(0, min(1439, lj))
        hr = int((dt.timestamp() - EPOCH_2025) / 3600)
        key = (li, lj, hr)
        val = cache.get(key, (0.0, 0.0, 0.0, 0.0))
        result.append(val)
    return result


def _weather_and_current(points, datetimes):
    """Original point-by-point DuckDB lookup (fallback when no bbox cache)."""
    if not points or not datetimes:
        return []
    if len(points) == 1 and len(datetimes) > 1:
        points = points * len(datetimes)

    snapped = []
    for lat, lon in points:
        li = int(round((lat + 90.0) / 0.25))
        lj = int(round((lon + 180.0) / 0.25))
        li = max(0, min(719, li))
        lj = max(0, min(1439, lj))
        snapped.append((li, lj))

    month_groups = {}
    for i, (pt, dt) in enumerate(zip(snapped, datetimes)):
        key = (dt.year, dt.month)
        month_groups.setdefault(key, {"indices": [], "pts": [], "dts": []})
        month_groups[key]["indices"].append(i)
        month_groups[key]["pts"].append(pt)
        month_groups[key]["dts"].append(dt)

    result = [None] * len(datetimes)

    for (year, month), group in month_groups.items():
        f = _file_for(year, month)
        if f is None:
            for i in group["indices"]:
                result[i] = (0.0, 0.0, 0.0, 0.0)
            continue

        unique_pts = list(set(group["pts"]))
        val_parts = [f"({li},{lj})" for li, lj in unique_pts]
        val_clause = "VALUES " + ",".join(val_parts)

        hour_bounds = [int((dt.timestamp() - EPOCH_2025) / 3600) for dt in group["dts"]]
        hr_min = min(hour_bounds)
        hr_max = max(hour_bounds)

        duck_path = f.replace('\\', '/')
        query = f"""
            SELECT p.lat_idx, p.lon_idx, w.hour_rel,
                   w.wind_speed, w.wind_dir, w.current_u, w.current_v
            FROM ({val_clause}) AS p(lat_idx, lon_idx)
            INNER JOIN read_parquet('{duck_path}') w
              ON p.lat_idx = w.lat_idx AND p.lon_idx = w.lon_idx
            WHERE w.hour_rel BETWEEN {hr_min} AND {hr_max}
        """
        con = duckdb.connect()
        con.execute("SET temp_directory=''")
        try:
            rows = con.execute(query).fetchall()
        finally:
            con.close()

        lookup = {}
        for row in rows:
            key = (row[0], row[1], row[2])
            lookup[key] = (
                float(row[3]) / 100.0,
                float(row[4]) / 10.0,
                float(row[5]) / 1000.0,
                float(row[6]) / 1000.0,
            )

        for pt, dt, idx in zip(group["pts"], group["dts"], group["indices"]):
            hr = int((dt.timestamp() - EPOCH_2025) / 3600)
            key = (pt[0], pt[1], hr)
            val = lookup.get(key)
            if val is not None:
                result[idx] = val
            else:
                result[idx] = (0.0, 0.0, 0.0, 0.0)

    return result


def weather_and_current_at_points(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float, float, float]]:
    return _weather_and_current(points, datetimes)


def wind_at_points_hourly(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float]]:
    full = _weather_and_current(points, datetimes)
    return [(ws, wd) for ws, wd, _, _ in full]


def ensure_month_loaded(year: int, month: int) -> bool:
    return _file_for(year, month) is not None


def clear_cache():
    _cache.clear()
    _mtime_cache.clear()
    _bbox_cache.clear()


__all__ = [
    "weather_and_current_at_points",
    "wind_at_points_hourly",
    "ensure_month_loaded",
    "clear_cache",
    "preload_bounding_box",
]

