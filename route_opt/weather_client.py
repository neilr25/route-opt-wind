"""Fetch weather at waypoints — memory-efficient annual per-point cache.

Phase 1 optimisation: reads each monthly Parquet once and extracts ONLY
the ~488 unique ERA5 grid cells needed by the requested points.
Builds a per-point (n_points × n_days) numpy table directly —
no full 721×1440 grid ever loaded into memory.

Memory: ~500 cells × 365 days × 2 cols × 4 bytes ≈ 1.4 MB (vs 3 GB for full grid).
Speed: ~20s for first year load, then ~0.1 ms per date lookup.

Legacy DuckDB+cKDTree path retained for dashboard single-date calls.
"""

import time
from datetime import date as date_cls
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Legacy DuckDB+cKDTree (kept for single-date API calls from dashboard)
# ---------------------------------------------------------------------------
import duckdb

_weather_cache: dict[str, tuple] = {}
_CACHE_TTL = 300

_scipy_available: bool = False
try:
    from scipy.spatial import cKDTree
    _scipy_available = True
except Exception:
    pass


def _find_parquet(date: str) -> Optional[Path]:
    month = date[:7]
    clean = Path(r"C:\app\data") / f"weather_{month}_clean.parquet"
    raw = Path(r"C:\app\data") / f"weather_{month}.parquet"
    if clean.exists():
        return clean
    if raw.exists():
        return raw
    return None


def _load_con_for_month(date: str):
    global _weather_cache
    month = date[:7]
    now = time.time()
    if month in _weather_cache:
        con, loaded_at = _weather_cache[month]
        if now - loaded_at < _CACHE_TTL:
            return con
    p = _find_parquet(date)
    if not p:
        raise FileNotFoundError(f"No weather file for {date}")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW weather AS SELECT * FROM read_parquet('{p}')")
    _weather_cache[month] = (con, now)
    return con


# ---------------------------------------------------------------------------
# Phase 1: Memory-efficient per-point annual cache
# ---------------------------------------------------------------------------
# Each unique point maps to a single ERA5 grid cell (lat_idx, lon_idx).
# We load each monthly Parquet ONCE, extract only the cells we need,
# and accumulate into (n_points, n_days) arrays.

_N_LAT = 721       # -90 to 90 in 0.25° steps
_N_LON = 1440      # -180 to 179.75 in 0.25° steps

_point_year_cache: Dict[int, dict] = {}
# Structure: _point_year_cache[year] = {
#     "key":        tuple of points (for cache-hit check),
#     "ws":         np.float32 (n_points, n_days),
#     "wd":         np.float32 (n_points, n_days),
#     "dates":      list of datetime.date objects,
#     "day_index":  dict {date_obj: int}  for O(1) lookup,
# }


def _lat_lon_to_idx(lat: float, lon: float) -> Tuple[int, int]:
    """Convert lat/lon to ERA5 grid indices (clamped to valid range)."""
    lat_i = int(round((lat + 90.0) / 0.25))
    lon_i = int(round((lon + 180.0) / 0.25))
    return (
        max(0, min(_N_LAT - 1, lat_i)),
        max(0, min(_N_LON - 1, lon_i)),
    )


def _build_point_cache_direct(year: int, points: List[Tuple[float, float]]) -> dict:
    """Build (n_points x n_days) cache using DuckDB region-filtered extraction.

    Strategy: load each monthly Parquet via DuckDB with lat/lon range filters,
    then use vectorised pandas merge to extract values for all points at once.
    Benchmarked at ~1.2s per month (~15s total for 12 months).
    """
    n_pts = len(points)
    t0 = time.time()

    # Snap all points to nearest 0.25° grid cell for exact match
    snapped_points = []
    for lat, lon in points:
        snapped_lat = round(round(lat * 4) / 4, 2)
        snapped_lon = round(round(lon * 4) / 4, 2)
        snapped_points.append((snapped_lat, snapped_lon))

    # Build lookup DataFrame for our points
    pts_df = pd.DataFrame(snapped_points, columns=["latitude", "longitude"])
    pts_df["pt_idx"] = range(n_pts)

    # Compute corridor bounding box
    lat_min = float(pts_df["latitude"].min()) - 1.0
    lat_max = float(pts_df["latitude"].max()) + 1.0
    lon_min = float(pts_df["longitude"].min()) - 1.0
    lon_max = float(pts_df["longitude"].max()) + 1.0

    month_dfs: list = []

    for month in range(1, 13):
        p = Path(r"C:\app\data") / f"weather_{year}-{month:02d}_clean.parquet"
        if not p.exists():
            p = Path(r"C:\app\data") / f"weather_{year}-{month:02d}.parquet"
        if not p.exists():
            print(f"  [weather_client] WARNING: no parquet for {year}-{month:02d}")
            continue

        con = duckdb.connect()
        p_str = str(p).replace("\\", "/")
        con.execute(f"CREATE VIEW w AS SELECT * FROM read_parquet('{p_str}')")
        df = con.execute(
            f"SELECT time::date AS day, latitude, longitude, "
            f"wind_speed_10m, wind_direction_10m "
            f"FROM w "
            f"WHERE latitude BETWEEN {lat_min} AND {lat_max} "
            f"AND longitude BETWEEN {lon_min} AND {lon_max}"
        ).fetchdf()
        con.close()

        if not df.empty:
            # Round coords to 2dp for exact merge matching
            df["latitude"] = df["latitude"].round(2)
            df["longitude"] = df["longitude"].round(2)
            month_dfs.append(df)

    if not month_dfs:
        raise FileNotFoundError(f"No weather data found for {year}")

    # Combine all months
    all_data = pd.concat(month_dfs, ignore_index=True)
    all_data["day"] = pd.to_datetime(all_data["day"]).dt.date

    # Inner merge with our points — this gives us exactly the rows we need
    merged = all_data.merge(pts_df, on=["latitude", "longitude"], how="inner")

    # Pivot into (n_points, n_days) arrays
    dates = sorted(merged["day"].unique())
    day_index = {d: i for i, d in enumerate(dates)}
    n_days = len(dates)

    ws = np.zeros((n_pts, n_days), dtype=np.float32)
    wd = np.zeros((n_pts, n_days), dtype=np.float32)

    # Vectorised fill using the merged DataFrame
    pt_idxs = merged["pt_idx"].values
    day_idxs = np.array([day_index[d] for d in merged["day"]], dtype=np.int32)
    ws[pt_idxs, day_idxs] = merged["wind_speed_10m"].values.astype(np.float32)
    wd[pt_idxs, day_idxs] = merged["wind_direction_10m"].values.astype(np.float32)

    elapsed = time.time() - t0
    mem_mb = (ws.nbytes + wd.nbytes) / 1e6
    print(f"  [weather_client] Built {year} point cache: "
          f"{n_pts} pts x {n_days} days, {mem_mb:.1f} MB in {elapsed:.1f}s")

    result = {
        "key": tuple(points),
        "ws": ws,
        "wd": wd,
        "dates": dates,
        "day_index": day_index,
    }
    _point_year_cache[year] = result
    return result


def _get_point_cache(year: int, points: List[Tuple[float, float]]) -> Optional[dict]:
    """Get or build the per-point cache for a year.

    If the requested points are a subset of the cached points, return cache hit.
    If there are new points, rebuild the cache with all accumulated + new points.
    """
    requested = set(points)
    cache = _point_year_cache.get(year)
    if cache is not None:
        cached_pts = set(cache.get("key", []))
        if requested.issubset(cached_pts):
            # Cache hit — all requested points already cached
            return cache
        # Merge new points with cached points
        all_points = list(set(points) | cached_pts)
    else:
        all_points = list(set(points))
    return _build_point_cache_direct(year, all_points)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wind_at_points(
    points: List[Tuple[float, float]],
    date: str,
    db_path: str = "",
) -> List[Tuple[float, float]]:
    """
    Return [(wind_speed_ms, wind_direction_deg), ...] for each point.

    Phase 1 fast path: use pre-loaded annual per-point cache.
    Legacy fallback: DuckDB + cKDTree per-date lookup.
    """
    year = int(date[:4])

    # Try fast path
    cache = _get_point_cache(year, points)
    if cache is not None:
        parts = date.split("-")
        target = date_cls(int(parts[0]), int(parts[1]), int(parts[2]))
        day_idx = cache["day_index"].get(target, -1)
        if day_idx >= 0:
            # Map requested points to cache indices
            cache_key_list = list(cache["key"])
            cache_pt_map = {pt: i for i, pt in enumerate(cache_key_list)}
            result = []
            for pt in points:
                ci = cache_pt_map.get(pt, -1)
                if ci >= 0 and ci < cache["ws"].shape[0]:
                    result.append((float(cache["ws"][ci, day_idx]), float(cache["wd"][ci, day_idx])))
                else:
                    result.append((0.0, 0.0))
            return result

    # Legacy fallback
    con = _load_con_for_month(date)
    df = con.execute(
        f"SELECT latitude, longitude, wind_speed_10m, wind_direction_10m "
        f"FROM weather WHERE time::date = '{date}'"
    ).fetchdf()

    if df.empty:
        raise ValueError(f"No weather rows for {date}")

    if _scipy_available and len(points) > 50:
        tree = cKDTree(df[["latitude", "longitude"]].values)
        pts_arr = pd.DataFrame(points, columns=["lat", "lon"]).values
        _, idxs = tree.query(pts_arr, k=1)
        rows = df.iloc[idxs]
        return list(zip(rows["wind_speed_10m"].values, rows["wind_direction_10m"].values))

    results: List[Tuple[float, float]] = []
    for lat, lon in points:
        df["dist"] = (df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2
        row = df.loc[df["dist"].idxmin()]
        results.append((float(row["wind_speed_10m"]), float(row["wind_direction_10m"])))
    return results


def preload_year(year: int, sample_points: Optional[List[Tuple[float, float]]] = None) -> None:
    """Pre-load an entire year of weather into memory (call before batch)."""
    print(f"[weather_client] Pre-loading year {year}...")
    if sample_points:
        _get_point_cache(year, sample_points)
    print(f"[weather_client] Year {year} ready.")


def invalidate_cache():
    """Close all cached DuckDB connections and clear year grids."""
    global _point_year_cache, _weather_cache
    for con, _ in _weather_cache.values():
        con.close()
    _weather_cache.clear()
    _point_year_cache.clear()