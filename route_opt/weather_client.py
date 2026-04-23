"""Fetch weather at waypoints — DuckDB-native Parquet lookup.

Uses DuckDB for WHERE-pushdown Parquet reads (2–3 s/month vs 80–140 s with pandas).
DataFrame is used for in-memory nearest-neighbour only after DuckDB has filtered to the day.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

# Global cache: {month_str: (DuckDB con, loaded_at)} — one con per month
_weather_cache: dict[str, tuple[duckdb.DuckDBPyConnection, float]] = {}
_CACHE_TTL = 300

# Optional cKDTree — falls back to DataFrame scan if unavailable
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


def _load_con_for_month(date: str) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection with the month-Parquet already registered."""
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


def wind_at_points(
    points: List[Tuple[float, float]],
    date: str,
    db_path: str = "",
) -> List[Tuple[float, float]]:
    """
    Return [(wind_speed_ms, wind_direction_deg), ...] for each point.
    Uses DuckDB WHERE-pushdown to read only the target date, then
    vectorised nearest-neighbour on the resulting DataFrame.
    NOTE: ERA5 wind_speed_10m is in m/s — do not divide by 3.6 downstream.
    """
    con = _load_con_for_month(date)
    df = con.execute(
        f"SELECT latitude, longitude, wind_speed_10m, wind_direction_10m "
        f"FROM weather WHERE time::date = '{date}'"
    ).fetchdf()

    if df.empty:
        raise ValueError(f"No weather rows for {date}")

    if _scipy_available and len(points) > 50:
        # Fast path: build a cKDTree once, batch-query all points
        tree = cKDTree(df[["latitude", "longitude"]].values)
        pts_arr = pd.DataFrame(points, columns=["lat", "lon"]).values
        _, idxs = tree.query(pts_arr, k=1)
        rows = df.iloc[idxs]
        return list(zip(rows["wind_speed_10m"].values, rows["wind_direction_10m"].values))

    # Fallback for small point counts or missing scipy
    results: List[Tuple[float, float]] = []
    for lat, lon in points:
        df["dist"] = (df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2
        row = df.loc[df["dist"].idxmin()]
        results.append((float(row["wind_speed_10m"]), float(row["wind_direction_10m"])))
    return results

def invalidate_cache():
    """Close all cached DuckDB connections."""
    global _weather_cache
    for con, _ in _weather_cache.values():
        con.close()
    _weather_cache.clear()
