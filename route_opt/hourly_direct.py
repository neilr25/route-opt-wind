"""Direct hourly weather reader for single-point lookup."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def _find_hourly_parquet(year: int, month: int) -> Path | None:
    p = Path(rf"C:\app\data\hourly\weather_{year}-{month:02d}_hourly.parquet")
    return p if p.exists() else None


def read_hourly_for_point(lat: float, lon: float, date_str: str) -> List[Tuple[float, float]]:
    """Read 24 hourly wind values for a point from hourly Parquet.
    
    Snaps to nearest 0.25° grid cell, then looks up all hours.
    """
    year = int(date_str[:4])
    month = int(date_str[5:7])
    p = _find_hourly_parquet(year, month)
    if not p:
        raise FileNotFoundError(f"No hourly data for {year}-{month:02d}")
    
    snap_lat = round(round(lat * 4) / 4, 2)
    snap_lon = round(round(lon * 4) / 4, 2)
    
    df = pd.read_parquet(p, columns=["time", "latitude", "longitude", "wind_speed_10m", "wind_direction_10m"])
    cell = df[(df["latitude"] == snap_lat) & (df["longitude"] == snap_lon)].sort_values("time")
    
    if cell.empty:
        # Try nearest neighbor within 1 degree
        cell = df[
            (abs(df["latitude"] - lat) <= 1.0) &
            (abs(df["longitude"] - lon) <= 1.0)
        ].copy()
        if cell.empty:
            raise ValueError(f"No hourly data near ({lat}, {lon})")
        # Pick nearest by great-circle approx
        cell["dlat"] = cell["latitude"] - lat
        cell["dlon"] = cell["longitude"] - lon
        cell["dist"] = cell["dlat"]**2 + cell["dlon"]**2
        nearest = cell.loc[cell["dist"].idxmin(), "latitude":"longitude"].values
        snap_lat, snap_lon = round(nearest[0], 2), round(nearest[1], 2)
        cell = df[(df["latitude"] == snap_lat) & (df["longitude"] == snap_lon)].sort_values("time")
    
    # Filter to requested date range
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = dt.replace(hour=0, minute=0, second=0)
    end = dt.replace(hour=23, minute=59, second=59)
    cell = cell[(cell["time"] >= start) & (cell["time"] <= end)]
    
    if cell.empty:
        raise ValueError(f"No hourly data for {date_str}")
    
    return [(float(r["wind_speed_10m"]), float(r["wind_direction_10m"])) for _, r in cell.iterrows()]
