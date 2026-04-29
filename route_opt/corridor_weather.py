"""Grid-native weather lookup — O(1) direct array index.

Loads corridor_YYYY-MM_{ws,wd,cu,cv}.npy files (single month per set).
Arrays are [lat, lon, hour] shaped, matching ERA5 0.25deg grid.
Currents are already snapped to same grid at build time.

Lookup: snap to grid -> array index -> direct read.
"""
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DATA_DIR = Path(r"C:\app\data\corridor")

# In-memory cache keyed by (year, month)
_month_cache: Dict[Tuple[int, int], dict] = {}

LAT_MIN, LAT_STEP = -90.0, 0.25
LON_MIN, LON_STEP = -180.0, 0.25


def _load_month(year: int, month: int) -> Optional[dict]:
    """Load a month's .npy files into memory-mapped arrays."""
    key = (year, month)
    if key in _month_cache:
        return _month_cache[key]

    base = DATA_DIR / f"corridor_{year}-{month:02d}"
    files = {
        'ws': Path(f"{base}_ws.npy"),
        'wd': Path(f"{base}_wd.npy"),
        'cu': Path(f"{base}_cu.npy"),
        'cv': Path(f"{base}_cv.npy"),
    }
    if not all(f.exists() for f in files.values()):
        return None

    t0 = time.time()
    cache = {
        "ws": np.load(files['ws'], mmap_mode='r'),
        "wd": np.load(files['wd'], mmap_mode='r'),
        "cu": np.load(files['cu'], mmap_mode='r'),
        "cv": np.load(files['cv'], mmap_mode='r'),
        "lat_min": LAT_MIN,
        "lat_step": LAT_STEP,
        "lon_min": LON_MIN,
        "lon_step": LON_STEP,
        "n_lat": 721,
        "n_lon": 1440,
    }
    _month_cache[key] = cache
    print(f"  [corridor] Loaded {year}-{month:02d}: mmap in {time.time()-t0:.2f}s")
    return cache


def _snap_lat(lat: float) -> int:
    return int(np.clip(np.round((lat - LAT_MIN) / LAT_STEP), 0, 720))


def _snap_lon(lon: float) -> int:
    return int(np.clip(np.round((lon - LON_MIN) / LON_STEP), 0, 1439))


def _hour_index(dt: datetime) -> int:
    return (dt.day - 1) * 24 + dt.hour


def weather_at_point(lat: float, lon: float, dt: datetime) -> Tuple[float, float, float, float]:
    """Return (wind_speed_ms, wind_dir_deg, current_u_ms, current_v_ms)."""
    cache = _load_month(dt.year, dt.month)
    if cache is None:
        return (0.0, 0.0, 0.0, 0.0)

    li = _snap_lat(lat)
    lj = _snap_lon(lon)
    hr = _hour_index(dt)

    if hr < 0 or hr >= cache["ws"].shape[2]:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        float(cache["ws"][li, lj, hr]),
        float(cache["wd"][li, lj, hr]),
        float(cache["cu"][li, lj, hr]),
        float(cache["cv"][li, lj, hr]),
    )


def weather_and_current_at_points(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float, float, float]]:
    if not points or not datetimes:
        return []
    if len(points) == 1 and len(datetimes) > 1:
        points = points * len(datetimes)
    return [weather_at_point(pt[0], pt[1], dt) for pt, dt in zip(points, datetimes)]


def wind_at_points_hourly(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float]]:
    full = weather_and_current_at_points(points, datetimes)
    return [(ws, wd) for ws, wd, _, _ in full]


def ensure_month_loaded(year: int, month: int) -> bool:
    return _load_month(year, month) is not None


def clear_cache():
    _month_cache.clear()
