"""Ocean current lookup for SOG calculations.

Reads ocean current data from Parquet files stored in C:\app\data\ocean_currents.
Provides current velocity (u, v) components at any lat/lon/time point.

Data source (when available): HYCOM GLBy0.08 or CMEMS MULTIOBS.
Format expected:
    time (datetime), latitude, longitude,
    current_u_ms, current_v_ms, current_speed_ms
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Parquet file cache: keyed by (year, month, day)
# ---------------------------------------------------------------------------
_current_cache: Dict[Tuple[int, int, int], pd.DataFrame] = {}

DATA_DIR = Path(r"C:\app\data\ocean_currents")


def _find_current_files(year: int, month: int, day: int) -> List[Path]:
    """Return all Parquet paths for a given date (handles split east/west files)."""
    files = []
    # ESPC files first (new format)
    espc_stems = [
        f"currents_espc_{year}-{month:02d}-{day:02d}",
        f"currents_espc_{year}_{month:02d}_{day:02d}"
    ]
    for stem in espc_stems:
        p = DATA_DIR / f"{stem}.parquet"
        if p.exists():
            files.append(p)
            return files  # ESPC file found
    # Legacy combined files
    for stem in [f"currents_{year}-{month:02d}-{day:02d}", f"currents_{year}_{month:02d}_{day:02d}"]:
        p = DATA_DIR / f"{stem}.parquet"
        if p.exists():
            files.append(p)
            return files  # Combined file found
    # Split east / west files
    for suffix in ["_e", "_w"]:
        p = DATA_DIR / f"currents_{year}-{month:02d}-{day:02d}{suffix}.parquet"
        if p.exists():
            files.append(p)
    return files


def _load_currents(year: int, month: int, day: int) -> Optional[pd.DataFrame]:
    """Load current data for a single day into memory (handles split files)."""
    files = _find_current_files(year, month, day)
    if not files:
        print(f"  [current] No data for {year}-{month:02d}-{day:02d}")
        return None

    t0 = time.time()
    dfs = []
    for p in files:
        df = pd.read_parquet(p)
        if "current_u_ms" not in df.columns or "current_v_ms" not in df.columns:
            print(f"  [current] Missing current columns in {p.name}")
            continue
        dfs.append(df)
    
    if not dfs:
        return None
    
    df = pd.concat(dfs, ignore_index=True)

    # Round coordinates to match ERA5 grid for easy lookup
    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)

    # Parse time
    if "time" in df.columns:
        df["hour"] = pd.to_datetime(df["time"]).dt.floor("h")
    else:
        print(f"  [current] No time column")
        return None

    key = (year, month, day)
    _current_cache[key] = df

    print(f"  [current] Loaded {year}-{month:02d}-{day:02d}: {len(df):,} rows "
          f"in {time.time()-t0:.1f}s")
    return df


def current_at_points(
    points: List[Tuple[float, float]],
    datetimes: List[datetime],
) -> List[Tuple[float, float]]:
    """Return [(current_u_ms, current_v_ms), ...] for each point/datetime pair.

    Uses nearest-neighbor lookup (no grid snapping needed).
    Returns (0.0, 0.0) if no data available.
    """
    if not points or not datetimes:
        return []
    if len(points) == 1 and len(datetimes) > 1:
        points = points * len(datetimes)

    # Group points by date
    date_groups: Dict[Tuple[int, int, int], dict] = {}
    for i, (pt, dt) in enumerate(zip(points, datetimes)):
        key = (dt.year, dt.month, dt.day)
        date_groups.setdefault(key, {"indices": [], "pts": [], "dts": []})
        date_groups[key]["indices"].append(i)
        date_groups[key]["pts"].append(pt)
        date_groups[key]["dts"].append(dt.replace(minute=0, second=0, microsecond=0))

    result = [(0.0, 0.0)] * len(datetimes)

    for (year, month, day), group in date_groups.items():
        key = (year, month, day)
        df = _current_cache.get(key)
        if df is None:
            df = _load_currents(year, month, day)
            if df is None:
                continue

        # Fast nearest-neighbor lookup
        for i, ((lat, lon), dt) in enumerate(zip(group["pts"], group["dts"])):
            # Filter by time first
            hour_df = df[df["hour"] == dt]
            if hour_df.empty:
                continue

            # Find nearest by lat/lon
            hour_df = hour_df.copy()
            hour_df["dist"] = (hour_df["latitude"] - lat)**2 + (hour_df["longitude"] - lon)**2
            nearest = hour_df.nsmallest(1, "dist")
            if nearest.empty:
                continue

            idx = group["indices"][i]
            u = float(nearest["current_u_ms"].values[0]) if pd.notna(nearest["current_u_ms"].values[0]) else 0.0
            v = float(nearest["current_v_ms"].values[0]) if pd.notna(nearest["current_v_ms"].values[0]) else 0.0
            result[idx] = (u, v)

    return result


def clear_cache() -> None:
    _current_cache.clear()
