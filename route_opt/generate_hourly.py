"""Generate hourly weather Parquet for a bounding box from daily data.

This enables on-the-fly ATOBVIAC scenarios: for any route we compute the
bbox, generate hourly for that region from daily data, then cache it.
"""

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _interp_wind_components(wd_prev, ws_prev, wd_next, ws_next, frac):
    """Interpolate wind via u/v components to avoid wrapping."""
    rad_prev = math.radians(wd_prev)
    rad_next = math.radians(wd_next)
    u_prev = ws_prev * math.cos(rad_prev)
    v_prev = ws_prev * math.sin(rad_prev)
    u_next = ws_next * math.cos(rad_next)
    v_next = ws_next * math.sin(rad_next)
    u = u_prev + frac * (u_next - u_prev)
    v = v_prev + frac * (v_next - v_prev)
    ws = math.sqrt(u**2 + v**2)
    wd = math.degrees(math.atan2(v, u)) % 360
    return ws, wd


def generate_hourly_for_bbox(
    year: int, month: int,
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    daily_dir: Path = Path(r"C:\app\data"),
    output_dir: Path = Path(r"C:\app\data\hourly"),
) -> Path:
    """Generate hourly Parquet for a bbox from daily data."""
    daily_file = daily_dir / f"weather_{year}-{month:02d}.parquet"
    if not daily_file.exists():
        daily_file = daily_dir / f"weather_{year}-{month:02d}_clean.parquet"
    if not daily_file.exists():
        raise FileNotFoundError(f"No daily data for {year}-{month:02d}")

    print(f"[generate_hourly] Reading {daily_file}...")
    df = pd.read_parquet(daily_file, columns=["time", "latitude", "longitude", "wind_speed_10m", "wind_direction_10m"])

    # Filter to bbox + 1 degree margin
    df = df[
        (df["latitude"] >= lat_min - 1) & (df["latitude"] <= lat_max + 1) &
        (df["longitude"] >= lon_min - 1) & (df["longitude"] <= lon_max + 1)
    ]

    if df.empty:
        raise ValueError(f"No weather data in bbox")

    # Round to grid
    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)

    hourly_rows = []
    grouped = df.groupby(["latitude", "longitude"])

    for (lat, lon), cell_df in grouped:
        cell_df = cell_df.sort_values("time").reset_index(drop=True)
        if len(cell_df) < 2:
            continue

        # Interpolate hourly between days
        for i in range(len(cell_df) - 1):
            t_prev = cell_df.iloc[i]["time"]
            t_next = cell_df.iloc[i + 1]["time"]
            ws_prev = float(cell_df.iloc[i]["wind_speed_10m"])
            ws_next = float(cell_df.iloc[i + 1]["wind_speed_10m"])
            wd_prev = float(cell_df.iloc[i]["wind_direction_10m"])
            wd_next = float(cell_df.iloc[i + 1]["wind_direction_10m"])

            nhours = max(1, int((t_next - t_prev).total_seconds() / 3600))
            for h in range(nhours):
                frac = h / nhours
                t = t_prev + pd.Timedelta(hours=h)
                ws, wd = _interp_wind_components(wd_prev, ws_prev, wd_next, ws_next, frac)
                hourly_rows.append({
                    "time": t,
                    "latitude": lat,
                    "longitude": lon,
                    "wind_speed_10m": round(ws, 2),
                    "wind_direction_10m": round(wd, 1),
                })

        # Pad last 24h
        last = cell_df.iloc[-1]
        base_t = last["time"].replace(hour=0, minute=0, second=0)
        for h in range(24):
            t = base_t + pd.Timedelta(hours=h)
            hourly_rows.append({
                "time": t,
                "latitude": lat,
                "longitude": lon,
                "wind_speed_10m": float(last["wind_speed_10m"]),
                "wind_direction_10m": float(last["wind_direction_10m"]),
            })

    hourly = pd.DataFrame(hourly_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"weather_{year}-{month:02d}_hourly.parquet"
    hourly.to_parquet(out, engine="pyarrow", compression="snappy")
    print(f"[generate_hourly] Saved {out}: {len(hourly)} rows, {out.stat().st_size/1024/1024:.1f} MB")
    return out


def ensure_hourly_for_bbox(year: int, month: int, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """Ensure hourly file exists for bbox; generate if missing."""
    out = Path(rf"C:\app\data\hourly\weather_{year}-{month:02d}_hourly.parquet")
    if out.exists():
        return out
    return generate_hourly_for_bbox(year, month, lat_min, lat_max, lon_min, lon_max)
