"""Create realistic synthetic hourly weather from daily data.

Reads daily Parquet, then for each grid cell interpolates between consecutive
days at hourly intervals using linear interpolation on wind components,
then resaves as hourly Parquet files.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _interp_wind_components(wd_prev, ws_prev, wd_next, ws_next, frac):
    """Linearly interpolate wind speed and direction components."""
    # Avoid wrapping issues by interpolating u/v instead of angle
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


def daily_to_hourly(daily_parquet: Path, year: int, month: int, output_dir: Path) -> Path:
    """Convert daily regional Parquet to synthetic hourly."""
    print(f"Converting {daily_parquet} -> hourly...")
    df = pd.read_parquet(daily_parquet, columns=["time", "latitude", "longitude", "wind_speed_10m", "wind_direction_10m"])

    # Ensure regional subset (daily files may be full global)
    df = df[
        (df["latitude"] >= 29) & (df["latitude"] <= 61) &
        (df["longitude"] >= -81) & (df["longitude"] <= 11)
    ]

    # Round grid coords
    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)

    unique_lats = sorted(df["latitude"].unique())[:10]  # DEBUG: only 10 lats
    unique_lons = sorted(df["longitude"].unique())[:10]   # DEBUG: only 10 lons

    # Subset
    df = df[df["latitude"].isin(unique_lats) & df["longitude"].isin(unique_lons)]

    # Build per-cell interpolation
    hourly_rows = []
    grouped = df.groupby(["latitude", "longitude"])

    for (lat, lon), cell_df in grouped:
        cell_df = cell_df.sort_values("time").reset_index(drop=True)
        if len(cell_df) < 2:
            # Only one day — repeat for all hours
            row = cell_df.iloc[0]
            for h in range(24):
                t = row["time"].replace(hour=h, minute=0, second=0)
                hourly_rows.append({
                    "time": t,
                    "latitude": lat,
                    "longitude": lon,
                    "wind_speed_10m": row["wind_speed_10m"],
                    "wind_direction_10m": row["wind_direction_10m"],
                })
            continue

        for i in range(len(cell_df) - 1):
            t_prev = cell_df.iloc[i]["time"]
            t_next = cell_df.iloc[i + 1]["time"]
            ws_prev = float(cell_df.iloc[i]["wind_speed_10m"])
            ws_next = float(cell_df.iloc[i + 1]["wind_speed_10m"])
            wd_prev = float(cell_df.iloc[i]["wind_direction_10m"])
            wd_next = float(cell_df.iloc[i + 1]["wind_direction_10m"])

            # Generate hourly between them
            nhours = int((t_next - t_prev).total_seconds() / 3600)
            for h in range(nhours):
                frac = h / nhours if nhours > 0 else 0
                t = t_prev + pd.Timedelta(hours=h)
                ws, wd = _interp_wind_components(wd_prev, ws_prev, wd_next, ws_next, frac)
                hourly_rows.append({
                    "time": t,
                    "latitude": lat,
                    "longitude": lon,
                    "wind_speed_10m": round(ws, 2),
                    "wind_direction_10m": round(wd, 1),
                })

        # Last day: repeat last day's values
        last = cell_df.iloc[-1]
        for h in range(24):
            t = last["time"].replace(hour=h, minute=0, second=0)
            hourly_rows.append({
                "time": t,
                "latitude": lat,
                "longitude": lon,
                "wind_speed_10m": float(last["wind_speed_10m"]),
                "wind_direction_10m": float(last["wind_direction_10m"]),
            })

    hourly = pd.DataFrame(hourly_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"weather_{year}-{month:02d}_hourly.parquet"
    hourly.to_parquet(output_file, engine="pyarrow", compression="snappy")
    print(f"Hourly file: {output_file} ({len(hourly)} rows, {output_file.stat().st_size/1024/1024:.1f} MB)")
    return output_file


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", required=True, help="Path to daily Parquet")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--output", default=r"C:\app\data\hourly")
    args = parser.parse_args()
    daily_to_hourly(Path(args.daily), args.year, args.month, Path(args.output))
