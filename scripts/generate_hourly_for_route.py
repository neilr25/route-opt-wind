"""Create realistic hourly weather for mesh/baseline points from daily data.

Extracts daily weather for all unique mesh+baseline points, then interpolates
hourly values between consecutive days using wind-component linear interpolation.
Output: a small Parquet (~200K rows for ~400 pts x 30 days x 24h).
"""

import math
from pathlib import Path

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


def generate_hourly_for_route(year, month, mesh_points, baseline_points, daily_parquet_path, output_dir):
    """Generate hourly Parquet for mesh+baseline points."""
    print(f"Reading daily data from {daily_parquet_path}...")
    all_points = list(set(mesh_points + baseline_points))
    df = pd.read_parquet(daily_parquet_path, columns=["time", "latitude", "longitude", "wind_speed_10m", "wind_direction_10m"])
    
    # Filter to our corridor region
    lats = [p[0] for p in all_points]
    lons = [p[1] for p in all_points]
    lat_min, lat_max = min(lats) - 1, max(lats) + 1
    lon_min, lon_max = min(lons) - 1, max(lons) + 1
    
    df = df[
        (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max) &
        (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
    ]
    
    # Round coords for exact matching
    df["latitude"] = df["latitude"].round(2)
    df["longitude"] = df["longitude"].round(2)
    
    # Snap points to grid
    snapped = {p: (round(round(p[0]*4)/4, 2), round(round(p[1]*4)/4, 2)) for p in all_points}
    unique_snapped = list(set(snapped.values()))
    
    # Build per-point daily series
    hourly_rows = []
    for lat, lon in unique_snapped:
        cell_df = df[(df["latitude"] == lat) & (df["longitude"] == lon)].sort_values("time")
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
        
        # Last day: repeat last day's values for 24h
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"weather_{year}-{month:02d}_hourly.parquet"
    hourly.to_parquet(out, engine="pyarrow", compression="snappy")
    print(f"Saved {out}: {len(hourly)} rows, {out.stat().st_size/1024/1024:.1f} MB")
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from route_opt.baseline import baseline_route
    from route_opt.mesh import corridor_graph
    
    baseline = baseline_route((51.9244, 4.4777), (40.7128, -74.0060))
    G = corridor_graph(baseline)
    
    mesh_points = [(d["lat"], d["lon"]) for _, d in G.nodes(data=True)]
    baseline_points = baseline
    
    generate_hourly_for_route(
        year=2025, month=6,
        mesh_points=mesh_points, baseline_points=baseline_points,
        daily_parquet_path=r"C:\app\data\weather_2025-06.parquet",
        output_dir=r"C:\app\data\hourly",
    )
