"""Build unified wind+current Parquet files for any month.

Reads:
  Wind: C:\app\data\hourly\weather_YYYY-MM_hourly.parquet (ERA5)
  Currents: C:\app\data\ocean_currents\currents_espc_YYYY-MM-DD.parquet (ESPC-D-V02)

Output:
  C:\app\data\unified\unified_YYYY-MM.parquet

Snap currents to wind grid (0.25°) using floor snapping:
  current_lat -> floor(current_lat * 4) / 4
  current_lon -> floor(current_lon * 4) / 4
"""
import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
from calendar import monthrange

WIND_DIR = Path(r"C:\app\data\hourly")
CURRENT_DIR = Path(r"C:\app\data\ocean_currents")
OUTPUT_DIR = Path(r"C:\app\data\unified")

def snap_to_wind_grid(df):
    """Floor snap to wind grid (0.25 degree increments)."""
    df["latitude"] = (np.floor(df["latitude"] * 4) / 4).round(2)
    df["longitude"] = (np.floor(df["longitude"] * 4) / 4).round(2)
    return df

def build_unified_month(year: int, month: int, corridor_only=False):
    """Build unified file for one month."""
    t0 = time.time()
    
    wind_file = WIND_DIR / f"weather_{year}-{month:02d}_hourly.parquet"
    if not wind_file.exists():
        raise FileNotFoundError(f"No wind data: {wind_file}")
    
    print(f"Loading wind from {wind_file} ...")
    wind = pd.read_parquet(wind_file)
    wind["time"] = pd.to_datetime(wind["time"])
    wind["hour"] = wind["time"].dt.floor("h")
    wind["latitude"] = wind["latitude"].round(2)
    wind["longitude"] = wind["longitude"].round(2)
    
    if corridor_only:
        mask = (
            (wind["latitude"] >= 30.0) & (wind["latitude"] <= 70.0) &
            (wind["longitude"] >= -90.0) & (wind["longitude"] <= 30.0)
        )
        wind = wind[mask]
    
    print(f"  Wind rows: {len(wind):,}")
    
    # Find all current files for this month
    ndays = monthrange(year, month)[1]
    current_files = []
    for d in range(1, ndays + 1):
        f = CURRENT_DIR / f"currents_espc_{year}-{month:02d}-{d:02d}.parquet"
        if f.exists():
            current_files.append(f)
    
    print(f"\nFound {len(current_files)} current file(s) for {year}-{month:02d}")
    
    if not current_files:
        print(f"  No current data: writing wind-only file")
    
    current_dfs = []
    for f in current_files:
        df = pd.read_parquet(f)
        df["time"] = pd.to_datetime(df["time"])
        df["hour"] = df["time"].dt.floor("h")
        df = snap_to_wind_grid(df)
        
        # Average duplicates
        grouped = df.groupby(["hour", "latitude", "longitude"]).agg({
            "current_u_ms": "mean",
            "current_v_ms": "mean",
        }).reset_index()
        current_dfs.append(grouped)
    
    if current_dfs:
        current = pd.concat(current_dfs, ignore_index=True)
        current = current.groupby(["hour", "latitude", "longitude"]).agg({
            "current_u_ms": "mean",
            "current_v_ms": "mean",
        }).reset_index()
        print(f"  Current rows after snap: {len(current):,}")
    else:
        current = pd.DataFrame(columns=["hour", "latitude", "longitude", "current_u_ms", "current_v_ms"])
    
    print("Merging ...")
    merged = wind.merge(current, on=["hour", "latitude", "longitude"], how="left")
    merged["current_u_ms"] = merged["current_u_ms"].fillna(0.0)
    merged["current_v_ms"] = merged["current_v_ms"].fillna(0.0)
    merged["current_speed_ms"] = np.sqrt(merged["current_u_ms"]**2 + merged["current_v_ms"]**2)
    
    output = merged[[
        "hour", "latitude", "longitude",
        "wind_speed_10m", "wind_direction_10m",
        "current_u_ms", "current_v_ms", "current_speed_ms"
    ]].rename(columns={"hour": "time"})
    
    output_file = OUTPUT_DIR / f"unified_{year}-{month:02d}.parquet"
    output.to_parquet(output_file, engine="pyarrow", compression="snappy")
    
    elapsed = time.time() - t0
    size_mb = output_file.stat().st_size / (1024 * 1024)
    non_zero = (output["current_speed_ms"] > 0).sum()
    
    print(f"\n{'='*70}")
    print(f"Saved: {output_file}")
    print(f"  Rows: {len(output):,}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Current rows: {non_zero:,} ({non_zero/len(output)*100:.0f}%)")
    print(f"  Max speed: {output['current_speed_ms'].max():.2f} m/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unified wind+current Parquet")
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("--corridor", action="store_true", help="Subset to transatlantic corridor")
    args = parser.parse_args()
    
    build_unified_month(args.year, args.month, args.corridor)
