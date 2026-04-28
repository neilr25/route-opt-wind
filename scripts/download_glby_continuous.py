"""Continuous GLBy downloader — no timeouts, survives restarts.

Usage: python download_glby_continuous.py

Runs indefinitely downloading ALL missing months. No background agents.
- Natural resume: skips existing parquet files automatically
- No subprocess timeouts (allows each day to take however long needed)
- Logs every download with timestamp
- Prints progress every 5 days so you can watch it live
- If killed: just run it again, it'll pick up where it left off

Estimated: ~1 month per 50 minutes
15 remaining months = ~12 hours total
"""
import sys, time, traceback
from datetime import datetime
from ftplib import FTP
from pathlib import Path
import netCDF4
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

OUTPUT = Path(r"C:\app\data\ocean_currents")
OUTPUT.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path(r"C:\Projects\route-opt\continuous_download.log")

# All incomplete months (we'll auto-skip complete ones)
TARGETS = [
    (2024, 2), (2024, 3), (2024, 5), (2024, 6),
    (2024, 7), (2024, 8), (2024, 9),
    (2023, 2), (2023, 3), (2023, 4), (2023, 5), (2023, 6),
    (2023, 7), (2023, 8), (2023, 9), (2023, 11), (2023, 12),
]

def log(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def month_complete(year, month) -> bool:
    import calendar
    ndays = calendar.monthrange(year, month)[1]
    have = len(list(OUTPUT.glob(f"currents_espc_{year}-{month:02d}-*.parquet")))
    return have >= ndays

def download_single_day(date_str: str) -> bool:
    """Download one day using the same logic as download_glby_nc4.py."""
    import netCDF4, numpy as np, pandas as pd
    
    f_out = OUTPUT / f"currents_espc_{date_str}.parquet"
    if f_out.exists():
        return True
    
    url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
    
    try:
        dt = pd.Timestamp(date_str)
        base = pd.Timestamp("2000-01-01")
        target_h = (dt - base).total_seconds() / 3600
        
        nc = netCDF4.Dataset(url)
        time_arr = np.array(nc.variables["time"][:])
        lat = np.array(nc.variables["lat"][:])
        lon = np.array(nc.variables["lon"][:])
        
        idx = int(np.argmin(np.abs(time_arr - target_h)))
        actual = time_arr[idx]
        if abs(actual - target_h) > 3:
            nc.close()
            return False
        
        steps = []
        for i in range(8):
            t_idx = idx + i
            u_raw = nc.variables["water_u"][t_idx, 0, :, :]
            v_raw = nc.variables["water_v"][t_idx, 0, :, :]
            
            u = np.ma.filled(u_raw, np.nan)
            v = np.ma.filled(v_raw, np.nan)
            
            lat_all = np.repeat(lat, len(lon))
            lon_all = np.tile(lon, len(lat))
            df = pd.DataFrame({
                "lat": lat_all,
                "lon": lon_all,
                "u": u.ravel(),
                "v": v.ravel(),
            })
            df = df.dropna(subset=["u", "v"])
            df = df[(df["u"].abs() <= 5) & (df["v"].abs() <= 5)]
            df["lon"] = df["lon"].apply(lambda x: x - 360 if x > 180 else x)
            df["latitude"] = (np.floor(df["lat"] * 4) / 4).round(2)
            df["longitude"] = (np.floor(df["lon"] * 4) / 4).round(2)
            
            grouped = df.groupby(["latitude", "longitude"]).agg({
                "u": "mean", "v": "mean"
            }).reset_index()
            
            nc_time = base + pd.Timedelta(hours=float(time_arr[t_idx]))
            grouped["time"] = nc_time
            grouped = grouped.rename(columns={"u": "current_u_ms", "v": "current_v_ms"})
            grouped["current_speed_ms"] = np.sqrt(grouped["current_u_ms"]**2 + grouped["current_v_ms"]**2)
            
            steps.append(grouped[["time", "latitude", "longitude", "current_u_ms", "current_v_ms", "current_speed_ms"]])
        
        nc.close()
        
        full = pd.concat(steps, ignore_index=True)
        full.to_parquet(f_out, engine="pyarrow", compression="snappy")
        return True
        
    except Exception as e:
        log(f"ERROR {date_str}: {e}")
        return False

def main():
    log("=== Starting continuous downloader ===")
    
    for year, month in TARGETS:
        if month_complete(year, month):
            log(f"SKIP {year}-{month:02d}: complete")
            continue
        
        import calendar
        ndays = calendar.monthrange(year, month)[1]
        log(f"DOWNLOAD {year}-{month:02d}: {ndays} days")
        
        ok = fail = 0
        for d in range(1, ndays + 1):
            ds = f"{year}-{month:02d}-{d:02d}"
            f = OUTPUT / f"currents_espc_{ds}.parquet"
            if f.exists():
                continue
            
            t0 = time.time()
            success = download_single_day(ds)
            elapsed = time.time() - t0
            
            if success:
                ok += 1
                log(f"  OK {ds} ({elapsed:.0f}s)")
            else:
                fail += 1
                log(f"  FAIL {ds}")
            
            total = ok + fail
            if total % 5 == 0:
                log(f"  PROGRESS: {total}/{ndays} done, {ok} OK, {fail} fail")
        
        log(f"DONE {year}-{month:02d}: {ok} OK, {fail} fail")
    
    log("=== All targets processed ===")

if __name__ == "__main__":
    main()
