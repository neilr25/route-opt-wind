"""Fast raw-to-npy converter using only vectorized NumPy.

Wind is already on 0.25deg grid. Currents are 4251x4500.
Process: snap current lat/lon to indices, np.add.at() to aggregate
multiple current points falling into same wind cell.
"""
import argparse, time, gc
from pathlib import Path
from calendar import monthrange
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr

WIND_DIR = Path(r"C:\app\data\hourly")
OUTPUT_DIR = Path(r"C:\app\data\corridor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NLAT, NLON = 721, 1440

# Precompute mapping at module level
wind_lats = np.arange(-90, 90.25, 0.25)
wind_lons = np.arange(-180, 180, 0.25)

# ESPC grid confirmed
espc_lats = np.linspace(-80, 90, 4251)
espc_lons = np.linspace(0, 360, 4501)[:-1]

lat_map = np.array([np.argmin(np.abs(espc_lats - wl)) for wl in wind_lats])
lon_map = np.array([np.argmin(np.abs(espc_lons - ((wl + 360) % 360))) for wl in wind_lons])

def convert_month(year: int, month: int):
    base = OUTPUT_DIR / f"corridor_{year}-{month:02d}"
    if Path(f"{base}_ws.npy").exists():
        print(f"SKIP {year}-{month:02d}")
        return

    t0 = time.time()
    ndays = monthrange(year, month)[1]
    nhours = ndays * 24

    # Wind (already on 0.25deg grid, directly indexed)
    ws = np.full((NLAT, NLON, nhours), np.nan, dtype=np.float32)
    wd = np.full((NLAT, NLON, nhours), np.nan, dtype=np.float32)
    cu = np.full((NLAT, NLON, nhours), np.nan, dtype=np.float32)
    cv = np.full((NLAT, NLON, nhours), np.nan, dtype=np.float32)

    # --- WIND ---
    wf = WIND_DIR / f"weather_{year}-{month:02d}_hourly.parquet"
    print(f"Wind: {wf.name}")
    pf = pq.ParquetFile(wf)
    for rg in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        h = np.clip((df['time'].dt.day - 1) * 24 + df['time'].dt.hour, 0, nhours - 1).values
        li = np.clip(np.round((df['latitude'].values + 90) / 0.25).astype(int), 0, NLAT - 1)
        lj = np.clip(np.round((df['longitude'].values + 180) / 0.25).astype(int), 0, NLON - 1)
        ws[li, lj, h] = df['wind_speed_10m'].values.astype(np.float32)
        wd[li, lj, h] = df['wind_direction_10m'].values.astype(np.float32)
        if rg % 200 == 0:
            print(f"  RG {rg}/{pf.metadata.num_row_groups}")
    del pf, df, li, lj, h
    gc.collect()

    # --- CURRENTS (DAP) ---
    print(f"Currents: {ndays} days via DAP")
    url_u = f"http://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z/{year}"
    url_v = f"http://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z/{year}"

    ds_u = xr.open_dataset(url_u, decode_times=False)
    ds_v = xr.open_dataset(url_v, decode_times=False)

    time_offset = (pd.Timestamp(year, month, 1) - pd.Timestamp(year, 1, 1)).days * 8

    for d in range(1, ndays + 1):
        day_start = time_offset + (d - 1) * 8
        u_block = ds_u.isel(time=slice(day_start, day_start + 8), depth=0)['water_u'].compute().values
        v_block = ds_v.isel(time=slice(day_start, day_start + 8), depth=0)['water_v'].compute().values

        for t_idx in range(8):
            # HYCOM day runs 12:00 UTC to 09:00 UTC next day
            # t_idx: 0=12, 1=15, 2=18, 3=21, 4=00, 5=03, 6=06, 7=09
            hour = (d - 1) * 24 + (t_idx * 3 + 12) % 24
            if hour + 3 > nhours:
                continue  # skip timesteps that spill past end of month
            # Nearest-neighbor mapping and broadcast to 3 hours
            cu[:, :, hour:hour+3] = u_block[t_idx][lat_map][:, lon_map][:, :, None]
            cv[:, :, hour:hour+3] = v_block[t_idx][lat_map][:, lon_map][:, :, None]

        if d % 5 == 0:
            print(f"  Day {d}/{ndays}")
            gc.collect()

    ds_u.close()
    ds_v.close()

    np.save(f"{base}_ws.npy", ws)
    np.save(f"{base}_wd.npy", wd)
    np.save(f"{base}_cu.npy", cu)
    np.save(f"{base}_cv.npy", cv)

    size = sum(Path(f"{base}_{k}.npy").stat().st_size / 1024**3 for k in ['ws','wd','cu','cv'])
    print(f"DONE: {size:.1f} GB in {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("year", type=int)
    p.add_argument("month", type=int)
    a = p.parse_args()
    convert_month(a.year, a.month)
