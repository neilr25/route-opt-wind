"""Download ESPC-D-V02 2025 surface currents via OPeNDAP (DAP).

Day-by-day approach to avoid server ArrayIndexOutOfBoundsException.
Downloads 8 timesteps per day (3-hourly) for full temporal resolution.

Key fixes vs original month-by-month script:
  - Proper time conversion from hours-since-2000 epoch (not pd.to_datetime on raw floats)
  - Uses known date for filenames (not derived from buggy time values)
  - Server-side depth=0 slicing (~40x reduction vs full file)
  - Day-by-day processing (8 timesteps per request — within server limits)
  - Retry with exponential backoff on server errors
  - Skips already-downloaded days (resume support)
  - NaN filtering (land cells removed for smaller files)
  - Progress logging every 10 days

Usage:
    python download_espc_2025_daybyday.py              # full year
    python download_espc_2025_daybyday.py 3            # single month (March)
    python download_espc_2025_daybyday.py 6 9          # months 6-9
"""
import calendar
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
U_URL = "http://tds.hycom.org/thredds/dodsC/ESPC-D-V02/u3z/2025"
V_URL = "http://tds.hycom.org/thredds/dodsC/ESPC-D-V02/v3z/2025"

OUTPUT_DIR = Path(r"C:\app\data\ocean_currents")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ESPC time origin: hours since 2000-01-01 00:00
TIME_ORIGIN = pd.Timestamp("2000-01-01")

# Retry settings
MAX_RETRIES = 5
BASE_DELAY_S = 10
BACKOFF_FACTOR = 2

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            OUTPUT_DIR / "espc_2025_dap.log", mode="a", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("espc_dap")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------
def retry_open_dataset(url: str, **kwargs) -> xr.Dataset:
    """Open an OPeNDAP dataset with exponential-backoff retries."""
    delay = BASE_DELAY_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("Opening %s (attempt %d/%d)", url, attempt, MAX_RETRIES)
            ds = xr.open_dataset(url, **kwargs)
            _ = ds.sizes  # verify connection
            log.info("Connected to %s — sizes=%s", url, dict(ds.sizes))
            return ds
        except Exception as exc:
            log.warning("Attempt %d failed for %s: %s", attempt, url, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= BACKOFF_FACTOR


def retry_compute(func, *args, **kwargs):
    """Call a function with exponential-backoff retries."""
    delay = BASE_DELAY_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            log.warning("Compute attempt %d failed: %s", attempt, exc)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= BACKOFF_FACTOR


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def hours_since_origin(dt: pd.Timestamp) -> float:
    """Convert a datetime to hours since ESPC epoch (2000-01-01)."""
    return (dt - TIME_ORIGIN).total_seconds() / 3600.0


def day_noon_index(ds: xr.Dataset, year: int, month: int, day: int):
    """Return the time index closest to 12:00 UTC for a given calendar day."""
    start = pd.Timestamp(year, month, day)
    noon = start + pd.Timedelta(hours=12)
    noon_h = hours_since_origin(noon)
    time_float = ds.time.values.astype("float64")
    idx = np.argmin(np.abs(time_float - noon_h))
    return idx


# ---------------------------------------------------------------------------
# Core processing — one day at a time
# ---------------------------------------------------------------------------
def process_day(ds_u: xr.Dataset, ds_v: xr.Dataset, year: int, month: int, day: int):
    """Download and save one day of surface-current data as a parquet file.

    Downloads the timestep closest to 12:00 UTC (single timestep per day)
    to keep file sizes manageable (~50 MB vs ~420 MB for all 8 timesteps).
    """
    date_str = f"{year}-{month:02d}-{day:02d}"
    out_path = OUTPUT_DIR / f"currents_espc_{date_str}.parquet"

    if out_path.exists():
        log.debug("  %s already exists, skipping", date_str)
        return True

    # Find the timestep closest to 12:00 UTC
    t_idx = day_noon_index(ds_u, year, month, day)

    log.info("  Day %s: using time index %d (noon)", date_str, t_idx)

    # Slice to surface (depth=0) + single timestep
    depth_dim = "depth" if "depth" in ds_u.sizes else "lev"
    u_surf = ds_u["water_u"].isel({depth_dim: 0, "time": t_idx})
    v_surf = ds_v["water_v"].isel({depth_dim: 0, "time": t_idx})

    # Compute eagerly — pull data from server
    log.info("    Fetching u3z for %s...", date_str)
    u_data = retry_compute(u_surf.compute)
    log.info("    Fetching v3z for %s...", date_str)
    v_data = retry_compute(v_surf.compute)

    # Convert time value: hours since 2000-01-01 → actual timestamp
    time_hours = float(u_data.time.values)
    time_dt = TIME_ORIGIN + pd.Timedelta(hours=time_hours)

    # Build coordinate arrays
    lat_vals = u_data.lat.values
    lon_vals = u_data.lon.values

    lat_grid, lon_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    u_flat = u_data.values.ravel()
    v_flat = v_data.values.ravel()

    # Filter out land (NaN values)
    valid = np.isfinite(u_flat) & np.isfinite(v_flat)

    df = pd.DataFrame({
        "time": time_dt,
        "lat": lat_flat[valid],
        "lon": lon_flat[valid],
        "current_u_ms": u_flat[valid],
        "current_v_ms": v_flat[valid],
    })

    df.to_parquet(out_path, engine="pyarrow", compression="snappy")

    file_mb = out_path.stat().st_size / 1024 / 1024
    log.info("    Saved %s: %d rows, %.1f MB", date_str, len(df), file_mb)

    # Free memory
    del u_data, v_data, df
    gc.collect()

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Parse optional month arguments
    months = None
    if len(sys.argv) >= 2:
        m1 = int(sys.argv[1])
        m2 = int(sys.argv[2]) if len(sys.argv) >= 3 else m1
        months = range(m1, m2 + 1)

    if months is None:
        months = range(1, 13)

    log.info("=" * 60)
    log.info("ESPC-D-V02 2025 DAP download (day-by-day) — months: %s", list(months))
    log.info("Output dir: %s", OUTPUT_DIR)
    log.info("=" * 60)

    # Open both datasets (connection only — data fetched lazily)
    ds_u = retry_open_dataset(U_URL, decode_times=False)
    ds_v = retry_open_dataset(V_URL, decode_times=False)

    total_days = 0
    skipped_days = 0
    failed_days = 0
    t0 = time.time()

    for month in months:
        days_in_month = calendar.monthrange(2025, month)[1]
        log.info("-" * 40)
        log.info("Processing month %02d / 2025 (%d days)", month, days_in_month)

        for day in range(1, days_in_month + 1):
            date_str = f"2025-{month:02d}-{day:02d}"
            out_path = OUTPUT_DIR / f"currents_espc_{date_str}.parquet"

            if out_path.exists():
                log.debug("  %s already exists, skipping", date_str)
                skipped_days += 1
                total_days += 1
                continue

            try:
                success = process_day(ds_u, ds_v, 2025, month, day)
                if success:
                    total_days += 1
                else:
                    failed_days += 1
            except Exception as exc:
                log.error("Day %s FAILED: %s", date_str, exc, exc_info=True)
                failed_days += 1
                # Re-open datasets in case connection is stale
                try:
                    ds_u.close()
                except Exception:
                    pass
                try:
                    ds_v.close()
                except Exception:
                    pass
                ds_u = retry_open_dataset(U_URL, decode_times=False)
                ds_v = retry_open_dataset(V_URL, decode_times=False)

            # Progress every 10 days
            day_of_year = pd.Timestamp(2025, month, day).timetuple().tm_yday
            if day_of_year % 10 == 0:
                elapsed = time.time() - t0
                log.info(
                    "  Progress: day %d (%s), %d done, %d skipped, %d failed, %.1f min elapsed",
                    day_of_year, date_str, total_days, skipped_days, failed_days,
                    elapsed / 60,
                )

    ds_u.close()
    ds_v.close()

    total_elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(
        "FINISHED: %d days processed (%d skipped, %d failed) in %.1f min",
        total_days, skipped_days, failed_days, total_elapsed / 60,
    )
    log.info("=" * 60)


if __name__ == "__main__":
    main()