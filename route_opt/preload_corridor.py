"""Preload all months for a specific route corridor into RAM on startup.

Usage:
    from route_opt.preload_corridor import preload_2025_corridor
    preload_2025_corridor([(51.92, 4.48), (40.71, -74.01)])
"""
import time
from typing import List, Tuple

from route_opt.hourly_weather import _build_monthly_cache


def preload_2025_corridor(corridor_points: List[Tuple[float, float]], padding_deg: float = 2.0) -> None:
    """
    Preload all 12 months of ERA5 data covering the given corridor.

    Building a bounding box around the corridor points so that any
    intermediate route point will hit the cache.
    """
    lats = [p[0] for p in corridor_points]
    lons = [p[1] for p in corridor_points]
    lat_min, lat_max = min(lats) - padding_deg, max(lats) + padding_deg
    lon_min, lon_max = min(lons) - padding_deg, max(lons) + padding_deg

    # Create a dense grid covering the corridor (0.25° spacing like ERA5)
    sample_points = []
    lat = lat_min
    while lat <a= lat_max:
        lon = lon_min
        while lon <a= lon_max:
            sample_points.append((round(lat, 2), round(lon, 2)))
            lon += 0.25
        lat += 0.25

    print(f"[preload_corridor] Corridor area: {lat_min:.1f}-{lat_max:.1f}N, {lon_min:.1f}-{lon_max:.1f}E")
    print(f"                  Total preload points: {len(sample_points)}")

    t0 = time.time()
    year = 2025
    loaded = 0
    for month in range(1, 13):
        try:
            _build_monthly_cache(year, month, sample_points)
            loaded += 1
        except FileNotFoundError:
            print(f"  [{month:02d}/12] Missing data file — skipping")
        except Exception as e:
            print(f"  [{month:02d}/12] ERROR: {e}")

    elapsed = time.time() - t0
    print(f"[preload_corridor] Preloaded {loaded}/12 months in {elapsed:.1f}s")
    print(f"                   Ready for requests — all subsequent queries will be warm cache")
