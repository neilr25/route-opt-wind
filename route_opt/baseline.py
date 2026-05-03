"""Generate baseline route via ATOBVIAC precomputed JSON routes."""

import math
from typing import List, Tuple

from global_land_mask import globe


def _haversine_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(hav))


def _bearing(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _find_sea_gate(port: Tuple[float, float], max_nm: float = 20.0) -> Tuple[float, float]:
    """Walk outward from port in expanding circles until we hit the sea.
    Returns (lat, lon) of the sea gate."""
    # For ATOBVIAC routes, ports are at the start/end and ARE on land
    # but the remaining waypoints are already maritime. We only need to find
    # a sea gate for the very start/departure leg, not for every point.
    for r in (1, 2, 3, 5, 8, 12, 15, 20):
        if r > max_nm:
            break
        for theta in range(0, 360, 22):
            brng = math.radians(theta)
            dlat = r / 60.0 * math.cos(brng)
            dlon = r / (60.0 * math.cos(math.radians(port[0]))) * math.sin(brng)
            lat = port[0] + dlat
            lon = port[1] + dlon
            # Use global_land_mask but it can misclassify narrow TSS channels
            # as land. We'll look for the closest non-land point.
            if not globe.is_land(lat, lon):
                return (lat, lon)
    return port


def _is_land_strict(lat: float, lon: float) -> bool:
    """Check if a point is on land, with a stricter resolution threshold.
    
    global_land_mask can misclassify TSS lanes as land. We check a small
    radial window as well to confirm it's genuinely land, not just a narrow
    channel or port/estuary opening.
    """
    if not globe.is_land(lat, lon):
        return False
    
    # It claims land — check nearby to see if this is just a coastline
    # pixelation artifact. Look in 8 directions at ~500m (~0.0045 deg).
    for brng_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        brng = math.radians(brng_deg)
        dlat = 0.0045 / 69.0 * math.cos(brng)  # ~0.5km
        dlon = 0.0045 / (69.0 * math.cos(math.radians(lat))) * math.sin(brng)
        if not globe.is_land(lat + dlat, lon + dlon):
            return False  # coastline pixelation — not true land
    return True


def _interpolate_gaps(
    pts: List[Tuple[float, float]], max_gap_nm: float = 50.0
) -> List[Tuple[float, float]]:
    """Insert intermediate waypoints wherever consecutive points are > max_gap_nm apart."""
    if len(pts) < 2:
        return pts
    out: List[Tuple[float, float]] = [pts[0]]
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        dist = _haversine_nm(a, b)
        if dist <= max_gap_nm:
            out.append(b)
            continue
        steps = max(1, int(dist / max_gap_nm))
        for s in range(1, steps + 1):
            frac = s / (steps + 1)
            out.append((a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1])))
        out.append(b)
    return out


def baseline_route(
    start: Tuple[float, float], end: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Return list of (lat, lon) waypoints for the standard shipping lane.

    Uses the ATOBVIAC JSON loader to fetch precomputed baseline routes.
    """
    from route_opt.atobviac_loader import baseline_route as _atobviac_route

    return _atobviac_route(start, end)
