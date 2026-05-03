"""Baseline route loader — fetches maritime routes live from AtoBviaC API.

Demo key supports 6 ports:
  Chiba, Copenhagen, LOOP Terminal, Melbourne, Novorossiysk, Port Rashid

No local JSON files are used. All routes are fetched fresh from the API
at runtime (~1-2s call) and cached in memory for the session.
"""

import time
from typing import Dict, List, Optional, Tuple

import requests

# AtoBviaC demo URL (no key required for demo ports)
_ATOBVIAC_URL = "https://api.atobviac.com/v1/Voyage"

# Port coordinates (lat, lon) for name resolution
_PORT_COORDS = {
    "CHIBA":          (35.6074, 140.1065),
    "COPENHAGEN":     (55.6761, 12.5683),
    "LOOP TERMINAL":  (29.6167, -89.9167),
    "MELBOURNE":      (-37.8136, 144.9631),
    "NOVOROSSIYSK":   (44.7239, 37.7689),
    "PORT RASHID":    (25.2675, 55.2775),
}

# In-memory cache: key=(start_name,end_name), value=list of waypoints
_route_cache: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}


def _resolve_port_name(coord: Tuple[float, float]) -> Optional[str]:
    for name, (plat, plon) in _PORT_COORDS.items():
        if abs(coord[0] - plat) < 0.01 and abs(coord[1] - plon) < 0.01:
            return name
    return None


def _fetch_atobviac_route(start_name: str, end_name: str) -> List[Tuple[float, float]]:
    """Call AtoBviaC Voyage API and return list of (lat, lon) waypoints."""
    cache_key = (start_name.upper(), end_name.upper())
    if cache_key in _route_cache:
        return _route_cache[cache_key]

    url = (
        f"{_ATOBVIAC_URL}?port={start_name.replace(chr(32), chr(43))}"
        f"&port={end_name.replace(chr(32), chr(43))}&api_key=demo"
    )
    t0 = time.time()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    legs = data.get("Legs", []) if isinstance(data, dict) else []
    waypoints: List[Tuple[float, float]] = []
    for leg in legs:
        for wp in leg.get("Waypoints", []):
            lat = wp.get("LatGeodetic") if "LatGeodetic" in wp else wp.get("Lat")
            lon = wp.get("Lon") if "Lon" in wp else wp.get("Lng")
            if lat is not None and lon is not None:
                waypoints.append((float(lat), float(lon)))

    if not waypoints:
        raise ValueError(f"AtoBviaC returned empty waypoints for {start_name} -> {end_name}")

    _route_cache[cache_key] = waypoints
    print(f"  [atobviac] Fetched {start_name} -> {end_name}: {len(waypoints)} waypoints in {time.time()-t0:.1f}s")
    return waypoints


def baseline_route(start_ll: Tuple[float, float], end_ll: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Return list of (lat, lon) waypoints for the standard shipping lane.

    Fetches live from AtoBviaC API. No local files, no fallbacks.
    Only the 6 demo ports are supported.
    """
    start_name = _resolve_port_name(start_ll)
    end_name = _resolve_port_name(end_ll)

    if not start_name or not end_name:
        raise FileNotFoundError(
            f"No AtoBviaC route for {start_ll} -> {end_ll}. "
            f"Only demo ports are supported: {list(_PORT_COORDS.keys())}"
        )

    return _fetch_atobviac_route(start_name, end_name)


def clear_cache():
    """Clear in-memory route cache."""
    _route_cache.clear()


__all__ = ["baseline_route", "clear_cache"]
