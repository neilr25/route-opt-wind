import json
import math
from pathlib import Path
from typing import List, Tuple

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _haversine_nm(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 3440.065
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(hav))


def _route_file_name(start: Tuple[float, float], end: Tuple[float, float]) -> str:
    """Create a deterministic filename for a route."""
    s = f"{start[0]:.4f}_{start[1]:.4f}"
    e = f"{end[0]:.4f}_{end[1]:.4f}"
    return f"atobviac_{s}_to_{e}.json"


def baseline_route(start_ll: Tuple[float, float], end_ll: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Return list of (lat, lon) waypoints for the standard shipping lane.

    ONLY uses precomputed ATOBVIAC JSON files. No fallback, no searoute,
    no great-circle approximation. If no JSON exists, raises FileNotFoundError.
    """
    # Try route-specific JSON first
    route_file = _DATA_DIR / _route_file_name(start_ll, end_ll)
    if route_file.exists():
        with open(route_file, "r") as f:
            data = json.load(f)
        route_points = data["illustrative_route"][0]
        return [(float(p[0]), float(p[1])) for p in route_points]

    # Try the reverse-direction file (start/end swapped)
    reverse_file = _DATA_DIR / _route_file_name(end_ll, start_ll)
    if reverse_file.exists():
        with open(reverse_file, "r") as f:
            data = json.load(f)
        route_points = data["illustrative_route"][0]
        # Return reversed so it goes from start to end
        return [(float(p[0]), float(p[1])) for p in reversed(route_points)]

    # Try the legacy Rotterdam->New York file (only if coordinates match)
    legacy_path = _DATA_DIR / "atobviac_rotterdam_newyork.json"
    if legacy_path.exists():
        try:
            with open(legacy_path, "r") as f:
                data = json.load(f)
            route_points = data["illustrative_route"][0]
            pts = [(float(p[0]), float(p[1])) for p in route_points]
            # Check if start/end roughly match (within 5 nm)
            if _haversine_nm(start_ll, pts[0]) < 5 and _haversine_nm(end_ll, pts[-1]) < 5:
                return pts
        except Exception:
            pass

    raise FileNotFoundError(
        f"No ATOBVIAC baseline route file found for {start_ll} -> {end_ll}. "
        f"Expected: {route_file} or {reverse_file}. "
        f"Only ATOBVIAC precomputed routes are supported."
    )
