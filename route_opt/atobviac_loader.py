import json
from pathlib import Path
from typing import List, Tuple

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "atobviac_rotterdam_newyork.json"


def baseline_route(start_ll: Tuple[float, float], end_ll: Tuple[float, float]) -> List[Tuple[float, float]]:
    if not _DATA_PATH.exists():
        raise FileNotFoundError(f"Route data file not found: {_DATA_PATH}")
    with open(_DATA_PATH, "r") as f:
        data = json.load(f)
    route_points = data["illustrative_route"][0]
    return [(float(p[0]), float(p[1])) for p in route_points]
