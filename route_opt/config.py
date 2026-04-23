"""Configuration for SGS route-opt engine."""

from pathlib import Path

DATA_DIR = Path(r"C:\app\data")
WEATHER_DB = DATA_DIR / "staging.db"
POLARS_PATH = Path(__file__).resolve().parent.parent / "polars.xlsx"

# Corridor definition (nm)
CORRIDOR_WIDTH_NM = 200      # ±200 nm each side of baseline
LANE_SPACING_NM = 25          # lateral spacing between lanes  (was 50; increased per user request)
STAGE_SKIP = 4                # connect every 4th baseline waypoint (~200 nm stages)
MAX_LANE_CHANGE = 1           # max lane steps per stage

SHIP_SPEED_KTS = 12

# A* tuning
LAND_COST_PENALTY = 1e9