"""Wind-based cost engine using real polar power curves."""

import math
from typing import Tuple

from route_opt.polars_loader import POLAR

# Engine constants (from Mason et al. 2023, Panamax specs)
MCR_KW = 9500.0          # Maximum Continuous Rating (kW)
SFC_BASE_G_KWH = 185.0   # Specific fuel consumption at design point (g/kWh)

# Turning penalty is DISABLED by default (user requested set to 0)
# Configurable via this flag; set to True to re-enable the original
# penalty ramp (0% at ±15° to +50% at ±90°).
TURNING_PENALTY_ENABLED = False


def _twa(wind_dir_deg: float, ship_bearing_deg: float) -> float:
    """True Wind Angle (0=headwind, 180=tailwind)."""
    return abs((wind_dir_deg - ship_bearing_deg + 180) % 360 - 180)


def _fuel_rate_tph(power_kw: float) -> float:
    """Convert engine power (kW) to fuel consumption (t/h).
    
    Uses Mason et al. parabolic SFC equation:
        SFC(x) = SFCbase * (0.455*x² - 0.71*x + 1.28)
        where x = power / MCR (engine load fraction)
    """
    x = power_kw / MCR_KW
    sfc = SFC_BASE_G_KWH * (0.455 * x**2 - 0.71 * x + 1.28)
    fuel_rate_kgh = sfc * power_kw / 1000.0   # g/h → kg/h
    return fuel_rate_kgh / 1000.0              # kg/h → t/h


def fuel_without_wingsail(
    wind_speed_ms: float,
    wind_dir_deg: float,
    ship_bearing_deg: float,
    hours: float,
) -> float:
    """Fuel for a standard ship WITHOUT Flettner rotors in actual wind conditions.

    Without wingsails the ship still faces wind resistance.
    We use the 'power_without_wing' column from the polar table,
    but a simpler constant approximation is valid since the user
    just wants a comparable baseline.
    """
    twa = _twa(wind_dir_deg, ship_bearing_deg)
    ws_kts = wind_speed_ms * 1.94384
    p_no, _wi = POLAR.powers(ws_kts, twa)
    return _fuel_rate_tph(p_no) * hours


def fuel_with_wind(
    wind_speed_ms: float,
    wind_dir_deg: float,
    ship_bearing_deg: float,
    hours: float,
) -> float:
    """Fuel for a wind-assisted ship WITH Flettner rotors."""
    twa = _twa(wind_dir_deg, ship_bearing_deg)
    ws_kts = wind_speed_ms * 1.94384
    _no, p_wi = POLAR.powers(ws_kts, twa)
    return _fuel_rate_tph(p_wi) * hours


def fuel_no_wind(hours: float) -> float:
    """Constant baseline fuel for a Handymax at calm (TWS=0).

    Kept for backward compatibility when wind data is unavailable.
    """
    p_no, _wi = POLAR.powers(0.0, 180.0)
    return _fuel_rate_tph(p_no) * hours


def edge_cost(
    wind_speed_ms: float,
    wind_dir_deg: float,
    edge_bearing_deg: float,
    prev_bearing_deg: float | None,
    hours: float,
) -> float:
    """Total cost for traversing an edge.

    The wind ship only uses wingsail assist when it reduces fuel
    vs motoring through the same wind without wings.
    """
    fuel_wi = fuel_with_wind(wind_speed_ms, wind_dir_deg, edge_bearing_deg, hours)
    fuel_no = fuel_without_wingsail(wind_speed_ms, wind_dir_deg, edge_bearing_deg, hours)
    fuel = min(fuel_no, fuel_wi)
    if prev_bearing_deg is None:
        return fuel
    # No turning penalty as requested
    return fuel


def _turning_penalty(prev_bearing_deg: float, edge_bearing_deg: float) -> float:
    """Return multiplicative penalty for turning between edges.

    When TURNING_PENALTY_ENABLED is False (default per user request),
    returns 1.0 (no penalty).
    When enabled: linear ramp 1.0 (±15°) to 1.5 (±90°), >90° = 1e6.
    """
    if not TURNING_PENALTY_ENABLED:
        return 1.0
    turn = abs((edge_bearing_deg - prev_bearing_deg + 180) % 360 - 180)
    if turn <= 15.0:
        return 1.0
    if turn >= 90.0:
        return 1e6
    frac = (turn - 15.0) / 75.0
    return 1.0 + frac * 0.5