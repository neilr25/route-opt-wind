"""Wind-based cost engine using real polar power curves."""

import math

from route_opt.polars_loader import POLAR

# Engine constants
# MCR must always be sourced from the polar singleton (never hardcoded) to prevent fuel scaling bugs.
MCR_KW_DEFAULT = (POLAR.mcr_kw if POLAR.mcr_kw else 9500.0)   # Maximum Continuous Rating (kW)

# Guardrail: verify MCR is correct at module load time
if MCR_KW_DEFAULT != 5670.0:
    raise RuntimeError(f"MCR_KW_DEFAULT={MCR_KW_DEFAULT} but expected 5670.0 from polar singleton. "
                       f"POLAR.mcr_kw={POLAR.mcr_kw}. "
                       f"This means the polar CSV was not loaded correctly.")
SFC_BASE_G_KWH = 185.0   # Specific fuel consumption at design point (g/kWh)
SFC_X_MAX = 1.5             # Clamp load fraction to prevent SFC explosion at overload

# Turning penalty is DISABLED by default (user requested set to 0)
# Configurable via this flag; set to True to re-enable the original
# penalty ramp (0% at ±15° to +50% at ±90°).
TURNING_PENALTY_ENABLED = False


def _twa(wind_dir_deg: float, ship_bearing_deg: float) -> float:
    """True Wind Angle (0=headwind, 180=tailwind)."""
    return abs((wind_dir_deg - ship_bearing_deg + 180) % 360 - 180)


def _fuel_rate_tph(power_kw: float, mcr_kw: float = MCR_KW_DEFAULT) -> float:
    """Convert engine power (kW) to fuel consumption (t/h).
    
    Uses Mason et al. parabolic SFC equation:
        SFC(x) = SFCbase * (0.455*x² - 0.71*x + 1.28)
        where x = power / MCR (engine load fraction), clamped at SFC_X_MAX
    """
    x = min(power_kw / mcr_kw, SFC_X_MAX) if mcr_kw > 0 else 0.0
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


def _sog(stw_mps: float, current_u_ms: float, current_v_ms: float, bearing_deg: float) -> float:
    """Speed over ground (m/s) from speed through water + current vector.
    
    Physics: Ground velocity = Ship velocity (through water) + Current velocity
    NaN currents (land cells) are treated as zero.
    """
    if math.isnan(current_u_ms) or math.isnan(current_v_ms):
        current_u_ms = 0.0
        current_v_ms = 0.0
    # Convert STW to u,v components along bearing
    ship_u = stw_mps * math.sin(math.radians(bearing_deg))
    ship_v = stw_mps * math.cos(math.radians(bearing_deg))
    
    # Add current (current is in u,v east/north)
    sog_u = ship_u + current_u_ms
    sog_v = ship_v + current_v_ms
    
    # Ground speed magnitude
    return math.sqrt(sog_u**2 + sog_v**2)


def edge_cost(
    wind_speed_ms: float,
    wind_dir_deg: float,
    edge_bearing_deg: float,
    prev_bearing_deg: float | None,
    distance_nm: float,
    ship_speed_kts: float = 12.0,
    current_u_ms: float = 0.0,
    current_v_ms: float = 0.0,
) -> float:
    """Total cost for traversing an edge.
    
    Accounts for wind and ocean current effects on fuel and time.
    
    Uses speed-over-ground (SOG) for time calculation:
        SOG = STW + current projection along bearing
        hours = distance_nm / SOG_in_knots
    
    If no current data is provided, falls back to wind-only calculation
    (identical to previous behavior when current_u_ms=current_v_ms=0).
    """
    # Convert ship speed to m/s
    stw_mps = ship_speed_kts * 0.514444
    
    # Calculate speed over ground
    sog_mps = _sog(stw_mps, current_u_ms, current_v_ms, edge_bearing_deg)
    sog_kts = sog_mps / 0.514444
    
    # If current grounds the ship (rare), use STW as fallback
    if sog_kts <= 0:
        sog_kts = ship_speed_kts
    
    # Time at SOG
    hours = distance_nm / sog_kts
    
    fuel_wi = fuel_with_wind(wind_speed_ms, wind_dir_deg, edge_bearing_deg, hours)
    fuel_no = fuel_without_wingsail(wind_speed_ms, wind_dir_deg, edge_bearing_deg, hours)
    fuel = min(fuel_no, fuel_wi)
    
    if prev_bearing_deg is None:
        return fuel
    return fuel * _turning_penalty(prev_bearing_deg, edge_bearing_deg)


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