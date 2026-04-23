"""Load ship performance polars from CSV (Hadnymax-2FR35)."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# CSV path — shipped in repo data/ folder
_POLARS_CSV = Path(__file__).resolve().parent.parent / "data" / "Hadnymax-2FR35-polars.csv"


def _build_power_lookup(df: pd.DataFrame) -> Dict[float, Dict[float, Tuple[float, float]]]:
    """
    Build nested dict:
        {tws_ms: {twa_deg: (power_without_wing_kw, power_with_wing_kw)}}
    """
    table: Dict[float, Dict[float, Tuple[float, float]]] = {}
    for _, row in df.iterrows():
        tws = float(row["wind_speed"])
        twa = float(row["wind_direction"])
        p_no = float(row["power_without_wing"])
        p_wing = float(row["power_with_wing"])
        table.setdefault(tws, {})[twa] = (p_no, p_wing)
    return table


class PolarLoader:
    """Bilinear lookup for engine power (kW) with/without Flettner rotors."""

    def __init__(self, csv_path: Path = _POLARS_CSV):
        self._table = {}
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Trim whitespace in column names
            df.columns = [c.strip() for c in df.columns]
            self._table = _build_power_lookup(df)
        else:
            raise FileNotFoundError(f"Polar CSV not found: {csv_path}")

    def powers(self, tws_ms: float, twa_deg: float) -> Tuple[float, float]:
        """
        Return (power_without_wing_kw, power_with_wing_kw) for given TWS (m/s) and TWA (deg).
        Bilin interpolation on (tws, twa).
        Symmetric around TWA 0° (data only covers 0–180°).
        """
        # Normalise TWA to 0–180° (symmetrical)
        twa_norm = abs(twa_deg) % 360
        if twa_norm > 180:
            twa_norm = 360 - twa_norm

        tws_vals = sorted(self._table.keys())
        if not tws_vals:
            return (0.0, 0.0)

        # Clamp or interpolate TWS
        if tws_ms <= tws_vals[0]:
            tws_lo = tws_hi = tws_vals[0]
        elif tws_ms >= tws_vals[-1]:
            tws_lo = tws_hi = tws_vals[-1]
        else:
            tws_hi = next(v for v in tws_vals if v >= tws_ms)
            tws_lo = tws_vals[tws_vals.index(tws_hi) - 1]

        def _interp(tws: float) -> Tuple[float, float]:
            row = self._table[tws]
            twa_keys = sorted(row.keys())
            if twa_norm <= twa_keys[0]:
                return row[twa_keys[0]]
            if twa_norm >= twa_keys[-1]:
                return row[twa_keys[-1]]
            hi = next(k for k in twa_keys if k >= twa_norm)
            lo = twa_keys[twa_keys.index(hi) - 1]
            frac = (twa_norm - lo) / (hi - lo)
            p_no_lo, p_wi_lo = row[lo]
            p_no_hi, p_wi_hi = row[hi]
            return (
                p_no_lo + frac * (p_no_hi - p_no_lo),
                p_wi_lo + frac * (p_wi_hi - p_wi_lo),
            )

        if tws_lo == tws_hi:
            return _interp(tws_lo)
        frac = (tws_ms - tws_lo) / (tws_hi - tws_lo)
        p1 = _interp(tws_lo)
        p2 = _interp(tws_hi)
        return (
            p1[0] + frac * (p2[0] - p1[0]),
            p1[1] + frac * (p2[1] - p1[1]),
        )


# Global singleton
POLAR = PolarLoader()
