"""Batch run Rotterdam→New York for all of 2025, save to Parquet."""
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from route_opt.baseline import baseline_route
from route_opt.mesh import corridor_graph
from route_opt.optimizer import optimise

START = (51.9244, 4.4777)
END = (40.7128, -74.0060)
SPEED = 12.0
OUTPUT = Path("results_2025_rotterdam_newyork.parquet")

base = baseline_route(START, END)
print(f"Baseline: {len(base)} waypoints")
G = corridor_graph(base)
print(f"Mesh: {len(G.nodes)} nodes, {len(G.edges)} edges")

records = []
d = date(2025, 1, 1)
total_days = (date(2025, 12, 31) - d).days + 1

for day_num in range(total_days):
    date_str = d.strftime("%Y-%m-%d")
    start_t = time.time()
    try:
        path, cost_no_wind, cost_std_wind, cost_opt = optimise(
            G, base, START, END, date_str, SPEED
        )
        elapsed = time.time() - start_t
        savings_std = cost_no_wind - cost_std_wind
        savings_opt = cost_std_wind - cost_opt
        records.append({
            "date": date_str,
            "standard_no_wind_t": round(cost_no_wind, 2),
            "standard_with_wind_t": round(cost_std_wind, 2),
            "optimised_with_wind_t": round(cost_opt, 2),
            "wind_savings_vs_standard_t": round(savings_std, 2),
            "optimisation_extra_t": round(savings_opt, 2),
            "total_savings_t": round(savings_std + savings_opt, 2),
            "wind_savings_pct": round(savings_std / cost_no_wind * 100, 2) if cost_no_wind > 0 else 0,
            "optimisation_savings_pct": round(savings_opt / cost_std_wind * 100, 2) if cost_std_wind > 0 else 0,
            "path_length": len(path),
            "elapsed_s": round(elapsed, 2),
        })
        if day_num % 30 == 0:
            print(f"{date_str}: {len(records)} days done, last={cost_std_wind:.1f}t std, {cost_opt:.1f}t opt, {savings_opt:.1f}t saved ({savings_opt/cost_std_wind*100:.1f}%)")
    except Exception as exc:
        print(f"FAIL {date_str}: {exc}")
        records.append({
            "date": date_str,
            "standard_no_wind_t": None,
            "standard_with_wind_t": None,
            "optimised_with_wind_t": None,
            "wind_savings_vs_standard_t": None,
            "optimisation_extra_t": None,
            "total_savings_t": None,
            "wind_savings_pct": None,
            "optimisation_savings_pct": None,
            "path_length": None,
            "elapsed_s": None,
            "error": str(exc),
        })
    d += timedelta(days=1)

df = pd.DataFrame(records)
df.to_parquet(OUTPUT, index=False)
print(f"\nSaved {len(df)} rows to {OUTPUT}")
print(df.describe())
print(f"\nMean optimisation savings: {df['optimisation_extra_t'].mean():.1f} t ({df['optimisation_savings_pct'].mean():.1f}%)")
print(f"Mean wing savings vs standard: {df['wind_savings_vs_standard_t'].mean():.1f} t ({df['wind_savings_pct'].mean():.1f}%)")
print(f"Max optimisation savings: {df['optimisation_extra_t'].max():.1f} t on {df.loc[df['optimisation_extra_t'].idxmax(), 'date']}")
print(f"Min optimisation savings: {df['optimisation_extra_t'].min():.1f} t on {df.loc[df['optimisation_extra_t'].idxmin(), 'date']}")
