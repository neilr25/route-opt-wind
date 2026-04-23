"""Batch run Rotterdam to New York for all of 2025, save to Parquet.

Uses weather pre-loading for ~13s startup then ~0.02s per day.
Total: ~20s for 365 days (vs 27 min previously).
"""
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from route_opt.baseline import baseline_route
from route_opt.mesh import corridor_graph
from route_opt.optimizer import optimise
from route_opt.weather_client import preload_year

START = (51.9244, 4.4777)
END = (40.7128, -74.0060)
SPEED = 12.0
YEAR = 2025
OUTPUT = Path("results_2025_rotterdam_newyork.parquet")


def main():
    print(f"=== Batch Run {YEAR} ===")
    t_start = time.time()

    # Build mesh
    base = baseline_route(START, END)
    print(f"Baseline: {len(base)} waypoints")
    G = corridor_graph(base)
    print(f"Mesh: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Pre-load weather for the entire year
    mesh_points = [(d["lat"], d["lon"]) for _, d in G.nodes(data=True)]
    all_points = list(set(mesh_points + list(base)))
    print(f"Pre-loading {YEAR} weather for {len(all_points)} unique points...")
    t_preload = time.time()
    preload_year(YEAR, sample_points=all_points)
    print(f"Weather pre-load: {time.time() - t_preload:.1f}s")

    # Run optimisation for each day
    d = date(YEAR, 1, 1)
    total_days = (date(YEAR, 12, 31) - d).days + 1
    records = []

    for day_num in range(total_days):
        date_str = d.strftime("%Y-%m-%d")
        t0 = time.time()
        try:
            path, cost_no_wind, cost_std_wind, cost_opt, baseline_dist_nm, opt_dist_nm = optimise(
                G, base, START, END, date_str, SPEED
            )
            elapsed = time.time() - t0
            savings_std = cost_no_wind - cost_std_wind
            savings_opt = cost_std_wind - cost_opt
            detour_pct = ((opt_dist_nm - baseline_dist_nm) / baseline_dist_nm * 100) if baseline_dist_nm > 0 else 0
            detour_hours = (opt_dist_nm - baseline_dist_nm) / SPEED if opt_dist_nm > baseline_dist_nm else 0
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
                "baseline_distance_nm": round(baseline_dist_nm, 1),
                "optimised_distance_nm": round(opt_dist_nm, 1),
                "detour_pct": round(detour_pct, 1),
                "detour_hours": round(detour_hours, 1),
                "path_length": len(path),
                "elapsed_s": round(elapsed, 2),
            })
            if day_num % 30 == 0:
                print(f"  {date_str}: {day_num+1}/{total_days} done, "
                      f"std={cost_std_wind:.1f}t, opt={cost_opt:.1f}t, "
                      f"savings={savings_opt:.1f}t ({savings_opt/cost_std_wind*100:.1f}%), "
                      f"{elapsed:.3f}s")
        except Exception as exc:
            print(f"  FAIL {date_str}: {exc}")
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

    total_elapsed = time.time() - t_start

    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT, index=False)

    print(f"\n{'='*60}")
    print(f"Total wall-clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Per-day average: {total_elapsed/total_days:.3f}s")
    print(f"Saved {len(df)} rows to {OUTPUT}")
    print(f"\n{'='*60}")
    print(df.describe())
    ok = df.dropna(subset=["optimisation_extra_t"])
    print(f"\nMean optimisation savings: {ok['optimisation_extra_t'].mean():.1f}t "
          f"({ok['optimisation_savings_pct'].mean():.1f}%)")
    print(f"Mean wing savings vs standard: {ok['wind_savings_vs_standard_t'].mean():.1f}t "
          f"({ok['wind_savings_pct'].mean():.1f}%)")
    print(f"Max optimisation savings: {ok['optimisation_extra_t'].max():.1f}t "
          f"on {ok.loc[ok['optimisation_extra_t'].idxmax(), 'date']}")
    print(f"Min optimisation savings: {ok['optimisation_extra_t'].min():.1f}t "
          f"on {ok.loc[ok['optimisation_extra_t'].idxmin(), 'date']}")


if __name__ == "__main__":
    main()