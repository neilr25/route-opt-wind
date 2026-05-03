"""Multi-route batch test: 30 days (January 2025), demo routes, full metrics.

Collects:
- Fuel tonnes (std no wind, std with wind, optimised)
- Savings % (wind alone, optimisation extra, total)
- Route distances and detour
- Per-day elapsed time
- Per-route aggregate stats (mean, max savings, best day)

Output: results_multi_route_jan2025.parquet
"""
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from route_opt.baseline import baseline_route
from route_opt.mesh import corridor_graph
from route_opt.optimizer import optimise
from route_opt.unified_weather import _get_cache as _get_unified_cache
from route_opt.hourly_weather import _get_monthly_cache
from route_opt.api_helpers import _named_port

# ── Config ──────────────────────────────────────────────────────────
# Demo ports (frontend dropdown, AtoBviaC demo key supports these 6)
DEMO_PORTS = {
    "CHIBA": (35.6074, 140.1065),
    "COPENHAGEN": (55.6761, 12.5683),
    "LOOP TERMINAL": (29.6167, -89.9167),
    "MELBOURNE": (-37.8136, 144.9631),
    "NOVOROSSIYSK": (44.7239, 37.7689),
    "PORT RASHID": (25.2675, 55.2775),
}

# Only demo ports (AtoBviaC demo key supports these 6)
ROUTES: List[Tuple[str, str]] = [
    ("LOOP TERMINAL", "COPENHAGEN"),
    ("COPENHAGEN", "LOOP TERMINAL"),
    ("PORT RASHID", "NOVOROSSIYSK"),
    ("NOVOROSSIYSK", "PORT RASHID"),
    ("PORT RASHID", "COPENHAGEN"),
]

YEAR = 2025
MONTH = 1  # January
SPEED = 12.0
OUTPUT = Path("results_multi_route_jan2025.parquet")


def _months_in_range(year: int, month: int) -> List[Tuple[int, int]]:
    """Single month for this test."""
    return [(year, month)]


def run_route_batch(start_name: str, end_name: str) -> List[dict]:
    """Run 30-day batch for one route pair. Returns list of daily records."""
    start_ll = _named_port(start_name)
    end_ll = _named_port(end_name)

    print(f"\n=== {start_name} -> {end_name} ===")
    t0 = time.time()

    # Build mesh once
    base = baseline_route(start_ll, end_ll)
    print(f"  Baseline: {len(base)} waypoints")
    G = corridor_graph(base)
    print(f"  Mesh: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Pre-load weather caches for the month
    mesh_points = [(d["lat"], d["lon"]) for _, d in G.nodes(data=True)]
    all_points = list(set(mesh_points + list(base)))
    print(f"  Pre-loading weather for {len(all_points)} points...")
    t_w = time.time()
    for y, m in _months_in_range(YEAR, MONTH):
        _get_monthly_cache(y, m, all_points)
        _get_unified_cache(y, m, all_points)
    print(f"  Weather preload: {time.time() - t_w:.1f}s")

    # Run each day of June
    from_date = date(YEAR, MONTH, 1)
    to_date = date(YEAR, MONTH, 30)
    records = []

    for d in _daterange(from_date, to_date):
        date_str = d.strftime("%Y-%m-%d")
        t_day = time.time()
        try:
            path, cost_no_wind, cost_std_wind, cost_opt, baseline_dist, opt_dist, _, _ = optimise(
                G, base, start_ll, end_ll, date_str, SPEED
            )
            elapsed = time.time() - t_day

            savings_std = cost_no_wind - cost_std_wind
            savings_opt = cost_std_wind - cost_opt
            total_savings = savings_std + savings_opt
            wind_pct = (savings_std / cost_no_wind * 100) if cost_no_wind > 0 else 0
            opt_pct = (savings_opt / cost_std_wind * 100) if cost_std_wind > 0 else 0
            total_pct = (total_savings / cost_no_wind * 100) if cost_no_wind > 0 else 0
            detour_pct = ((opt_dist - baseline_dist) / baseline_dist * 100) if baseline_dist > 0 else 0

            records.append({
                "route": f"{start_name}->{end_name}",
                "date": date_str,
                "standard_no_wind_t": round(cost_no_wind, 2),
                "standard_with_wind_t": round(cost_std_wind, 2),
                "optimised_with_wind_t": round(cost_opt, 2),
                "wind_savings_t": round(savings_std, 2),
                "optimisation_extra_t": round(savings_opt, 2),
                "total_savings_t": round(total_savings, 2),
                "wind_savings_pct": round(wind_pct, 2),
                "optimisation_savings_pct": round(opt_pct, 2),
                "total_savings_pct": round(total_pct, 2),
                "baseline_distance_nm": round(baseline_dist, 1),
                "optimised_distance_nm": round(opt_dist, 1),
                "detour_pct": round(detour_pct, 1),
                "path_length": len(path),
                "elapsed_s": round(elapsed, 3),
            })
        except Exception as exc:
            print(f"  ERROR {date_str}: {exc}")
            records.append({
                "route": f"{start_name}->{end_name}",
                "date": date_str,
                "error": str(exc),
                "elapsed_s": round(time.time() - t_day, 3),
            })

    total_elapsed = time.time() - t0
    print(f"  Route done: {len(records)} days in {total_elapsed:.1f}s")
    return records


def _daterange(start: date, end: date):
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(days=n)


def main():
    print("=" * 60)
    print("MULTI-ROUTE BATCH: June 2025")
    print(f"Routes: {len(ROUTES)}")
    print(f"Days per route: 30")
    print(f"Total runs: {len(ROUTES) * 30}")
    print("=" * 60)

    all_records = []
    t_start = time.time()

    for start_name, end_name in ROUTES:
        records = run_route_batch(start_name, end_name)
        all_records.extend(records)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ALL ROUTES COMPLETE: {len(all_records)} records in {total_time:.1f}s")
    print(f"{'='*60}")

    # Save
    df = pd.DataFrame(all_records)
    df.to_parquet(OUTPUT)
    print(f"Saved: {OUTPUT.resolve()}")

    # Summary statistics
    print("\n--- SUMMARY BY ROUTE ---")
    for route, grp in df.groupby("route"):
        if "error" in grp.columns:
            grp = grp[grp["error"].isna()]
        if len(grp) == 0:
            print(f"  {route}: ALL ERRORS")
            continue

        print(f"\n  {route} ({len(grp)} days)")
        print(f"    Std no wind:  {grp['standard_no_wind_t'].mean():.1f} ± {grp['standard_no_wind_t'].std():.1f} t")
        print(f"    Std with wind:{grp['standard_with_wind_t'].mean():.1f} ± {grp['standard_with_wind_t'].std():.1f} t")
        print(f"    Optimised:    {grp['optimised_with_wind_t'].mean():.1f} ± {grp['optimised_with_wind_t'].std():.1f} t")
        print(f"    Wind savings: {grp['wind_savings_pct'].mean():.1f}% (max {grp['wind_savings_pct'].max():.1f}%)")
        print(f"    Opt savings:  {grp['optimisation_savings_pct'].mean():.1f}% (max {grp['optimisation_savings_pct'].max():.1f}%)")
        print(f"    TOTAL savings:{grp['total_savings_pct'].mean():.1f}% (max {grp['total_savings_pct'].max():.1f}%)")
        print(f"    Best day:     {grp.loc[grp['total_savings_pct'].idxmax(), 'date']} ({grp['total_savings_pct'].max():.1f}%)")
        print(f"    Avg time/day: {grp['elapsed_s'].mean():.2f}s (max {grp['elapsed_s'].max():.2f}s)")

    # Top 10 days across all routes
    print("\n--- TOP 10 SAVINGS DAYS (all routes) ---")
    top10 = df.nlargest(10, "total_savings_pct")[["route", "date", "total_savings_pct", "wind_savings_pct", "optimisation_savings_pct"]]
    for _, row in top10.iterrows():
        print(f"  {row['route']} {row['date']}: {row['total_savings_pct']:.1f}% total (wind {row['wind_savings_pct']:.1f}%, opt {row['optimisation_savings_pct']:.1f}%)")


if __name__ == "__main__":
    main()
