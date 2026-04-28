"""Batch route optimisation for an entire year (or any date range).

Example:
    python batch_runner.py --start ROTTERDAM --end NEW YORK \
        --from-date 2025-01-01 --to-date 2025-01-31 \
        --speed 12 --out results.parquet
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from route_opt.baseline import baseline_route
from route_opt.mesh import corridor_graph
from route_opt.optimizer import optimise
from route_opt.weather_client import _weather_cache


def daterange(start: date, end: date):
    """Yield dates from start to end inclusive."""
    for n in range(int((end - start).days) + 1):
        yield start + timedelta(days=n)


def run_batch(
    start_ll: Tuple[float, float],
    end_ll: Tuple[float, float],
    from_date: date,
    to_date: date,
    speed_kts: float,
    calm_rate: float = 2.5,
) -> List[dict]:
    """
    Run optimisation for every day in range.
    Baseline route and corridor graph are built once then cached.
    Weather cache is cleared between months to keep memory bounded.
    """
    # Build once
    baseline = baseline_route(start_ll, end_ll)
    G = corridor_graph(baseline)
    graph_nodes = list(G.nodes(data=True))

    results: List[dict] = []
    prev_month = None

    total_start = time.time()
    for i, voyage_date in enumerate(daterange(from_date, to_date), 1):
        date_str = voyage_date.isoformat()
        print(f"[{i:4d}/{((to_date - from_date).days + 1)}] {date_str} ...", end=" ", flush=True)

        t0 = time.time()
        try:
            result_tuple = optimise(
                G, baseline, start_ll, end_ll, date_str, speed_kts
            )
            path, cost_std_no_wind, cost_std_wind, cost_opt, baseline_dist_nm, opt_dist_nm, _ = result_tuple
            dur = time.time() - t0
            results.append({
                "date": date_str,
                "standard_no_wind_t": round(cost_std_no_wind, 2),
                "standard_with_wind_t": round(cost_std_wind, 2),
                "optimised_with_wind_t": round(cost_opt, 2),
                "wind_savings_vs_standard": round(cost_std_no_wind - cost_std_wind, 2),
                "optimisation_extra_t": round(cost_std_wind - cost_opt, 2),
                "total_savings_t": round(cost_std_no_wind - cost_opt, 2),
                "path_length": len(path),
                "elapsed_s": round(dur, 2),
            })
            print(f"OK  ({dur:.1f}s)  std={cost_std_no_wind:.1f}t  opt={cost_opt:.1f}t")
        except Exception as exc:
            dur = time.time() - t0
            print(f"FAIL ({dur:.1f}s)  {exc}")
            results.append({
                "date": date_str,
                "standard_no_wind_t": None,
                "standard_with_wind_t": None,
                "optimised_with_wind_t": None,
                "wind_savings_vs_standard": None,
                "optimisation_extra_t": None,
                "total_savings_t": None,
                "path_length": None,
                "elapsed_s": round(dur, 2),
                "error": str(exc),
            })

        # Flush weather cache on month boundary to keep memory bounded
        current_month = date_str[:7]
        if prev_month and current_month != prev_month:
            _weather_cache.clear()
        prev_month = current_month

    total_dur = time.time() - total_start
    print(f"\nDone. {len(results)} days in {total_dur:.1f}s ({total_dur/len(results):.1f}s/day)")
    return results


def save(results: List[dict], out_path: Path) -> None:
    df = pd.DataFrame(results)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        # default to parquet
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    print(f"Results saved to {out_path}")


def parse_location(loc_str: str) -> Tuple[float, float]:
    port_names = {
        "ROTTERDAM": (51.9244, 4.4777),
        "NEW YORK": (40.7128, -74.0060),
        "SINGAPORE": (1.3521, 103.8198),
        "SHANGHAI": (31.2304, 121.4737),
        "LOS ANGELES": (33.7362, -118.2922),
        "HAMBURG": (53.5488, 9.9872),
        "VALENCIA": (39.4699, -0.3763),
        "TOKYO": (35.6762, 139.6503),
        "BUSAN": (35.1145, 129.0403),
        "ALGECIRAS": (36.1333, -5.4500),
    }
    key = loc_str.upper().strip().replace("_", " ")
    if key in port_names:
        return port_names[key]
    parts = loc_str.split(",")
    return (float(parts[0]), float(parts[1]))


def main():
    parser = argparse.ArgumentParser(description="Batch route optimisation")
    parser.add_argument("--start", required=True, help="Start port or lat,lon")
    parser.add_argument("--end", required=True, help="End port or lat,lon")
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--speed", type=float, default=12.0, help="Ship speed kts")
    parser.add_argument("--calm-rate", type=float, default=2.5, help="Fuel tph at calm")
    parser.add_argument("--out", default="batch_results.parquet", help="Output path (.parquet or .csv)")
    args = parser.parse_args()

    start_ll = parse_location(args.start)
    end_ll = parse_location(args.end)
    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)

    results = run_batch(start_ll, end_ll, from_date, to_date, args.speed, args.calm_rate)
    save(results, Path(args.out))


if __name__ == "__main__":
    main()