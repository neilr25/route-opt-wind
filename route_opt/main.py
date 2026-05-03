"""CLI entrypoint for wind-aware shipping route optimiser."""

import argparse
import sys
import json

import route_opt.config
from route_opt.baseline import baseline_route
from route_opt.mesh import corridor_graph
from route_opt.optimizer import optimise
from route_opt.visualizer import plot_routes, print_savings


def parse_location(loc_str):
    """Parse location string: either port name or lat,lon coordinates.

    Args:
        loc_str: Port name (e.g., 'COPENHAGEN') or 'lat,lon' string.

    Returns:
        Tuple (lat, lon) as floats.

    Raises:
        ValueError: If location format is invalid or port name not found.
    """
    port_names = {
        # Demo ports (AtoBviaC demo key supports these 6 only)
        "CHIBA": (35.6074, 140.1065),
        "COPENHAGEN": (55.6761, 12.5683),
        "LOOP TERMINAL": (29.6167, -89.9167),
        "MELBOURNE": (-37.8136, 144.9631),
        "NOVOROSSIYSK": (44.7239, 37.7689),
        "PORT RASHID": (25.2675, 55.2775),
    }

    loc_upper = loc_str.upper().strip().replace("_", " ")
    if loc_upper in port_names:
        return port_names[loc_upper]

    # Try parsing as lat,lon
    try:
        parts = loc_str.split(",")
        if len(parts) == 2:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return (lat, lon)
    except ValueError:
        pass

    raise ValueError(
        f"Invalid location '{loc_str}'. Must be a port name ({', '.join(port_names.keys())}) "
        f"or lat,lon coordinates (e.g., '51.9224,4.4792')."
    )


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Wind-aware shipping route optimiser"
    )
    parser.add_argument(
        "--start",
        required=False,
        help="Starting port or lat,lon coordinates",
    )
    parser.add_argument(
        "--end",
        required=False,
        help="Destination port or lat,lon coordinates",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=12.0,
        help="Vessel speed in knots (default: 12)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2025-06-01",
        help="Date in YYYY-MM-DD format (default: 2025-06-01)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Generate visualisation HTML",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for visualisation HTML",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server on port 8002",
    )

    args = parser.parse_args()

    # Server mode
    if args.serve:
        import uvicorn
        from route_opt.api import app

        uvicorn.run(app, host="0.0.0.0", port=8002)
        return

    # Parse locations
    try:
        start_coords = parse_location(args.start)
        end_coords = parse_location(args.end)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build baseline route
    baseline = baseline_route(start_coords, end_coords)

    # Build corridor graph
    G = corridor_graph(baseline)

    # Optimise route
    result_tuple = optimise(
        G, baseline, start_coords, end_coords, args.date, args.speed
    )
    path, cost_std_no_wind, cost_std_wind, cost_opt, baseline_dist_nm, opt_dist_nm, _ = result_tuple

    # Print savings
    print_savings(cost_std_no_wind, cost_std_wind, cost_opt, baseline_dist_nm, opt_dist_nm, args.speed)

    # Generate visualisation if requested
    if args.viz:
        out_path = args.out if args.out else "route_comparison.html"
        fig = plot_routes(baseline, path, G=G, title=f"Route: {args.start} -> {args.end}", out_html=out_path)
        print(f"Visualisation saved to {out_path}")



if __name__ == "__main__":
    main()
