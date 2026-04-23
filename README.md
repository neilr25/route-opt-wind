# Wind-Assisted Shipping Route Optimiser

A dynamic, weather-aware route optimiser for wind-assisted vessels using real ERA5 meteorology and ship-specific performance polars.

## What This Does

Given two ports, a voyage date, and ship speed, the engine calculates **three fuel costs**:

| Route | Wind |
|-------|------|
| **Standard** (baseline ATOBVIAC shipping lane) | No wind assist |
| **Standard** (baseline ATOBVIAC shipping lane) | With wind assist (Flettner rotors)
| **Optimised** (A* search through corridor mesh) | With wind assist + routing optimisation |

The difference between line 2 and 3 is the **extra fuel saved by deviating from the standard lane** to exploit favourable winds.

## 2025 Annual Results (Rotterdam → New York, 12 kts)

| Metric | Mean | Best Day |
|--------|------|----------|
| Standard fuel (no wind) | 206.6 t | 159.9 t |
| Standard fuel (with wind) | 187.8 t | 95.0 t |
| **Optimised fuel (with wind)** | **169.3 t** | **80.0 t** |
| Wind savings alone | 18.9 t | 70.9 t |
| **Optimisation extra savings** | **18.5 t** | **61.2 t** |
| Total savings | 37.3 t | 86.6 t |

Best optimisation day: **2025-10-03** saved **61.2 t (27.8%)**.

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/neilr25/route-opt-wind.git
cd route-opt-wind
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Place weather Parquet files in C:\app\data\ (weather_2025-01.parquet … weather_2025-12.parquet)
#    See weather-proxy-api repo for how these are generated from ERA5.

# 3. Single day CLI
python -m route_opt.main --start "ROTTERDAM" --end "NEW YORK" --speed 12 --date 2025-06-01

# 4. Start dashboard server
python -m route_opt.main --serve
# Open http://localhost:8002/ in browser

# 5. Batch run (365 days)
python batch_run_2025.py
# Output: results_2025_rotterdam_newyork.parquet
```

## Dashboard

- URL: `http://localhost:8002/`
- Enter start/end ports, date, speed
- Interactive Leaflet map shows:
  - Baseline route (red)
  - Optimised route (green)
  - Corridor mesh overlay with node/edge land flags
  - Hover over mesh edges for debug info (bearing, distance, land crossing)

## Architecture

```
main.py
  ├── api.py              # FastAPI + HTML dashboard
  ├── baseline.py         # ATOBVIAC standard lane from JSON
  ├── mesh.py             # VOIDS-style corridor grid
  ├── cost_engine.py      # Wind → TWA → Polar → Fuel
  ├── optimizer.py        # A* search through graph
  ├── weather_client.py   # DuckDB Parquet lookup (cached)
  ├── polars_loader.py    # Hadnymax-2FR35 bilinear lookup
  ├── atobviac_loader.py  # Baseline route JSON loader
  ├── config.py           # Corridor resolution knobs
  └── visualizer.py       # Plotly map generation
```

## Key Technical Details

**Weather:** ERA5 0.25° hourly via `winddata.neil.ro`, stored in monthly Parquet files.

**Polars:** Hadnymax-2FR35 performance data (columns `power_without_wing`, `power_with_wing`).

**Mesh:**
- Baseline waypoints every ~50 nm
- Stages every 200 nm (4× waypoint skip)
- ±200 nm corridor, 25 nm lateral spacing, 17 lanes
- Hard blocks on offset land nodes and land-crossing edges
- Center lane exempt (ATOBVIAC is already maritime)

**Optimisation:** NetworkX A* with fuel-aware heuristic (no turning penalty per user config).

**Performance:** ~4-5 s per day (mesh cached after first build).

## Files

| File | Purpose |
|------|---------|
| `batch_run_2025.py` | 365-day batch, saves `results_2025_rotterdam_newyork.parquet` |
| `data/Hadnymax-2FR35-polars.csv` | Ship performance polars |
| `data/atobviac_rotterdam_newyork.json` | ATOBVIAC TSS baseline route (108 waypoints) |
| `results_2025_rotterdam_newyork.parquet` | Annual batch results (365 rows) |

## Assumptions

1. Ship speed is constant (user-specified, default 12 kts).
2. Weather sampled at segment destination node.
3. Wingsail used only when it reduces fuel vs motoring without them.
4. Center-lane nodes never flagged as land (ATOBVIAC is maritime).
5. Coastal stages (first/last 3) locked to centre lane to avoid port land.
6. `global_land_mask` resolution ≈ 1° — small islands may not be flagged.

## Licence

MIT — see `SPEC.md` for full technical specification.
