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

## 2025 Annual Results (Copenhagen → LOOP Terminal, 12 kts)

| Metric | Mean | Best Day |
|--------|------|----------|
| Standard fuel (no wind) | 196.4 t | 177.8 t |
| Standard fuel (with wind) | 180.8 t | 169.8 t |
| **Optimised fuel (with wind)** | **158.1 t** | **150.3 t** |
| Wind savings alone | 15.6 t | 18.5 t |
| **Optimisation extra savings** | **22.7 t** | **27.5 t** |
| Total savings | 38.3 t | 46.0 t |

Best optimisation day: **2025-01-07** saved **27.2% total** (wind 10.4% + opt 20.9%).

## Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/neilr25/route-opt-wind.git
cd route-opt-wind
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Place ERA5 hourly Parquet files in C:\app\data\ (weather_2025-01_hourly.parquet …)
#    + unified current files (unified_*.parquet) for ocean current data.
#    See weather-proxy-api repo for how these are generated from ERA5.

# 3. Single day CLI (demo ports only: CHIBA, COPENHAGEN, LOOP TERMINAL, MELBOURNE, NOVOROSSIYSK, PORT RASHID)
python -m route_opt.main --start "COPENHAGEN" --end "LOOP TERMINAL" --speed 12 --date 2025-01-15

# 4. Start dashboard server
python -m route_opt.main --serve
# Open http://localhost:8002/ in browser

# 5. Batch run (30 days, demo routes)
python batch_demo_fast.py
# Output: results_demo_fast.parquet
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
  ├── hourly_weather.py   # ERA5 hourly Parquet lookup (cached + disk cache)
  ├── unified_weather.py  # Wind + current unified lookup (compact numpy cache)
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
| `batch_demo_fast.py` | 30-day batch, saves `results_demo_fast.parquet` |
| `data/Hadnymax-2FR35-polars.csv` | Ship performance polars |
| `data/atobviac_copenhagen_loopterminal.json` | ATOBVIAC TSS baseline route (25 waypoints) |
| `results_demo_fast.parquet` | Monthly batch results (~150 rows) |

## Assumptions

1. Ship speed is constant (user-specified, default 12 kts).
2. Weather sampled at segment destination node.
3. Wingsail used only when it reduces fuel vs motoring without them.
4. Center-lane nodes never flagged as land (ATOBVIAC is maritime).
5. Coastal stages (first/last 3) locked to centre lane to avoid port land.
6. `global_land_mask` resolution ≈ 1° — small islands may not be flagged.

## Licence

MIT — see `SPEC.md` for full technical specification.
